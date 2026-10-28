"""Shared paths, data access, ladder features, detector and paired bootstrap (Section 4)."""

import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.ndimage import binary_dilation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("NSDOF_DATA_DIR", ROOT / "data"))
OUTPUTS_DIR = Path(os.environ.get("NSDOF_OUTPUTS_DIR", ROOT / "outputs"))
RESULTS_DIR = Path(os.environ.get("NSDOF_RESULTS_DIR", ROOT / "results"))
SPLITS_DIR = Path(os.environ.get("NSDOF_SPLITS_DIR", ROOT / "splits"))
REV = RESULTS_DIR / "revision" / "csv"
REV_FIG = RESULTS_DIR / "revision" / "figures"
HSI_Q1 = OUTPUTS_DIR / "hsi_outputs_q1"
HSI_REV = OUTPUTS_DIR / "hsi_outputs_revision"
WEIGHTS_REV = OUTPUTS_DIR / "weights_revision"

SCENES = {
    "pavia_u": ("PaviaU.mat", "PaviaU_gt.mat"),
    "salinas": ("Salinas_corrected.mat", "Salinas_gt.mat"),
}
SCENE_LIST = ["pavia_u", "salinas"]
BLOCK, FRAC, BUFFER = 16, 0.15, 6

N_BOOTSTRAP, BOOT_SEED = 3000, 1337
LEN_BALANCED_MAX_REL = 0.25  # length filter |l_c - l_r| / max(l_c, l_r) <= 0.25


# data
def load_scene(name):
    """(H, W, bands) float32 cube and (H, W) int32 ground truth (0 = unlabelled)."""
    xf, yf = SCENES[name]
    X = sio.loadmat(DATA_DIR / "hsi" / xf)
    Y = sio.loadmat(DATA_DIR / "hsi" / yf)
    xk = [k for k in X if not k.startswith("__")][0]
    yk = [k for k in Y if not k.startswith("__")][0]
    return X[xk].astype(np.float32), Y[yk].astype(np.int32)


def split_block_buffer(gt, frac, rng, block=16, buffer=6):
    """Train/test masks: whole 16x16 blocks to training until >= frac of labelled pixels,
    test pixels within `buffer` px (Chebyshev) of training discarded."""
    h, w = gt.shape
    bw = (w + block - 1) // block
    blk = np.zeros(gt.shape, np.int32)
    for by in range((h + block - 1) // block):
        for bx in range(bw):
            blk[by * block : (by + 1) * block, bx * block : (bx + 1) * block] = by * bw + bx

    labeled = gt > 0
    ids = np.unique(blk)
    rng.shuffle(ids)
    train = np.zeros(gt.shape, bool)
    got, target = 0, frac * labeled.sum()
    for i in ids:
        if got >= target:
            break
        m = blk == i
        if not (m & labeled).any():
            continue
        train |= m
        got += (m & labeled).sum()
    train &= labeled

    k = 2 * buffer + 1
    near = binary_dilation(train, structure=np.ones((k, k), bool))
    test = labeled & ~train & ~near
    return train, test


def make_patches(cube, mask, patch):
    """(n, bands, patch, patch) patches around the pixels of `mask` (reflect padding) and their (ys,
    xs)."""
    r = patch // 2
    pad = np.pad(cube, ((r, r), (r, r), (0, 0)), mode="reflect")
    ys, xs = np.where(mask)
    out = np.empty((len(ys), cube.shape[2], patch, patch), np.float32)
    for i, (y, x) in enumerate(zip(ys, xs, strict=True)):
        out[i] = pad[y : y + patch, x : x + patch, :].transpose(2, 0, 1)
    return out, ys, xs


def block_ids(shape, block=BLOCK):
    h, w = shape
    bw = (w + block - 1) // block
    yy, xx = np.mgrid[0:h, 0:w]
    return (yy // block) * bw + (xx // block)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def write_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def parse(path):
    """scene, model, split seed, training seed from '<scene>_<model>_s<split>_t<seed>.npz'."""
    base = os.path.basename(path)[:-4]
    sc, rest = ("pavia_u", base[8:]) if base.startswith("pavia_u") else base.split("_", 1)
    model, s, t = rest.rsplit("_", 2)
    return sc, model, int(s[1:]), int(t[1:])


def runs(models=None, folder=HSI_Q1):
    """(scene, model, split, seed, path) of the stored per-pixel outputs, in a fixed order."""
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        sc, m, s, t = parse(p)
        if models is None or m in models:
            out.append((sc, m, s, t, p))
    return out


# ladder features (Section 6.3)
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ent(p):
    """Natural-log entropy of each row."""
    return -(p * np.log(p + 1e-12)).sum(1)


def top2(a):
    s = np.sort(a, 1)
    return s[:, -1], s[:, -2]


def ladders_of(model, d):
    """Ladders of one run: list of (name, y, {rung: X}, [(upper, lower)], singles), and extras.
    y = 1 where the model's prediction is wrong (test pixels)."""
    y_true = d["y"].astype(int)
    out, extra = [], {}
    if model in ("softmax", "hybrid_softmax"):
        p = softmax(d["logits"])
        pred = p.argmax(1)
        H = ent(p)
        pm, p2 = top2(p)
        y = (pred != y_true).astype(int)
        out.append(
            (
                "S",
                y,
                {"S1": H[:, None], "S2": np.c_[H, pm], "S3": np.c_[H, pm, p2]},
                [("S2", "S1"), ("S3", "S2")],
                {"H": H, "p_max": pm, "p_2": p2},
            )
        )
        if "mc_pe" in d:
            mp = d["mc_mean_p"].astype(float)
            pe, mi, mpm = d["mc_pe"], d["mc_mi"], mp.max(1)
            extra["mc_oa"] = float((mp.argmax(1) == y_true).mean())
            out.append(
                (
                    "M",
                    y,
                    {"M1": pe[:, None], "M2": np.c_[pe, mi], "M3": np.c_[pe, mi, mpm]},
                    [("M2", "M1"), ("M3", "M2")],
                    {"PE": pe, "MI": mi, "p_max_mean": mpm},
                )
            )
    elif model in ("evidential", "hybrid_evidential"):
        ev = np.logaddexp(0, d["logits"])  # softplus evidence
        al = ev + 1
        S = al.sum(1)
        K = al.shape[1]
        H = ent(al / S[:, None])
        u = K / S
        b = ev / S[:, None]
        bm, b2 = top2(b)
        y = (al.argmax(1) != y_true).astype(int)
        out.append(
            (
                "R",
                y,
                {
                    "R1": H[:, None],
                    "R2": np.c_[H, u],
                    "R3plus": np.c_[H, u, bm, b2],
                    "R3": np.c_[bm, b2, u],
                },
                [("R2", "R1"), ("R3plus", "R2"), ("R3", "R2")],
                {"H": H, "u": u, "b_max": bm, "b_2": b2},
            )
        )
    elif model == "nsu":
        T_all, F_all = 1 / (1 + np.exp(-d["zT"])), 1 / (1 + np.exp(-d["zF"]))
        k = (T_all - F_all).argmax(1)
        r = np.arange(len(k))
        T, F = T_all[r, k], F_all[r, k]
        pt = softmax(d["zT"] - d["zF"])
        Ht, ptm = ent(pt), pt.max(1)
        y = (k != y_true).astype(int)
        extra["share_TF_gt1"] = float(np.mean(T + F > 1))
        extra["share_TF_lt1"] = float(np.mean(T + F < 1))
        extra["mean_TF"] = float(np.mean(T + F))
        out.append(
            (
                "N",
                y,
                {
                    "N1": (T - F)[:, None],
                    "N2": np.c_[T, F],
                    "N2base": np.c_[Ht, ptm],
                    "N2plus": np.c_[Ht, ptm, T, F],
                },
                [("N2", "N1"), ("N2plus", "N2base")],
                {"T_minus_F": T - F, "T": T, "F": F, "H_ptilde": Ht, "p_max_ptilde": ptm},
            )
        )
    return out, extra


def pred_of(model, d):
    if model == "nsu":
        return (1 / (1 + np.exp(-d["zT"])) - 1 / (1 + np.exp(-d["zF"]))).argmax(1)
    return d["logits"].argmax(1)


def metrics(d, pred):
    """OA, AA (mean per-class recall over classes present in test), kappa, errors; per-class
    accuracy and counts."""
    y = d["y"].astype(int)
    K = int(d["K"])
    nte = np.bincount(y, minlength=K)
    cor = np.bincount(y[pred == y], minlength=K)
    pc = np.where(nte > 0, cor / np.maximum(nte, 1), np.nan)
    return (
        dict(
            OA=float((pred == y).mean()),
            AA=float(np.nanmean(pc)),
            kappa=float(cohen_kappa_score(y, pred)),
            errors=int((pred != y).sum()),
        ),
        pc,
        nte,
    )


# detector and bootstrap (Section 4)
def oof_scores(X, y, seed=0):
    """Out-of-fold error probabilities: standardise + L2 logistic regression (C = 1), stratified
    5-fold."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]


def boot_delta(y, s_a, s_b, label_a, label_b):
    """Paired bootstrap of AUC(a) - AUC(b): mean, 95% percentile CI, P(delta > 0)."""
    rng = np.random.default_rng(BOOT_SEED)
    n = len(y)
    ds = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        ds.append(roc_auc_score(yb, s_a[idx]) - roc_auc_score(yb, s_b[idx]))
    d = np.array(ds)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {
        "label": f"{label_a} - {label_b}",
        "mean": float(d.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "p_gt0": float(np.mean(d > 0)),
    }


class _Sorted:
    def __init__(self, s):
        o = np.argsort(s, kind="mergesort")
        ss = s[o]
        start = np.r_[True, ss[1:] != ss[:-1]]
        self.o, self.starts = o, np.flatnonzero(start)


def _auc(srt, wp, wn):
    """Weighted AUC in O(n) after one sort; tied scores count 1/2, as in roc_auc_score."""
    P = np.add.reduceat(wp[srt.o], srt.starts)
    N = np.add.reduceat(wn[srt.o], srt.starts)
    below = np.cumsum(N) - N
    tp, tn = P.sum(), N.sum()
    return float((P * (below + 0.5 * N)).sum() / (tp * tn))


def boot_multi(y, scores, pairs, B=N_BOOTSTRAP, seed=BOOT_SEED):
    """Same resamples and skip rule as boot_delta (multiplicities c_i = bincount of the indices),
    for many pairs.
    Returns {(a, b): dict(mean, lo, hi, p_gt0, point, B_eff)}."""
    y = np.asarray(y, int)
    n = len(y)
    srt = {k: _Sorted(np.asarray(v, float)) for k, v in scores.items()}
    need = sorted({k for p in pairs for k in p})
    rng = np.random.default_rng(seed)
    ds = {p: [] for p in pairs}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        c = np.bincount(idx, minlength=n).astype(float)
        npos = (c * y).sum()
        if npos == 0 or npos == n:
            continue
        wp, wn = c * y, c * (1 - y)
        a = {k: _auc(srt[k], wp, wn) for k in need}
        for p in pairs:
            ds[p].append(a[p[0]] - a[p[1]])
    one = np.ones(n)
    pt = {k: _auc(srt[k], one * y, one * (1 - y)) for k in need}
    out = {}
    for p in pairs:
        d = np.array(ds[p])
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[p] = dict(
            mean=float(d.mean()),
            lo=float(lo),
            hi=float(hi),
            p_gt0=float(np.mean(d > 0)),
            point=pt[p[0]] - pt[p[1]],
            B_eff=len(d),
        )
    return out


def boot_dist(y, scores, pairs, B=N_BOOTSTRAP, seed=BOOT_SEED, groups=None):
    """As boot_multi, but also returns the bootstrap differences. groups=None resamples pixels (same
    rng stream);
    groups = block ids resamples blocks with replacement (spatial block bootstrap)."""
    y = np.asarray(y, int)
    n = len(y)
    srt = {k: _Sorted(np.asarray(v, float)) for k, v in scores.items()}
    need = sorted({k for p in pairs for k in p})
    rng = np.random.default_rng(seed)
    if groups is not None:
        ug, inv = np.unique(groups, return_inverse=True)
        G = len(ug)
    ds = {p: [] for p in pairs}
    for _ in range(B):
        if groups is None:
            c = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        else:
            c = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)[inv]
        npos = (c * y).sum()
        if npos == 0 or npos == c.sum():
            continue
        wp, wn = c * y, c * (1 - y)
        a = {k: _auc(srt[k], wp, wn) for k in need}
        for p in pairs:
            ds[p].append(a[p[0]] - a[p[1]])
    one = np.ones(n)
    pt = {k: _auc(srt[k], one * y, one * (1 - y)) for k in need}
    return {p: (np.array(ds[p]), pt[p[0]] - pt[p[1]]) for p in pairs}, pt


def summ(d, point):
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(
        point=point,
        mean=float(d.mean()),
        lo=float(lo),
        hi=float(hi),
        p_gt0=float(np.mean(d > 0)),
        share_le0=float(np.mean(d <= 0)),
        share_ge0=float(np.mean(d >= 0)),
        B_eff=len(d),
    )


def rule(mean, lo, thr=0.03):
    """Rule (10): mean delta-AUC > thr and lower CI bound > 0."""
    return "yes" if (mean > thr and lo > 0) else "no"
