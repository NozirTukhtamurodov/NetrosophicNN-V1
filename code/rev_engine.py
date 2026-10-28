"""Revision analyses (post hoc), comments 9, 10-11, 12, 13, 15 (B4-B8): every ladder of every run
recomputed from
the stored outputs (no training), in several variants:
  orig                 as in Table 4 (stratified 5-fold OOF, C = 1, pixel bootstrap B = 3000,
  seed 1337)
  C=<c>                B5: other regularisation strengths; C = inf is the unpenalised fit
  origcv_blockboot16   B4: original OOF scores, bootstrap over 16x16 test blocks
  blockcv16/blockcv32  B4: GroupKFold(5) by 16x16 / 32x32 blocks + bootstrap over the same blocks
  absent_removed       B7: Pavia split 0 and Salinas split 1 without the test pixels of
  training-absent classes
Steps: Table 4 steps + single-coordinate steps (B6). Per rung: AUC, AUPRC, AURC (B8) and
design-matrix rank (B5b).
Writes engine_steps, engine_rungs, engine_evid_sums, engine_foldauc (results/revision/csv)."""

import os
import time
import warnings

import common as C
import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CS = [0.01, 0.1, 10.0, 100.0, np.inf]
ABSENT = {("pavia_u", 0), ("salinas", 1)}
WIDTH = {sc: C.load_scene(sc)[1].shape[1] for sc in C.SCENE_LIST}


def oof(X, y, Cval=1.0, groups=None):
    """OOF error probabilities and the number of ConvergenceWarnings; C = 1, groups = None equals
    common.oof_scores."""
    lr = (
        LogisticRegression(max_iter=2000)
        if Cval == 1.0
        else LogisticRegression(C=Cval, max_iter=2000)
    )
    clf = make_pipeline(StandardScaler(), lr)
    cv = StratifiedKFold(5, shuffle=True, random_state=0) if groups is None else GroupKFold(5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        s = cross_val_predict(clf, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    return s, sum(issubclass(x.category, ConvergenceWarning) for x in w)


def fold_auc(y, s, groups=None):
    """Mean AUC within the folds used by cross_val_predict (diagnostic for blocked CV)."""
    cv = StratifiedKFold(5, shuffle=True, random_state=0) if groups is None else GroupKFold(5)
    a = [
        roc_auc_score(y[te], s[te])
        for _, te in cv.split(s[:, None], y, groups)
        if 0 < y[te].sum() < len(te)
    ]
    return float(np.mean(a)), len(a)


def aurc(y, s):
    """Area under the risk-coverage curve: pixels accepted by increasing error score; mean risk over
    the n coverages."""
    o = np.argsort(s, kind="mergesort")
    risk = np.cumsum(y[o]) / np.arange(1, len(y) + 1)
    return float(risk.mean())


def extend(model, lad):
    """Adds the single-coordinate rungs (B6). Returns (rungs, Table 4 steps, B6 steps, b_max + b_2 +
    u summary)."""
    name, y, feats, steps, sg = lad
    feats = dict(feats)
    b6, sums = [], None
    if name == "R":
        H, u, bm, b2 = sg["H"], sg["u"], sg["b_max"], sg["b_2"]
        feats.update(
            {
                "R2_bmax": np.c_[H, u, bm],
                "R2_b2": np.c_[H, u, b2],
                "BMU": np.c_[bm, u],
                "BF": np.c_[bm, 1 - u - bm, u],
            }
        )
        b6 = [
            ("R2_bmax", "R2"),
            ("R3plus", "R2_bmax"),
            ("R2_b2", "R2"),
            ("BF", "R2"),
            ("BF", "BMU"),
        ]
        t = bm + b2 + u
        sums = (float(t.mean()), float(t.min()), float(np.median(t)))
    if name == "N":
        Ht, pm, T, F = sg["H_ptilde"], sg["p_max_ptilde"], sg["T"], sg["F"]
        feats.update({"N2base_T": np.c_[Ht, pm, T], "N2base_F": np.c_[Ht, pm, F]})
        b6 = [("N2base_T", "N2base"), ("N2plus", "N2base_T"), ("N2base_F", "N2base")]
    return feats, steps, b6, sums


def rank_info(X):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-300)
    sv = np.linalg.svd(Z, compute_uv=False)
    tol = sv.max() * max(Z.shape) * np.finfo(float).eps
    return int((sv > tol).sum()), X.shape[1], float(sv.max() / max(sv.min(), 1e-300))


def ladders_for(tag, d):
    model = tag[1]
    if model == "ensemble5":
        return d["ens_ladders"]
    lad, _ = C.ladders_of(model, d)
    return lad


def run_job(tag, d):
    sc, model, s, t = tag
    out_s, out_r, out_e, out_f = [], [], [], []
    ys, xs = d["ys"].astype(int), d["xs"].astype(int)
    w = WIDTH[sc]
    g16 = (ys // 16) * ((w + 15) // 16) + xs // 16
    g32 = (ys // 32) * ((w + 31) // 32) + xs // 32
    for lad in ladders_for(tag, d):
        name, y = lad[0], lad[1]
        feats, steps, b6, sums = extend(model, lad)
        if sums:
            out_e.append([*tag, name, *[f"{v:.6f}" for v in sums]])
        allsteps = steps + b6
        kind = {st: "orig" for st in steps} | {st: "B6" for st in b6}

        def emit(variant, scores, pairs, groups=None, yy=y, conv=None, extra_r=False):
            res, pt = C.boot_dist(yy, scores, pairs, groups=groups)
            for p in pairs:
                dd, point = res[p]
                sm = C.summ(dd, point)
                out_s.append(
                    [
                        *tag,
                        name,
                        variant,
                        f"{p[0]}-{p[1]}",
                        kind.get(p, "orig"),
                        len(yy),
                        int(yy.sum()),
                        f"{pt[p[0]]:.6f}",
                        f"{pt[p[1]]:.6f}",
                        f"{sm['point']:+.6f}",
                        f"{sm['mean']:+.6f}",
                        f"{sm['lo']:+.6f}",
                        f"{sm['hi']:+.6f}",
                        f"{sm['p_gt0']:.5f}",
                        f"{sm['share_le0']:.5f}",
                        f"{sm['share_ge0']:.5f}",
                        sm["B_eff"],
                        (conv or {}).get(p[0], 0) + (conv or {}).get(p[1], 0),
                    ]
                )
            if extra_r:
                for k, sc_ in scores.items():
                    rk, nc, cond = rank_info(feats[k] if yy is y else feats_sub[k])
                    out_r.append(
                        [
                            *tag,
                            name,
                            variant,
                            k,
                            len(yy),
                            f"{roc_auc_score(yy, sc_):.6f}",
                            f"{average_precision_score(yy, sc_):.6f}",
                            f"{aurc(yy, sc_):.6f}",
                            rk,
                            nc,
                            f"{cond:.3e}",
                            (conv or {}).get(k, 0),
                        ]
                    )

        sc1, cv1 = {}, {}
        for k, X in feats.items():
            sc1[k], cv1[k] = oof(X, y)
        feats_sub = feats
        emit("orig", sc1, allsteps, conv=cv1, extra_r=True)
        for k in {x for p in steps for x in p}:
            out_f.append(
                [*tag, name, "orig", k, f"{roc_auc_score(y, sc1[k]):.6f}", *fold_auc(y, sc1[k])]
            )
        for c in CS:
            scc, cvc = {}, {}
            for k, X in feats.items():
                scc[k], cvc[k] = oof(X, y, Cval=c)
            emit(f"C={c:g}", scc, allsteps, conv=cvc, extra_r=True)
        emit("origcv_blockboot16", sc1, steps, groups=g16, conv=cv1)
        for gname, g in (("blockcv16", g16), ("blockcv32", g32)):
            if len(np.unique(g)) < 5:
                continue
            sg_, cg_ = {}, {}
            for k in {x for p in steps for x in p}:
                sg_[k], cg_[k] = oof(feats[k], y, groups=g)
            emit(gname, sg_, steps, groups=g, conv=cg_, extra_r=False)
            for k in sg_:
                out_f.append(
                    [
                        *tag,
                        name,
                        gname,
                        k,
                        f"{roc_auc_score(y, sg_[k]):.6f}",
                        *fold_auc(y, sg_[k], g),
                    ]
                )
        if (
            (sc, s) in ABSENT
            and model != "ensemble5"
            or (model == "ensemble5" and (sc, 0) in ABSENT)
        ):
            tc = d["train_counts"]
            keep = tc[d["y"].astype(int)] > 0
            yk = y[keep]
            feats_sub = {k: X[keep] for k, X in feats.items()}
            ss, cs_ = {}, {}
            for k, X in feats_sub.items():
                ss[k], cs_[k] = oof(X, yk)
            emit("absent_removed", ss, allsteps, yy=yk, conv=cs_, extra_r=True)
    return out_s, out_r, out_e, out_f


def ensemble_data(sc):
    ds = [dict(np.load(os.path.join(C.HSI_Q1, f"{sc}_softmax_s0_t{t}.npz"))) for t in range(5)]
    P = np.stack([C.softmax(d["logits"]) for d in ds])
    pm = P.mean(0)
    yt = ds[0]["y"].astype(int)
    H = C.ent(pm)
    MI = H - np.mean([C.ent(p) for p in P], 0)
    y = (pm.argmax(1) != yt).astype(int)
    lad = [
        (
            "E",
            y,
            {"E1": H[:, None], "E2": np.c_[H, MI], "E3": np.c_[H, MI, pm.max(1)]},
            [("E2", "E1"), ("E3", "E2")],
            {},
        )
    ]
    return dict(
        ys=ds[0]["ys"],
        xs=ds[0]["xs"],
        y=ds[0]["y"],
        train_counts=ds[0]["train_counts"],
        ens_ladders=lad,
    )


if __name__ == "__main__":
    t0 = time.time()
    jobs = []
    for sc, m, s, t, p in C.runs():
        if m == "softmax" and t >= 3:
            continue  # extra ensemble members only
        jobs.append(((sc, m, s, t), dict(np.load(p))))
    for sc in C.SCENE_LIST:
        jobs.append(((sc, "ensemble5", 0, "0-4"), ensemble_data(sc)))
    print(f"{len(jobs)} jobs", flush=True)
    res = Parallel(n_jobs=20, verbose=5)(delayed(run_job)(tag, d) for tag, d in jobs)
    S = [x for r in res for x in r[0]]
    R = [x for r in res for x in r[1]]
    E = [x for r in res for x in r[2]]
    C.write_csv(
        C.REV / "engine_steps.csv",
        [
            "scene",
            "model",
            "split_seed",
            "train_seed",
            "ladder",
            "variant",
            "step",
            "step_kind",
            "n",
            "errors",
            "auc_upper",
            "auc_lower",
            "delta_point",
            "delta_boot_mean",
            "ci_lo",
            "ci_hi",
            "p_gt0",
            "share_le0",
            "share_ge0",
            "B_eff",
            "convergence_warnings",
        ],
        S,
    )
    C.write_csv(
        C.REV / "engine_rungs.csv",
        [
            "scene",
            "model",
            "split_seed",
            "train_seed",
            "ladder",
            "variant",
            "rung",
            "n",
            "auc",
            "auprc_errors_positive",
            "aurc",
            "rank",
            "n_cols",
            "cond_number",
            "convergence_warnings",
        ],
        R,
    )
    C.write_csv(
        C.REV / "engine_evid_sums.csv",
        [
            "scene",
            "model",
            "split_seed",
            "train_seed",
            "ladder",
            "mean_bmax_b2_u",
            "min_bmax_b2_u",
            "median_bmax_b2_u",
        ],
        E,
    )
    C.write_csv(
        C.REV / "engine_foldauc.csv",
        [
            "scene",
            "model",
            "split_seed",
            "train_seed",
            "ladder",
            "variant",
            "rung",
            "auc_pooled_oof",
            "auc_mean_within_fold",
            "n_folds",
        ],
        [x for r in res for x in r[3]],
    )
    print(f"done in {time.time() - t0:.0f} s")
