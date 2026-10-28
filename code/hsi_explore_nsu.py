"""Fig. 2b and Section 6.4 (exploratory, decided after seeing the results): where the NS-U
information sits.
Steps T vs (H(p~), p~max), (T, F) over T, N2+ over (T, F), T vs T - F; same detector and
bootstrap as Table 4.
Writes results/q1/X_nsu_exploratory.csv."""

import argparse
import glob
import os

import numpy as np
from common import HSI_Q1, RESULTS_DIR, boot_multi, ent, oof_scores, softmax, write_csv
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

STEPS = [("T_only", "N2base"), ("N2", "T_only"), ("N2plus", "N2"), ("T_only", "N1")]


def feats(path):
    d = np.load(path)
    Ta, Fa = 1 / (1 + np.exp(-d["zT"])), 1 / (1 + np.exp(-d["zF"]))
    k = (Ta - Fa).argmax(1)
    r = np.arange(len(k))
    T, F = Ta[r, k], Fa[r, k]
    pt = softmax(d["zT"] - d["zF"])
    y = (k != d["y"]).astype(int)
    return y, {
        "N2base": np.c_[ent(pt), pt.max(1)],
        "T_only": T[:, None],
        "N1": (T - F)[:, None],
        "N2": np.c_[T, F],
        "N2plus": np.c_[ent(pt), pt.max(1), T, F],
    }


def run(tag, y, X):
    sc = {k: oof_scores(v, y) for k, v in X.items()}
    au = {k: roc_auc_score(y, s) for k, s in sc.items()}
    bs = boot_multi(y, sc, STEPS)
    return [
        [
            *tag,
            f"{a}-{b}",
            len(y),
            int(y.sum()),
            f"{au[a]:.4f}",
            f"{au[b]:.4f}",
            f"{r['point']:+.4f}",
            f"{r['mean']:+.4f}",
            f"{r['lo']:+.4f}",
            f"{r['hi']:+.4f}",
            "yes" if (r["mean"] > 0.03 and r["lo"] > 0) else "no",
        ]
        for (a, b), r in bs.items()
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS_DIR / "q1"))
    out = ap.parse_args().out
    paths = sorted(glob.glob(os.path.join(HSI_Q1, "*_nsu_*.npz")))
    F = {p: feats(p) for p in paths}

    def one(p):
        b = os.path.basename(p)[:-4].split("_")
        return run((b[0] if b[0] != "pavia" else "pavia_u", b[-2], b[-1]), *F[p])

    rows = [x for r in Parallel(n_jobs=12)(delayed(one)(p) for p in paths) for x in r]
    for sc in ("pavia_u", "salinas"):
        ps = [p for p in paths if os.path.basename(p).startswith(sc)]
        y = np.concatenate([F[p][0] for p in ps])
        X = {k: np.concatenate([F[p][1][k] for p in ps]) for k in F[ps[0]][1]}
        rows += run((sc, "pooled", f"all({len(ps)})"), y, X)
    write_csv(
        os.path.join(out, "X_nsu_exploratory.csv"),
        [
            "scene",
            "split",
            "train_seed",
            "step",
            "n",
            "errors",
            "auc_upper",
            "auc_lower",
            "delta_point",
            "delta_boot_mean",
            "ci_lo",
            "ci_hi",
            "pass_rule",
        ],
        rows,
    )


if __name__ == "__main__":
    main()
