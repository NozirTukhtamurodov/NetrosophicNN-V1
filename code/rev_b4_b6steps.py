"""Revision analyses (post hoc), comment 12 (B4) for the single-coordinate steps (B6): evidential
HSINet and NS-U,
all 36 runs, same variants as rev_engine.py. Writes engine_steps_b4_b6.csv."""

import common as C
import numpy as np
from rev_engine import WIDTH, extend, oof
from joblib import Parallel, delayed


def job(sc, m, s, t, p):
    d = dict(np.load(p))
    ys, xs, w = d["ys"].astype(int), d["xs"].astype(int), WIDTH[sc]
    g16 = (ys // 16) * ((w + 15) // 16) + xs // 16
    g32 = (ys // 32) * ((w + 31) // 32) + xs // 32
    rows = []
    for lad in C.ladders_of(m, d)[0]:
        name, y = lad[0], lad[1]
        feats, steps, b6, _ = extend(m, lad)
        rungs = sorted({k for p_ in b6 for k in p_})
        sc1 = {k: oof(feats[k], y)[0] for k in rungs}
        variants = [("origcv_blockboot16", sc1, g16)]
        for gname, g in (("blockcv16", g16), ("blockcv32", g32)):
            variants.append((gname, {k: oof(feats[k], y, groups=g)[0] for k in rungs}, g))
        for vname, scores, g in variants:
            res, pt = C.boot_dist(y, scores, b6, groups=g)
            for p_ in b6:
                sm = C.summ(*res[p_])
                rows.append(
                    [
                        sc,
                        m,
                        s,
                        t,
                        name,
                        vname,
                        f"{p_[0]}-{p_[1]}",
                        "B6",
                        len(y),
                        int(y.sum()),
                        f"{pt[p_[0]]:.6f}",
                        f"{pt[p_[1]]:.6f}",
                        f"{sm['point']:+.6f}",
                        f"{sm['mean']:+.6f}",
                        f"{sm['lo']:+.6f}",
                        f"{sm['hi']:+.6f}",
                        f"{sm['p_gt0']:.5f}",
                        f"{sm['share_le0']:.5f}",
                        f"{sm['share_ge0']:.5f}",
                        sm["B_eff"],
                        0,
                    ]
                )
    return rows


if __name__ == "__main__":
    res = Parallel(n_jobs=18, verbose=5)(delayed(job)(*r) for r in C.runs(["evidential", "nsu"]))
    C.write_csv(
        C.REV / "engine_steps_b4_b6.csv",
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
        [x for r in res for x in r],
    )
