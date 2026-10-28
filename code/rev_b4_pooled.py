"""Revision analyses (post hoc), comment 12 (B4) for the 28 pooled comparisons: original detector
and pixel
bootstrap vs GroupKFold by (run, 16x16 / 32x32 block) with a bootstrap over the same groups."""

import common as C
import numpy as np
from rev_engine import WIDTH, oof
from joblib import Parallel, delayed


def job(sc, model):
    lads, g16, g32 = [], [], []
    for i, (sc2, m, _s, t, p) in enumerate(C.runs([model])):
        if sc2 != sc or (m == "softmax" and t >= 3):
            continue
        d = dict(np.load(p))
        lads.append(C.ladders_of(model, d)[0])
        ys, xs, w = d["ys"].astype(int), d["xs"].astype(int), WIDTH[sc]
        g16.append(i * 10**6 + (ys // 16) * ((w + 15) // 16) + xs // 16)
        g32.append(i * 10**6 + (ys // 32) * ((w + 31) // 32) + xs // 32)
    g16, g32 = np.concatenate(g16), np.concatenate(g32)
    rows = []
    for li in range(len(lads[0])):
        name, steps = lads[0][li][0], lads[0][li][3]
        y = np.concatenate([L[li][1] for L in lads])
        rungs = sorted({k for p in steps for k in p})
        feats = {k: np.concatenate([L[li][2][k] for L in lads]) for k in rungs}
        for variant, groups in (("orig", None), ("blockcv16", g16), ("blockcv32", g32)):
            sc_ = {k: oof(X, y, groups=groups)[0] for k, X in feats.items()}
            res, pt = C.boot_dist(y, sc_, steps, groups=groups)
            for p in steps:
                sm = C.summ(*res[p])
                rows.append(
                    [
                        sc,
                        model,
                        f"pooled ({len(lads)} runs)",
                        name,
                        variant,
                        f"{p[0]}-{p[1]}",
                        len(y),
                        int(y.sum()),
                        f"{pt[p[0]]:.6f}",
                        f"{pt[p[1]]:.6f}",
                        f"{sm['mean']:+.6f}",
                        f"{sm['lo']:+.6f}",
                        f"{sm['hi']:+.6f}",
                        C.rule(sm["mean"], sm["lo"]),
                    ]
                )
    return rows


if __name__ == "__main__":
    jobs = [
        (sc, m)
        for sc in C.SCENE_LIST
        for m in ("softmax", "evidential", "nsu", "hybrid_softmax", "hybrid_evidential")
    ]
    res = Parallel(n_jobs=10, verbose=5)(delayed(job)(sc, m) for sc, m in jobs)
    C.write_csv(
        C.REV / "c12_blocked_vs_original_pooled.csv",
        [
            "scene",
            "model",
            "runs",
            "ladder",
            "variant",
            "step",
            "n",
            "errors",
            "auc_upper",
            "auc_lower",
            "delta_mean",
            "ci_lo",
            "ci_hi",
            "rule_pass",
        ],
        [x for r in res for x in r],
    )
