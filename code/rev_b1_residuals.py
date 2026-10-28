"""Revision analyses (post hoc), comment 7 (B1): NS-U residual T + F - 1 at k* = argmax(T - F), test
pixels
(Table S7, Figs. S1-S3)."""

import os

import common as C
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sig = lambda z: 1 / (1 + np.exp(-z.astype(np.float64)))  # noqa: E731
rows, data = [], {}
for sc, m, s, t, p in C.runs(["nsu"]):
    d = np.load(p)
    Ta, Fa = sig(d["zT"]), sig(d["zF"])
    k = (Ta - Fa).argmax(1)
    i = np.arange(len(k))
    T, F = Ta[i, k], Fa[i, k]
    r = T + F - 1
    a = np.abs(r)
    q = np.percentile(r, [5, 50, 95, 99])
    rows.append(
        [
            sc,
            s,
            t,
            len(r),
            f"{a.mean():.5f}",
            f"{np.median(a):.5f}",
            f"{a.std(ddof=1):.5f}",
            *[f"{x:+.5f}" for x in q],
            f"{np.mean(a > 0.05):.4f}",
            f"{np.mean(T + F > 1):.4f}",
        ]
    )
    T32, F32 = (
        (1 / (1 + np.exp(-d["zT"])))[i, k],
        (1 / (1 + np.exp(-d["zF"])))[i, k],
    )  # float32, as in hsi_analyze.py
    rows[-1] += [f"{np.mean(T32 + F32 > 1):.4f}", f"{np.mean(np.abs(r) < 1e-6):.4f}"]
    data[(sc, s, t)] = (T, F, k)
hdr = [
    "scene",
    "split_seed",
    "train_seed",
    "n_test_px",
    "mean_abs_res",
    "median_abs_res",
    "sd_abs_res",
    "p05_res",
    "p50_res",
    "p95_res",
    "p99_res",
    "share_abs_res_gt_0.05",
    "share_TplusF_gt_1",
    "share_TplusF_gt_1_float32_as_in_q1",
    "share_abs_res_lt_1e-6",
]
C.write_csv(C.REV / "c07_residuals_by_run.csv", hdr, rows)
os.makedirs(C.REV_FIG, exist_ok=True)

pts = [
    [sc, s, t, int(kk) + 1, f"{1 - TT:.6f}", f"{FF:.6f}"]
    for (sc, s, t), (T, F, k) in data.items()
    for TT, FF, kk in zip(T, F, k, strict=True)
]
C.write_csv(
    C.REV / "c07_points_F_vs_1minusT.csv",
    ["scene", "split_seed", "train_seed", "pred_class", "one_minus_T", "F"],
    pts,
)


def hb(ax, x, y, title):
    ax.hexbin(x, y, gridsize=60, bins="log", extent=(0, 1, 0, 1), mincnt=1, cmap="viridis")
    ax.plot([0, 1], [0, 1], "r-", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


fig, ax = plt.subplots(1, 2, figsize=(8, 4))
for j, sc in enumerate(C.SCENE_LIST):
    T = np.concatenate([v[0] for kk, v in data.items() if kk[0] == sc])
    F = np.concatenate([v[1] for kk, v in data.items() if kk[0] == sc])
    hb(ax[j], 1 - T, F, f"{sc}: all 9 runs (n={len(T)})")
    ax[j].set_xlabel("1 − T (predicted class)")
    ax[j].set_ylabel("F (predicted class)")
fig.suptitle("NS-U, test pixels; red line F = 1 − T (revision analysis, post hoc)", fontsize=9)
fig.tight_layout()
fig.savefig(f"{C.REV_FIG}/c07_F_vs_1minusT_by_scene.png", dpi=300)
plt.close(fig)

for sc in C.SCENE_LIST:
    fig, ax = plt.subplots(3, 3, figsize=(9, 9))
    for s in range(3):
        for t in range(3):
            T, F, _ = data[(sc, s, t)]
            hb(ax[s, t], 1 - T, F, f"split {s}, train seed {t}")
    fig.suptitle(f"{sc}: F vs 1 − T by run (revision analysis, post hoc)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{C.REV_FIG}/c07_F_vs_1minusT_{sc}_by_seed.png", dpi=300)
    plt.close(fig)
    K = max(int(v[2].max()) for kk, v in data.items() if kk[0] == sc) + 1
    nc = 4
    fig, ax = plt.subplots((K + nc - 1) // nc, nc, figsize=(10, 2.6 * ((K + nc - 1) // nc)))
    ax = ax.ravel()
    T = np.concatenate([v[0] for kk, v in data.items() if kk[0] == sc])
    F = np.concatenate([v[1] for kk, v in data.items() if kk[0] == sc])
    k = np.concatenate([v[2] for kk, v in data.items() if kk[0] == sc])
    for c in range(K):
        m = k == c
        hb(ax[c], 1 - T[m], F[m], f"predicted class {c + 1} (n={m.sum()})")
    for c in range(K, len(ax)):
        ax[c].axis("off")
    fig.suptitle(
        f"{sc}: F vs 1 − T by predicted class, 9 runs pooled (revision analysis, post hoc)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(f"{C.REV_FIG}/c07_F_vs_1minusT_{sc}_by_class.png", dpi=300)
    plt.close(fig)
