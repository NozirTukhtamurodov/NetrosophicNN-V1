"""Revision analyses (post hoc), comments 5-6, C1(a), Table S11: theorem check on frozen NS-U
backbones
(retrained replicas). Only the heads T (W, b) and F (V, c) are refit on the training-pixel
features h:
  L = mean_i sum_k [BCE(sigma(W h_i + b)_k, y_ik) + BCE(sigma(V h_i + c)_k, 1 - y_ik)] + lambda/2
  (|W|^2 + |b|^2 + |V|^2 + |c|^2),
full batch, L-BFGS (strong Wolfe), float64, from the trained heads, until |grad| < 1e-6; lambda
in {0, 1e-4, 1e-3, 1e-2}."""

import glob
import os
import time

import common as C
import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
LAMS = [0.0, 1e-4, 1e-3, 1e-2]
TOL, MAX_OUTER = 1e-6, 400


def resid_stats(T, F_):
    r = T + F_ - 1
    a = np.abs(r)
    q = np.percentile(r, [5, 50, 95, 99])
    return [
        f"{a.mean():.3e}",
        f"{np.median(a):.3e}",
        f"{a.std(ddof=1):.3e}",
        *[f"{x:+.3e}" for x in q],
        f"{np.mean(a > 0.05):.4f}",
        f"{np.mean(T + F_ > 1):.4f}",
    ]


def at_pred(zT, zF):
    Ta, Fa = torch.sigmoid(zT).numpy(), torch.sigmoid(zF).numpy()
    k = (Ta - Fa).argmax(1)
    i = np.arange(len(k))
    return Ta[i, k], Fa[i, k], k


rows = []
paths = sorted(glob.glob(os.path.join(C.HSI_REV, "replica", "*_nsu_*.npz")))
for p in paths:
    sc, m, s, t = C.parse(p)
    z = np.load(p)
    sd = torch.load(
        os.path.join(C.WEIGHTS_REV, "replica", os.path.basename(p)[:-4] + ".pt"), map_location="cpu"
    )
    Htr, Hte = (
        torch.from_numpy(z["h_train"].astype(np.float64)),
        torch.from_numpy(z["h_test"].astype(np.float64)),
    )
    ytr = torch.from_numpy(z["y_train"].astype(np.int64))
    K = int(z["K"])
    Y = F.one_hot(ytr, K).double()
    for lam in LAMS:
        t0 = time.time()
        W = sd["hT.weight"].double().clone().requires_grad_(True)
        b = sd["hT.bias"].double().clone().requires_grad_(True)
        V = sd["hF.weight"].double().clone().requires_grad_(True)
        c = sd["hF.bias"].double().clone().requires_grad_(True)
        P = [W, b, V, c]

        def obj():
            zT, zF = Htr @ W.T + b, Htr @ V.T + c
            L = (
                F.binary_cross_entropy_with_logits(zT, Y, reduction="none").sum(1)
                + F.binary_cross_entropy_with_logits(zF, 1 - Y, reduction="none").sum(1)
            ).mean()
            return L + lam / 2 * sum((q**2).sum() for q in P)

        opt = torch.optim.LBFGS(
            P,
            lr=1,
            max_iter=100,
            history_size=50,
            tolerance_grad=1e-12,
            tolerance_change=0,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            L = obj()
            L.backward()
            return L

        gn, outer = np.inf, 0
        while outer < MAX_OUTER:
            opt.step(closure)
            outer += 1
            opt.zero_grad()
            L = obj()
            L.backward()
            gn = float(torch.cat([q.grad.flatten() for q in P]).norm())
            if gn < TOL or not np.isfinite(gn):
                break
        with torch.no_grad():
            dW, dc = (V + W).abs(), (c + b).abs()
            Ttr, Ftr, _ = at_pred(Htr @ W.T + b, Htr @ V.T + c)
            Tte, Fte, kte = at_pred(Hte @ W.T + b, Hte @ V.T + c)
            acc = float((kte == z["y"].astype(int)).mean())
            wnorm = float(sum((q**2).sum() for q in P).sqrt())
        rows.append(
            [
                sc,
                s,
                t,
                f"{lam:g}",
                outer,
                f"{gn:.3e}",
                "yes" if gn < TOL else "NO",
                f"{float(L):.8f}",
                f"{wnorm:.4e}",
                f"{float(dW.max()):.3e}",
                f"{float(dW.mean()):.3e}",
                f"{float(dc.max()):.3e}",
                f"{float(dc.mean()):.3e}",
                f"{acc:.4f}",
                *resid_stats(Ttr, Ftr),
                *resid_stats(Tte, Fte),
                f"{time.time() - t0:.1f}",
            ]
        )
        print(rows[-1][:8], flush=True)
st = [
    "mean_abs_res",
    "median_abs_res",
    "sd_abs_res",
    "p05_res",
    "p50_res",
    "p95_res",
    "p99_res",
    "share_abs_gt0.05",
    "share_TplusF_gt1",
]
C.write_csv(
    C.REV / "c06_head_refit_by_run.csv",
    [
        "scene",
        "split_seed",
        "train_seed",
        "lambda",
        "lbfgs_outer_steps",
        "final_grad_norm",
        "grad_lt_1e-6",
        "final_objective",
        "head_param_norm",
        "max_abs_V_plus_W",
        "mean_abs_V_plus_W",
        "max_abs_c_plus_b",
        "mean_abs_c_plus_b",
        "test_accuracy_argmax_T_minus_F",
        *[f"train_{x}" for x in st],
        *[f"test_{x}" for x in st],
        "wall_s",
    ],
    rows,
)
