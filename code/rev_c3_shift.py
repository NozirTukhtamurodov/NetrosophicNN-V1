"""Revision analyses (post hoc), comment 17 (C3), Tables S14-S15.
(i) Covariate shift on the retrained replicas (no retraining): Gaussian noise N(0, (sigma
sd_b)^2) added to the raw
    spectra before standardisation and PCA, sigma in {0.05, 0.1, 0.2}, sd_b = SD of band b over
    the training pixels
    of the split; separately 10% and 20% of the bands set to 0 (same bands for every run). Noise
    and band choice use
    rng 1337. Then the run's stored standardisation and PCA, same 11x11 patches; accuracy, NLL,
    ECE and ladders.
(ii) OOD: test pixels of training-absent classes (Pavia split 0, Salinas split 1) vs the other
test pixels; AUROC of
    single uncertainty scores from the stored q1 outputs.
Stages: gpu (shifted outputs), cpu (calibration and ladders), summary, ood."""

import os
import sys

import common as C
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hsi_models import HSINet, HybridSN, NSUNet
from joblib import Parallel, delayed
from rev_calibration_evidence import calib, probs
from sklearn.metrics import roc_auc_score

R = os.path.join(C.HSI_REV, "replica")
M = os.path.join(C.WEIGHTS_REV, "replica")
CONDS = [
    ("none", None),
    ("noise_0.05", 0.05),
    ("noise_0.1", 0.1),
    ("noise_0.2", 0.2),
    ("bands_zero_10pct", 0.10),
    ("bands_zero_20pct", 0.20),
]


def shifted_cube(cube, cond, val, sd_raw):
    if cond == "none":
        return cube
    rng = np.random.default_rng(1337)
    out = cube.copy()
    if cond.startswith("noise"):
        out += rng.normal(0, 1, cube.shape).astype(np.float32) * (val * sd_raw).astype(np.float32)
    else:
        nb = int(round(val * cube.shape[2]))
        out[:, :, rng.choice(cube.shape[2], nb, replace=False)] = 0
    return out


def forward(model, net, X, dev, tseed):
    res = {}
    with torch.no_grad():
        o = [net(torch.from_numpy(X[i : i + 512]).to(dev)) for i in range(0, len(X), 512)]
        if model == "nsu":
            res["zT"] = torch.cat([q[0] for q in o]).float().cpu().numpy()
            res["zF"] = torch.cat([q[1] for q in o]).float().cpu().numpy()
        else:
            res["logits"] = torch.cat(o).float().cpu().numpy()
        if model == "softmax":
            torch.manual_seed(1000 + tseed)
            for mm in net.modules():
                if isinstance(mm, nn.Dropout):
                    mm.train()
            psum = np.zeros((len(X), res["logits"].shape[1]))
            hsum = np.zeros(len(X))
            for _ in range(20):
                pr = (
                    torch.cat(
                        [
                            F.softmax(net(torch.from_numpy(X[i : i + 512]).to(dev)), 1)
                            for i in range(0, len(X), 512)
                        ]
                    )
                    .double()
                    .cpu()
                    .numpy()
                )
                psum += pr
                hsum += -(pr * np.log(pr + 1e-12)).sum(1)
            pm = psum / 20
            res["mc_mean_p"] = pm.astype(np.float32)
            res["mc_pe"] = -(pm * np.log(pm + 1e-12)).sum(1)
            res["mc_mi"] = res["mc_pe"] - hsum / 20
            net.eval()
    return res


def gpu_stage():
    """Replica outputs under every shift condition ->
    outputs/hsi_outputs_revision/shift/<cond>/<tag>.npz"""
    dev = "cuda"
    for sc in C.SCENE_LIST:
        cube0, gt = C.load_scene(sc)
        for f in sorted(os.listdir(R)):
            if not f.startswith(sc + "_") or not f.endswith(".npz"):
                continue
            sc2, model, s, t = C.parse(f)
            z = np.load(os.path.join(R, f))
            tr_m, te_m = C.split_block_buffer(gt, 0.15, np.random.default_rng(s), buffer=6)
            K = int(z["K"])
            Net = HybridSN if model.startswith("hybrid") else (NSUNet if model == "nsu" else HSINet)
            net = Net(30, K, 11).to(dev)
            net.load_state_dict(torch.load(os.path.join(M, f[:-4] + ".pt"), map_location=dev))
            net.eval()
            sd_raw = z["prep_sd"] - 1e-8
            for cond, val in CONDS:
                out = os.path.join(C.HSI_REV, "shift", cond, f)
                if os.path.exists(out):
                    continue
                os.makedirs(os.path.dirname(out), exist_ok=True)
                cube = shifted_cube(cube0, cond, val, sd_raw)
                flat = cube.reshape(-1, cube.shape[2])
                pc = (((flat - z["prep_mu"]) / z["prep_sd"]) - z["prep_pca_mean"]) @ z[
                    "prep_pca_comp"
                ].T
                pc = pc.reshape(cube.shape[0], cube.shape[1], -1).astype(np.float32)
                Xte, ys, xs = C.make_patches(pc, te_m, 11)
                res = forward(model, net, Xte, dev, t)
                np.savez_compressed(
                    out,
                    y=z["y"],
                    ys=ys.astype(np.int16),
                    xs=xs.astype(np.int16),
                    K=K,
                    train_counts=z["train_counts"],
                    **res,
                )
            print(f"shift: {f}", flush=True)


def cpu_job(cond, f):
    sc, model, s, t = C.parse(f)
    d = dict(np.load(os.path.join(C.HSI_REV, "shift", cond, f)))
    y = d["y"].astype(int)
    out_m, out_d = [], []
    for o, pr in probs(model, d).items():
        r = calib(pr, y)
        out_m.append(
            [cond, sc, model, o, s, t, *[f"{r[k]:.6f}" for k in ("acc", "nll", "brier", "ece")]]
        )
    if not (model == "softmax" and t >= 3):
        for name, yy, feats, steps, _ in C.ladders_of(model, d)[0]:
            sc_ = {k: C.oof_scores(X, yy) for k, X in feats.items()}
            res, pt = C.boot_dist(yy, sc_, steps)
            for p in steps:
                sm = C.summ(*res[p])
                out_d.append(
                    [
                        cond,
                        sc,
                        model,
                        s,
                        t,
                        name,
                        f"{p[0]}-{p[1]}",
                        len(yy),
                        int(yy.sum()),
                        f"{pt[p[0]]:.6f}",
                        f"{pt[p[1]]:.6f}",
                        f"{sm['mean']:+.6f}",
                        f"{sm['lo']:+.6f}",
                        f"{sm['hi']:+.6f}",
                        C.rule(sm["mean"], sm["lo"]),
                    ]
                )
    return out_m, out_d


def ens_job(cond, sc):
    fs = [os.path.join(C.HSI_REV, "shift", cond, f"{sc}_softmax_s0_t{t}.npz") for t in range(5)]
    ds = [np.load(f) for f in fs]
    P = np.stack([C.softmax(d["logits"].astype(float)) for d in ds])
    pm = P.mean(0)
    yt = ds[0]["y"].astype(int)
    r = calib(pm, yt)
    H = C.ent(pm)
    MI = H - np.mean([C.ent(p) for p in P], 0)
    y = (pm.argmax(1) != yt).astype(int)
    feats = {"E1": H[:, None], "E2": np.c_[H, MI], "E3": np.c_[H, MI, pm.max(1)]}
    steps = [("E2", "E1"), ("E3", "E2")]
    sc_ = {k: C.oof_scores(X, y) for k, X in feats.items()}
    res, pt = C.boot_dist(y, sc_, steps)
    out_d = []
    for p in steps:
        sm = C.summ(*res[p])
        out_d.append(
            [
                cond,
                sc,
                "ensemble5",
                0,
                "0-4",
                "E",
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
    return [
        [
            cond,
            sc,
            "ensemble5",
            "mean p",
            0,
            "0-4",
            *[f"{r[k]:.6f}" for k in ("acc", "nll", "brier", "ece")],
        ]
    ], out_d


def ood():
    rows = []
    for sc, s in (("pavia_u", 0), ("salinas", 1)):
        for sc2, model, s2, t, p in C.runs():
            if sc2 != sc or s2 != s or (model == "softmax" and t >= 3):
                continue
            d = dict(np.load(p))
            ood_ = (d["train_counts"][d["y"].astype(int)] == 0).astype(int)
            S = {}
            if model in ("softmax", "hybrid_softmax"):
                pr = C.softmax(d["logits"].astype(float))
                S = {"H(p)": C.ent(pr), "1-p_max": 1 - pr.max(1)}
                if "mc_pe" in d:
                    S.update({"MC PE": d["mc_pe"], "MC MI": d["mc_mi"]})
            elif model in ("evidential", "hybrid_evidential"):
                ev = np.logaddexp(0, d["logits"].astype(float))
                Sx = ev.sum(1) + ev.shape[1]
                S = {
                    "H(alpha/S)": C.ent((ev + 1) / Sx[:, None]),
                    "u=K/S": ev.shape[1] / Sx,
                    "1-b_max": 1 - ev.max(1) / Sx,
                }
            elif model == "nsu":
                Ta, Fa = (
                    1 / (1 + np.exp(-d["zT"].astype(float))),
                    1 / (1 + np.exp(-d["zF"].astype(float))),
                )
                k = (Ta - Fa).argmax(1)
                i = np.arange(len(k))
                pt_ = C.softmax(d["zT"].astype(float) - d["zF"].astype(float))
                S = {
                    "-(T-F)": -(Ta[i, k] - Fa[i, k]),
                    "1-T": 1 - Ta[i, k],
                    "F": Fa[i, k],
                    "H(p~)": C.ent(pt_),
                    "1-p~_max": 1 - pt_.max(1),
                }
            for nm, v in S.items():
                rows.append(
                    [
                        sc,
                        s,
                        model,
                        t,
                        nm,
                        int(ood_.sum()),
                        int((1 - ood_).sum()),
                        f"{roc_auc_score(ood_, v):.4f}",
                    ]
                )
        if s == 0:
            ds = [np.load(os.path.join(C.HSI_Q1, f"{sc}_softmax_s0_t{t}.npz")) for t in range(5)]
            P = np.stack([C.softmax(d["logits"].astype(float)) for d in ds])
            pm = P.mean(0)
            ood_ = (ds[0]["train_counts"][ds[0]["y"].astype(int)] == 0).astype(int)
            H = C.ent(pm)
            for nm, v in (
                ("H(mean)", H),
                ("MI", H - np.mean([C.ent(p) for p in P], 0)),
                ("1-p_max(mean)", 1 - pm.max(1)),
            ):
                rows.append(
                    [
                        sc,
                        s,
                        "ensemble5",
                        "0-4",
                        nm,
                        int(ood_.sum()),
                        int((1 - ood_).sum()),
                        f"{roc_auc_score(ood_, v):.4f}",
                    ]
                )
    C.write_csv(
        C.REV / "c17_ood_auroc.csv",
        ["scene", "split_seed", "model", "train_seed", "score", "n_ood", "n_id", "auroc"],
        rows,
    )


def summary():
    """Mean accuracy / NLL / ECE per condition and counts of comparisons passing rule (10) (Table
    S14)."""
    import csv
    from collections import defaultdict

    with open(C.REV / "c17_shift_calibration.csv", encoding="utf-8") as f:
        Cal = list(csv.DictReader(f))
    g = defaultdict(list)
    for r in Cal:
        if r["model"] == "softmax" and r["train_seed"] in ("3", "4"):
            continue
        g[(r["scene"], r["model"], r["output"], r["condition"])].append(r)
    rows = []
    conds = [c for c, _ in CONDS]
    for sc in ("pavia_u", "salinas"):
        for m, o in [
            ("softmax", "softmax p"),
            ("softmax", "MC dropout mean p"),
            ("ensemble5", "mean p"),
            ("evidential", "alpha/S"),
            ("nsu", "softmax(zT-zF)"),
            ("nsu", "T/sum(T)"),
            ("hybrid_softmax", "softmax p"),
            ("hybrid_evidential", "alpha/S"),
        ]:
            line = [sc, m, o]
            for c in conds:
                v = g[(sc, m, o, c)]
                line.append(
                    "/".join(
                        f"{np.mean([float(r[k]) for r in v]):.3f}"
                        for k in ("accuracy_argmax", "nll", "ece15")
                    )
                )
            rows.append(line)
    C.write_csv(
        C.REV / "c17_shift_summary.csv",
        ["scene", "model", "output"] + [f"{c} acc/nll/ece (mean over runs)" for c in conds],
        rows,
    )
    with open(C.REV / "c17_shift_ladders.csv", encoding="utf-8") as f:
        L = list(csv.DictReader(f))
    g = defaultdict(list)
    for r in L:
        g[
            (
                r["condition"],
                "HybridSN"
                if r["model"].startswith("hybrid")
                else "ensemble"
                if r["model"] == "ensemble5"
                else "CNN",
            )
        ].append(r)
    out = [
        [c]
        + [
            f"{sum(r['rule_pass'] == 'yes' for r in g[(c, f)])}/{len(g[(c, f)])}"
            for f in ("CNN", "ensemble", "HybridSN")
        ]
        for c in conds
    ]
    C.write_csv(
        C.REV / "c17_shift_ladder_counts.csv",
        ["condition", "CNN_pass", "ensemble_pass", "HybridSN_pass"],
        out,
    )


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "summary":
        summary()
    elif stage == "ood":
        ood()
    elif stage == "gpu":
        gpu_stage()
    elif stage == "cpu":
        jobs = [
            (c, f)
            for c, _ in CONDS
            for f in sorted(os.listdir(os.path.join(C.HSI_REV, "shift", c)))
            if f.endswith(".npz")
        ]
        res = Parallel(n_jobs=16, verbose=2)(delayed(cpu_job)(c, f) for c, f in jobs)
        res += Parallel(n_jobs=8)(delayed(ens_job)(c, sc) for c, _ in CONDS for sc in C.SCENE_LIST)
        C.write_csv(
            C.REV / "c17_shift_calibration.csv",
            [
                "condition",
                "scene",
                "model",
                "output",
                "split_seed",
                "train_seed",
                "accuracy_argmax",
                "nll",
                "brier",
                "ece15",
            ],
            [x for r in res for x in r[0]],
        )
        C.write_csv(
            C.REV / "c17_shift_ladders.csv",
            [
                "condition",
                "scene",
                "model",
                "split_seed",
                "train_seed",
                "ladder",
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
            [x for r in res for x in r[1]],
        )
