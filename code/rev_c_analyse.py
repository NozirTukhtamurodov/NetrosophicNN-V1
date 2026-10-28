"""Revision analyses (post hoc): analysis of the new training runs.
fidelity  retrained replicas vs the stored q1 outputs (Table S18)
c1b       NS-U with explicit head penalty and matched control: residuals, accuracy, NS-U ladder
(Table S12)
c2        alternative evidential objectives: accuracy, evidential ladder, calibration,
diagnostics (Table S13)"""

import glob
import os
import sys

import common as C
import matplotlib
import numpy as np
from rev_calibration_evidence import calib, probs
from rev_engine import extend, oof
from joblib import Parallel, delayed

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = str(C.HSI_REV)


def fidelity():
    rows = []
    for p in sorted(glob.glob(os.path.join(RUNS, "replica", "*.npz"))):
        f = os.path.basename(p)
        a, b = np.load(os.path.join(C.HSI_Q1, f)), np.load(p)
        ks = [k for k in ("logits", "zT", "zF", "mc_mean_p") if k in a.files]
        mx = max(float(np.abs(a[k].astype(float) - b[k].astype(float)).max()) for k in ks)
        rows.append(
            [
                f[:-4],
                f"{(a['pred'] == b['pred']).mean():.6f}",
                f"{(a['pred'] == a['y']).mean():.6f}",
                f"{(b['pred'] == b['y']).mean():.6f}",
                f"{mx:.3e}",
                "yes" if mx == 0 else "no",
            ]
        )
    C.write_csv(
        C.REV / "c21_replica_fidelity.csv",
        ["run", "pred_agreement", "OA_q1", "OA_replica", "max_abs_diff_outputs", "bit_identical"],
        rows,
    )
    print(f"replicas: {len(rows)}, bit-identical: {sum(r[-1] == 'yes' for r in rows)}")


def ladder_rows(vtag, tag, d, model):
    out = []
    for lad in C.ladders_of(model, d)[0]:
        name, y = lad[0], lad[1]
        feats, steps, b6, _ = extend(model, lad)
        sc = {k: oof(X, y)[0] for k, X in feats.items()}
        res, pt = C.boot_dist(y, sc, steps + b6)
        for p in steps + b6:
            sm = C.summ(*res[p])
            out.append(
                [
                    vtag,
                    *tag,
                    name,
                    f"{p[0]}-{p[1]}",
                    "orig" if p in steps else "B6",
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
    return out


def resid(d):
    Ta, Fa = 1 / (1 + np.exp(-d["zT"].astype(float))), 1 / (1 + np.exp(-d["zF"].astype(float)))
    k = (Ta - Fa).argmax(1)
    i = np.arange(len(k))
    r = Ta[i, k] + Fa[i, k] - 1
    a = np.abs(r)
    q = np.percentile(r, [5, 50, 95, 99])
    return [
        f"{a.mean():.5f}",
        f"{np.median(a):.5f}",
        f"{a.std(ddof=1):.5f}",
        *[f"{x:+.5f}" for x in q],
        f"{np.mean(a > 0.05):.4f}",
        f"{np.mean(r > 0):.4f}",
    ]


def c1b_job(p):
    vd = os.path.basename(os.path.dirname(p))
    sc, m, s, t = C.parse(p)
    d = dict(np.load(p))
    oa = float((d["pred"] == d["y"]).mean())
    lc = d["loss_curve"]
    last10 = float(abs(lc[-1] - lc[-11]) / abs(lc[-11])) if len(lc) > 10 else np.nan
    row = [
        vd,
        sc,
        s,
        t,
        f"{float(d['lam']):g}",
        int(d["epochs"]),
        f"{oa:.4f}",
        f"{lc[-1]:.6f}",
        f"{last10:.4f}",
        f"{float(d['head_grad_norm_end']):.3e}",
        f"{float(d['full_train_loss_end_eval_mode']):.6f}",
        *resid(d),
    ]
    return row, ladder_rows(vd, (sc, m, s, t), d, "nsu"), (vd, sc, s, t, lc)


def c2_job(p):
    vd = os.path.basename(os.path.dirname(p))
    sc, m, s, t = C.parse(p)
    d = dict(np.load(p))
    y = d["y"].astype(int)
    r = calib(probs("evidential", d)["alpha/S"], y)
    e = np.logaddexp(0, d["logits"].astype(float))
    tot = e.sum(1)
    row = [
        vd,
        sc,
        s,
        t,
        *[f"{r[k]:.6f}" for k in ("acc", "nll", "brier", "ece")],
        f"{tot.mean():.4f}",
        *[f"{x:.4f}" for x in np.percentile(tot, [5, 50, 95])],
        f"{np.mean((e > 0.01).sum(1) <= 1):.4f}",
        f"{np.mean(tot < 0.01):.4f}",
        f"{np.mean(tot < 0.1):.4f}",
    ]
    pc = [
        [
            vd,
            sc,
            s,
            t,
            k + 1,
            int((y == k).sum()),
            int(d["train_counts"][k]),
            f"{(e[y == k].argmax(1) == k).mean():.4f}",
        ]
        for k in range(e.shape[1])
        if (y == k).any()
    ]
    return row, ladder_rows(vd, (sc, m, s, t), d, "evidential"), pc


LH = [
    "variant",
    "scene",
    "model",
    "split_seed",
    "train_seed",
    "ladder",
    "step",
    "step_kind",
    "n",
    "errors",
    "auc_upper",
    "auc_lower",
    "delta_mean",
    "ci_lo",
    "ci_hi",
    "rule_pass",
]

if __name__ == "__main__":
    what = sys.argv[1]
    if what == "fidelity":
        fidelity()
    if what == "c1b":
        ps = sorted(glob.glob(os.path.join(RUNS, "nsu_pen_lam*", "*.npz")))
        res = Parallel(n_jobs=16)(delayed(c1b_job)(p) for p in ps)
        C.write_csv(
            C.REV / "c05_06_penalised_nsu_by_run.csv",
            [
                "variant",
                "scene",
                "split_seed",
                "train_seed",
                "lambda",
                "epochs",
                "accuracy",
                "final_epoch_loss",
                "rel_loss_change_last10_epochs",
                "head_grad_norm_end_full_batch",
                "full_train_objective_eval_mode",
                "mean_abs_res",
                "median_abs_res",
                "sd_abs_res",
                "p05_res",
                "p50_res",
                "p95_res",
                "p99_res",
                "share_abs_gt0.05",
                "share_TplusF_gt1",
            ],
            [r[0] for r in res],
        )
        C.write_csv(C.REV / "c05_06_penalised_nsu_ladder.csv", LH, [x for r in res for x in r[1]])
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for vd, sc, s, t, lc in [r[2] for r in res]:
            ax[C.SCENE_LIST.index(sc)].plot(
                np.arange(1, len(lc) + 1),
                lc,
                lw=0.6,
                color={
                    "nsu_pen_lam0": "k",
                    "nsu_pen_lam0.0001": "tab:blue",
                    "nsu_pen_lam0.001": "tab:orange",
                    "nsu_pen_lam0.01": "tab:red",
                }[vd],
            )
        for j, sc in enumerate(C.SCENE_LIST):
            ax[j].set_yscale("log")
            ax[j].set_title(
                f"{sc}: training loss (black λ=0, blue 1e-4, orange 1e-3, red 1e-2)", fontsize=8
            )
            ax[j].set_xlabel("epoch")
        fig.tight_layout()
        fig.savefig(os.path.join(C.REV_FIG, "c05_06_penalised_nsu_loss_curves.png"), dpi=300)
        plt.close(fig)
        print(len(ps), "runs")
    if what == "c2":
        ps = sorted(glob.glob(os.path.join(RUNS, "evid_*", "*.npz")))
        res = Parallel(n_jobs=16)(delayed(c2_job)(p) for p in ps)
        C.write_csv(
            C.REV / "c16_alt_evidential_by_run.csv",
            [
                "variant",
                "scene",
                "split_seed",
                "train_seed",
                "accuracy",
                "nll",
                "brier",
                "ece15",
                "mean_S_minus_K",
                "p05_S_minus_K",
                "p50",
                "p95",
                "share_evidence_gt0.01_on_le1_class",
                "share_total_evidence_lt0.01",
                "share_total_evidence_lt0.1",
            ],
            [r[0] for r in res],
        )
        C.write_csv(C.REV / "c16_alt_evidential_ladder.csv", LH, [x for r in res for x in r[1]])
        C.write_csv(
            C.REV / "c16_alt_evidential_perclass_acc.csv",
            [
                "variant",
                "scene",
                "split_seed",
                "train_seed",
                "class",
                "n_test_class",
                "n_train_class",
                "accuracy",
            ],
            [x for r in res for x in r[2]],
        )
        print(len(ps), "runs")
