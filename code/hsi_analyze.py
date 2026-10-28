"""Tables 3 and 4: per-run accuracy and ladder contrasts (per run, pooled per scene x model, deep
ensemble),
from outputs/hsi_outputs_q1/*.npz. Writes C1_runs, C3_deltas, C3_aucs, C5_perclass,
B3_nsu_TF_shares,
B5_ensemble_metrics (results/q1)."""

import argparse
import glob
import os
import time

import numpy as np
from common import (
    HSI_Q1,
    RESULTS_DIR,
    boot_multi,
    ent,
    ladders_of,
    metrics,
    oof_scores,
    parse,
    pred_of,
    softmax,
    write_csv,
)
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score


def single_auc(y, x):
    return float(roc_auc_score(y, oof_scores(x[:, None], y)))


def analyse(tag, ladders):
    rows_d, rows_a = [], []
    for name, y, feats, steps, singles in ladders:
        sc = {k: oof_scores(X, y) for k, X in feats.items()}
        aucs = {k: float(roc_auc_score(y, s)) for k, s in sc.items()}
        bs = boot_multi(y, sc, steps)
        for (a, b), r in bs.items():
            rows_d.append(
                [
                    *tag,
                    name,
                    f"{a}-{b}",
                    len(y),
                    int(y.sum()),
                    f"{aucs[a]:.4f}",
                    f"{aucs[b]:.4f}",
                    f"{r['point']:+.4f}",
                    f"{r['mean']:+.4f}",
                    f"{r['lo']:+.4f}",
                    f"{r['hi']:+.4f}",
                    f"{r['p_gt0']:.3f}",
                    "yes" if (r["mean"] > 0.03 and r["lo"] > 0) else "no",
                    "yes" if (r["point"] > 0.03 and r["lo"] > 0) else "no",
                ]
            )
        for k, v in aucs.items():
            rows_a.append([*tag, name, "rung", k, f"{v:.4f}"])
        for k, v in singles.items():
            rows_a.append([*tag, name, "single(OOF)", k, f"{single_auc(y, v):.4f}"])
    return rows_d, rows_a


def job(path):
    sc, model, s, t = parse(path)
    d = dict(np.load(path))
    m, pc, nte = metrics(d, pred_of(model, d))
    lad, extra = ladders_of(model, d)
    is_member_only = model == "softmax" and t >= 3  # extra ensemble members: no ladder of their own
    rd, ra = ([], []) if is_member_only else analyse((sc, model, s, t), lad)
    row = [
        sc,
        model,
        s,
        t,
        int(d["n_train"]),
        len(d["y"]),
        int((d["train_counts"] == 0).sum()),
        ";".join(str(i + 1) for i in np.flatnonzero(d["train_counts"] == 0)),
        f"{m['OA']:.4f}",
        f"{m['AA']:.4f}",
        f"{m['kappa']:.4f}",
        m["errors"],
        f"{extra.get('mc_oa', float('nan')):.4f}",
        f"{float(d['pca_var']):.4f}",
        int(d["n_params"]),
        f"{float(d['wall_s']):.1f}",
        "ensemble member only" if is_member_only else "",
    ]
    pcr = [
        [
            sc,
            model,
            s,
            t,
            k + 1,
            int(d["train_counts"][k]),
            int(nte[k]),
            "" if np.isnan(pc[k]) else f"{pc[k]:.4f}",
        ]
        for k in range(len(pc))
    ]
    nsx = (
        [
            [
                sc,
                s,
                t,
                f"{extra['share_TF_gt1']:.4f}",
                f"{extra['share_TF_lt1']:.4f}",
                f"{extra['mean_TF']:.4f}",
            ]
        ]
        if model == "nsu"
        else []
    )
    return row, rd, ra, pcr, nsx, (sc, model, s, t, lad if not is_member_only else None)


def pool_job(key, lads):
    """Pooled contrast: test pixels of all runs of one scene x model concatenated."""
    merged = []
    for li in range(len(lads[0])):
        name = lads[0][li][0]
        y = np.concatenate([L[li][1] for L in lads])
        feats = {k: np.concatenate([L[li][2][k] for L in lads]) for k in lads[0][li][2]}
        singles = {k: np.concatenate([L[li][4][k] for L in lads]) for k in lads[0][li][4]}
        merged.append((name, y, feats, lads[0][li][3], singles))
    return analyse((key[0], key[1], "pooled", f"all({len(lads)})"), merged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS_DIR / "q1"))
    out = ap.parse_args().out
    t0 = time.time()
    paths = sorted(glob.glob(os.path.join(HSI_Q1, "*.npz")))
    print(f"{len(paths)} runs", flush=True)
    res = Parallel(n_jobs=12, verbose=5)(delayed(job)(p) for p in paths)
    runs = [r[0] for r in res]
    D = [x for r in res for x in r[1]]
    A = [x for r in res for x in r[2]]
    PC = [x for r in res for x in r[3]]
    NS = [x for r in res for x in r[4]]

    groups = {}
    for r in res:
        sc, model, s, t, lad = r[5]
        if lad is None:
            continue
        groups.setdefault((sc, model), []).append(lad)
    pres = Parallel(n_jobs=8, verbose=5)(delayed(pool_job)(k, v) for k, v in groups.items())
    for rd, ra in pres:
        D += rd
        A += ra

    # deep ensemble: split 0, training seeds 0..4
    ens = []
    for sc in ["pavia_u", "salinas"]:
        fs = [os.path.join(HSI_Q1, f"{sc}_softmax_s0_t{t}.npz") for t in range(5)]
        if not all(os.path.exists(f) for f in fs):
            continue
        ds = [dict(np.load(f)) for f in fs]
        P = np.stack([softmax(d["logits"]) for d in ds])
        pm = P.mean(0)
        yt = ds[0]["y"].astype(int)
        pred = pm.argmax(1)
        H = ent(pm)
        MI = H - np.mean([ent(p) for p in P], 0)
        y = (pred != yt).astype(int)
        m, pc, nte = metrics({"y": yt, "K": pm.shape[1]}, pred)
        ens.append([sc, 5, f"{m['OA']:.4f}", f"{m['AA']:.4f}", f"{m['kappa']:.4f}", m["errors"]])
        rd, ra = analyse(
            (sc, "ensemble5", 0, "0-4"),
            [
                (
                    "E",
                    y,
                    {"E1": H[:, None], "E2": np.c_[H, MI], "E3": np.c_[H, MI, pm.max(1)]},
                    [("E2", "E1"), ("E3", "E2")],
                    {"H_mean": H, "MI": MI, "p_max_mean": pm.max(1)},
                )
            ],
        )
        D += rd
        A += ra

    write_csv(
        os.path.join(out, "C1_runs.csv"),
        [
            "scene",
            "model",
            "split_seed",
            "train_seed",
            "n_train",
            "n_test",
            "n_classes_no_train_px",
            "classes_no_train_px",
            "OA",
            "AA",
            "kappa",
            "errors",
            "OA_mc_dropout_mean",
            "pca_explained_var",
            "n_params",
            "wall_s",
            "note",
        ],
        runs,
    )
    write_csv(
        os.path.join(out, "C3_deltas.csv"),
        [
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
            "delta_point",
            "delta_boot_mean",
            "ci_lo",
            "ci_hi",
            "p_gt0",
            "pass_bootmean_rule",
            "pass_point_rule",
        ],
        D,
    )
    write_csv(
        os.path.join(out, "C3_aucs.csv"),
        ["scene", "model", "split_seed", "train_seed", "ladder", "kind", "feature_or_rung", "auc"],
        A,
    )
    write_csv(
        os.path.join(out, "C5_perclass.csv"),
        [
            "scene",
            "model",
            "split_seed",
            "train_seed",
            "class",
            "n_train_class",
            "n_test_class",
            "accuracy",
        ],
        PC,
    )
    write_csv(
        os.path.join(out, "B3_nsu_TF_shares.csv"),
        [
            "scene",
            "split_seed",
            "train_seed",
            "share_T_plus_F_gt1",
            "share_T_plus_F_lt1",
            "mean_T_plus_F",
        ],
        NS,
    )
    write_csv(
        os.path.join(out, "B5_ensemble_metrics.csv"),
        ["scene", "members", "OA", "AA", "kappa", "errors"],
        ens,
    )
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
