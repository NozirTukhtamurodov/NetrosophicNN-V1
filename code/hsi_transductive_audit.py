"""Section 6.5: OA/AA/kappa and the evidential ladder (R2-R1, R3+-R2, R3-R2) of the transductive
runs.
Inputs outputs/hsi_outputs_transductive/*.jsonl; writes results/q1/transductive_a.json and
transductive_b6.json."""

import argparse
import glob
import json
import os

import numpy as np
from common import OUTPUTS_DIR, RESULTS_DIR, boot_delta, load_scene, oof_scores, split_block_buffer
from sklearn.metrics import roc_auc_score

DATA = os.path.join(OUTPUTS_DIR, "hsi_outputs_transductive")


def accuracy_summary():
    """OA, AA, kappa per scene x seed and for the concatenation of the three splits; split
    recomputed."""
    summary = {}
    for scene in ["pavia_u", "salinas"]:
        cube, gt = load_scene(scene)
        n_cls = int(gt.max())
        pooled = []
        for seed in (0, 1, 2):
            rng = np.random.default_rng(seed)
            tr, te = split_block_buffer(gt, 0.15, rng, buffer=6)
            p = os.path.join(DATA, f"hsi_{scene}_s{seed}.jsonl")
            with open(p, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            y = np.array([r["y"] for r in rows])
            pr = np.array([r["pred"] for r in rows])
            err = int((y != pr).sum())
            n = len(rows)
            oa = 1 - err / n
            cm = np.zeros((n_cls, n_cls), int)
            for a, b in zip(y, pr, strict=True):
                cm[a, b] += 1
            rec = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(n_cls), where=cm.sum(1) > 0)
            aa = rec[cm.sum(1) > 0].mean()
            pe = (cm.sum(0) * cm.sum(1)).sum() / n**2
            kap = (oa - pe) / (1 - pe)
            pooled += rows
            summary[f"{scene}_s{seed}"] = dict(
                train=int(tr.sum()),
                test=n,
                err=err,
                oa=round(oa, 4),
                aa=round(aa, 4),
                kappa=round(kap, 4),
            )
        y = np.array([r["y"] for r in pooled])
        pr = np.array([r["pred"] for r in pooled])
        err = int((y != pr).sum())
        n = len(pooled)
        oa = 1 - err / n
        cm = np.zeros((n_cls, n_cls), int)
        for a, b in zip(y, pr, strict=True):
            cm[a, b] += 1
        rec = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(n_cls), where=cm.sum(1) > 0)
        aa = rec[cm.sum(1) > 0].mean()
        pe = (cm.sum(0) * cm.sum(1)).sum() / n**2
        kap = (oa - pe) / (1 - pe)
        mean_oa = np.mean([summary[f"{scene}_s{s}"]["oa"] for s in (0, 1, 2)])
        summary[f"{scene}_pooled"] = dict(
            test=n,
            err=err,
            oa=round(oa, 4),
            aa=round(aa, 4),
            kappa=round(kap, 4),
            mean_oa_over_splits=round(float(mean_oa), 4),
        )
    return summary


def ladder(rows):
    y = np.array([r["err"] for r in rows], int)
    ent = np.array([r["entropy"] for r in rows])
    u = np.array([r["u"] for r in rows])
    bm = np.array([r["b_max"] for r in rows])
    b2 = np.array([r["b_2nd"] for r in rows])
    feats = {
        "R1  H                    [1]": ent.reshape(-1, 1),
        "R2  H,u                  [2]": np.c_[ent, u],
        "R3  b_max,b_2,u          [3]": np.c_[bm, b2, u],
        "R3+ H,u,b_max,b_2        [4]": np.c_[ent, u, bm, b2],
    }
    s, auc = {}, {}
    for k, X in feats.items():
        s[k] = oof_scores(X, y)
        auc[k] = float(roc_auc_score(y, s[k]))
    res = {"n": int(len(y)), "n_err": int(y.sum()), "auc": {k: auc[k] for k in feats}, "delta": {}}
    pairs = [
        ("R3+ H,u,b_max,b_2        [4]", "R2  H,u                  [2]", "ВЛОЖЕННЫЙ R3+ - R2"),
        ("R3  b_max,b_2,u          [3]", "R2  H,u                  [2]", "невложенный R3 - R2"),
        ("R2  H,u                  [2]", "R1  H                    [1]", "R2 - R1 (свидетельство)"),
    ]
    for (
        a,
        b,
        lbl,
    ) in pairs:  # JSON keys kept as in the original runs (nested / non-nested / evidence step)
        res["delta"][lbl] = boot_delta(y, s[a], s[b], a, b)
    return res


def ladder_summary():
    out = {}
    for scene in ["pavia_u", "salinas"]:
        pooled = []
        for p in sorted(glob.glob(os.path.join(DATA, f"hsi_{scene}_s*.jsonl"))):
            seed = os.path.basename(p).split("_s")[-1].split(".")[0]
            with open(p, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            out[f"{scene}_s{seed}"] = ladder(rows)
            pooled += rows
        out[f"{scene}_pooled"] = ladder(pooled)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS_DIR / "q1"))
    out = ap.parse_args().out
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "transductive_a.json"), "w", encoding="utf-8") as f:
        json.dump(accuracy_summary(), f, indent=2)
    with open(os.path.join(out, "transductive_b6.json"), "w", encoding="utf-8") as f:
        json.dump(ladder_summary(), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
