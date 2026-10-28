"""Tables 5-6: point delta-AUC of each preference ladder step under five fold assignments (fold
seeds 0-4);
fold seed 0 reproduces the stored reports. Writes
results/preference/D1_fold_seed_variability.csv."""

import argparse
import csv
import json

import numpy as np
from common import LEN_BALANCED_MAX_REL, RESULTS_DIR, ROOT, oof_scores, read_jsonl, write_csv
from sklearn.metrics import roc_auc_score

SEEDS = [0, 1, 2, 3, 4]
SCORES = ROOT / "preference_scores"
REPORTS = RESULTS_DIR / "preference"
RUNS = [  # dataset, scorer, score file, stored report
    ("orca n487-base", "Qwen2.5-1.5B", "ns_logprob.jsonl", "ns_logprob_report.json"),
    ("orca n487-base", "Qwen2.5-0.5B", "ns_logprob_orca__qwen05.jsonl", "rep_orca__qwen05.json"),
    ("orca n487-base", "SmolLM2-1.7B", "ns_logprob_orca__smol17.jsonl", "rep_orca__smol17.json"),
    (
        "UF gap>=4 (2000)",
        "Qwen2.5-1.5B",
        "ns_logprob_ultrafeedback_gap4.jsonl",
        "ns_logprob_report_uf_gap4.json",
    ),
    (
        "UF gap>=4 (2000)",
        "Qwen2.5-0.5B",
        "ns_logprob_ultrafeedback_gap4__qwen05.jsonl",
        "rep_ultrafeedback_gap4__qwen05.json",
    ),
    (
        "UF gap>=4 (2000)",
        "SmolLM2-1.7B",
        "ns_logprob_ultrafeedback_gap4__smol17.jsonl",
        "rep_ultrafeedback_gap4__smol17.json",
    ),
    ("orca big", "Qwen2.5-1.5B", "ns_logprob_orca_big.jsonl", "rep_orca_big.json"),
    (
        "UF gap>=4 big",
        "Qwen2.5-1.5B",
        "ns_logprob_ultrafeedback_gap4big.jsonl",
        "rep_ultrafeedback_gap4big.json",
    ),
    (
        "UF gap>=2 big",
        "Qwen2.5-1.5B",
        "ns_logprob_ultrafeedback_gap2big.jsonl",
        "rep_ultrafeedback_gap2big.json",
    ),
    ("HH-RLHF big", "Qwen2.5-1.5B", "ns_logprob_hh_rlhf_big.jsonl", "rep_hh_rlhf_big.json"),
]
KEYMAP = {
    "2over1": "delta::T, F над margin = T-F",
    "3over2": "delta::T, F, I над T, F",
    "3over1": "delta::T, F, I над margin = T-F",
}
KEYMAP_INV = {
    "2over1": "delta::|T-F|, T+F над |T-F|",
    "3over2": "delta::|T-F|, T+F, I над |T-F|, T+F",
    "3over1": "delta::|T-F|, T+F, I над |T-F|",
}


def rel(r):
    return abs(r["len_chosen"] - r["len_rejected"]) / max(r["len_chosen"], r["len_rejected"], 1)


def feats_std(rows, cov):
    T = np.array([r["chosen"]["mean_lp"] for r in rows])
    F = np.array([r["rejected"]["mean_lp"] for r in rows])
    I = np.array([(r["chosen"]["mean_ent"] + r["rejected"]["mean_ent"]) / 2 for r in rows])
    f = {"1": (T - F).reshape(-1, 1), "2": np.c_[T, F], "3": np.c_[T, F, I]}
    if cov:
        d = np.array([r["len_chosen"] - r["len_rejected"] for r in rows], float)
        f = {k: np.c_[v, d, np.abs(d)] for k, v in f.items()}
    return f


def feats_inv(rows, cov):
    T = np.array([r["chosen"]["mean_lp"] for r in rows])
    F = np.array([r["rejected"]["mean_lp"] for r in rows])
    I = np.array([(r["chosen"]["mean_ent"] + r["rejected"]["mean_ent"]) / 2 for r in rows])
    m = np.abs(T - F)
    f = {"1": m.reshape(-1, 1), "2": np.c_[m, T + F], "3": np.c_[m, T + F, I]}
    if cov:
        d = np.array([r["len_chosen"] - r["len_rejected"] for r in rows], float)
        f = {k: np.c_[v, d, np.abs(d)] for k, v in f.items()}
    return f


def run_block(y, fdict):
    out = {}
    for s in SEEDS:
        a = {k: roc_auc_score(y, oof_scores(X, y, seed=s)) for k, X in fdict.items()}
        out[s] = {"2over1": a["2"] - a["1"], "3over2": a["3"] - a["2"], "3over1": a["3"] - a["1"]}
    return out


def summary_row(head, v, stored):
    return (
        head
        + [f"{x:+.4f}" for x in v]
        + [
            f"{np.mean(v):+.4f}",
            f"{np.std(v, ddof=1):.4f}",
            f"{min(v):+.4f}",
            f"{max(v):+.4f}",
            f"{stored:+.4f}",
            f"{abs(v[0] - stored):.1e}",
        ]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPORTS / "D1_fold_seed_variability.csv"))
    out = ap.parse_args().out
    rows_out = []
    for ds, scorer, scored, rep in RUNS:
        R = read_jsonl(SCORES / scored)
        with open(REPORTS / rep, encoding="utf-8") as f:
            stored = json.load(f)
        for blk in stored:
            sub = (
                R if blk != "length_balanced" else [r for r in R if rel(r) <= LEN_BALANCED_MAX_REL]
            )
            y = np.array([r["corrupt"] for r in sub], int)
            res = run_block(y, feats_std(sub, blk == "length_covariate"))
            for st, kk in KEYMAP.items():
                v = [res[s][st] for s in SEEDS]
                rows_out.append(
                    summary_row([ds, scorer, blk, st, len(sub)], v, stored[blk][kk]["mean"])
                )

    with open(SCORES / "preference_pair_ids.csv", encoding="utf-8") as f:
        ratio = {
            r["pair_id"]: float(r["score_ratio"]) for r in csv.DictReader(f) if r["sample"] == "shp"
        }
    C = read_jsonl(SCORES / "ns_logprob_shp.jsonl")
    for r in C:
        r["score_ratio"] = ratio[r["id"]]
    bal = [r for r in C if rel(r) <= LEN_BALANCED_MAX_REL]
    with open(REPORTS / "rep_shp_natural.json", encoding="utf-8") as f:
        shp = json.load(f)
    for lo, hi in [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 1e9)]:
        sub = [r for r in bal if lo <= r["score_ratio"] < hi]
        y = np.array([r["corrupt"] for r in sub], int)
        res = run_block(y, feats_std(sub, False))
        for st, kk in KEYMAP.items():
            v = [res[s][st] for s in SEEDS]
            sv = shp[f"a::{lo}"][kk]["mean"] if f"a::{lo}" in shp else float("nan")
            rows_out.append(
                summary_row(
                    ["SHP E4a", "Qwen2.5-1.5B", f"length_balanced ratio[{lo},{hi})", st, len(sub)],
                    v,
                    sv,
                )
            )
    y = np.array([r["corrupt"] for r in bal], int)
    res = run_block(y, feats_std(bal, False))
    for st, kk in KEYMAP.items():
        v = [res[s][st] for s in SEEDS]
        rows_out.append(
            summary_row(
                ["SHP E4a", "Qwen2.5-1.5B", "length_balanced all ratios", st, len(bal)],
                v,
                shp["a::all"][kk]["mean"],
            )
        )
    for name, src, key in [
        ("length_balanced", bal, "b::length-balanced"),
        ("length_covariate", C, "b::все данные, длина-ковариата"),
    ]:
        sub = [r for r in src if r["score_ratio"] < 1.5 or r["score_ratio"] >= 3.0]
        y = np.array([r["score_ratio"] < 1.5 for r in sub], int)
        res = run_block(y, feats_inv(sub, name == "length_covariate"))
        for st, kk in KEYMAP_INV.items():
            v = [res[s][st] for s in SEEDS]
            rows_out.append(
                summary_row(
                    ["SHP E4b near-tie", "Qwen2.5-1.5B", name, st, len(sub)],
                    v,
                    shp[key][kk]["mean"],
                )
            )

    write_csv(
        out,
        [
            "dataset",
            "scorer",
            "block",
            "step",
            "n",
            "d_fold0",
            "d_fold1",
            "d_fold2",
            "d_fold3",
            "d_fold4",
            "mean",
            "sd_ddof1",
            "min",
            "max",
            "stored_d",
            "abs_dev_fold0_vs_stored",
        ],
        rows_out,
    )


if __name__ == "__main__":
    main()
