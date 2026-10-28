"""Inputs of Figs. 2-4 from the stored q1 tables and the preference-detector reports."""

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("NSDOF_RESULTS_DIR", ROOT / "results"))
Q1 = RESULTS / "q1"
PREF = RESULTS / "preference"
SCORES = ROOT / "preference_scores"
LEN_REL = 0.25  # length filter |l_c - l_r| / max(l_c, l_r) <= 0.25 (Tables 5-6)

NOISY = [  # label, detector report, score file
    ("orca", "rep_orca_big.json", "ns_logprob_orca_big.jsonl"),
    ("UF gap ≥ 4", "rep_ultrafeedback_gap4big.json", "ns_logprob_ultrafeedback_gap4big.jsonl"),
    ("UF gap ≥ 2", "rep_ultrafeedback_gap2big.json", "ns_logprob_ultrafeedback_gap2big.jsonl"),
    ("HH-RLHF", "rep_hh_rlhf_big.json", "ns_logprob_hh_rlhf_big.jsonl"),
]


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def rel_len(r):
    return abs(r["len_chosen"] - r["len_rejected"]) / max(r["len_chosen"], r["len_rejected"], 1)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def delta(report, block, key):
    """Bootstrap mean and 95% CI of one ladder step from a detector report."""
    d = next(
        v for k, v in report[block].items() if k.startswith("delta::") and k[7:].startswith(key)
    )
    return d["mean"], d["lo"], d["hi"]


def hsi_inputs():
    lad = [r for r in read_csv(Q1 / "C3_deltas.csv") if r["split_seed"] != "pooled"]
    write_csv(
        Q1 / "ladders_runs.csv",
        ["scene", "model", "step", "split_seed", "delta_boot_mean"],
        [[r["scene"], r["model"], r["step"], r["split_seed"], r["delta_boot_mean"]] for r in lad],
    )
    exp = [r for r in read_csv(Q1 / "X_nsu_exploratory.csv") if r["split"] != "pooled"]
    write_csv(
        Q1 / "nsu_explore.csv",
        ["scene", "step", "split", "delta_boot_mean"],
        [[r["scene"], r["step"], r["split"], r["delta_boot_mean"]] for r in exp],
    )
    # float32 share at k* = argmax(T - F), as computed by analyze_hsi.py
    nsu = read_csv(Q1 / "B3_nsu_TF_shares.csv")
    write_csv(
        Q1 / "nsu_sum_by_run.csv",
        ["scene", "split_seed", "train_seed", "share_TplusF_gt_1", "mean_TplusF"],
        [
            [
                r["scene"],
                r["split_seed"],
                r["train_seed"],
                r["share_T_plus_F_gt1"],
                r["mean_T_plus_F"],
            ]
            for r in nsu
        ],
    )


def preference_inputs():
    rows = []
    for label, rep, scores in NOISY:
        report = json.loads((PREF / rep).read_text(encoding="utf-8"))
        s = read_jsonl(SCORES / scores)
        n_filter, n_cov = sum(rel_len(r) <= LEN_REL for r in s), len(s)
        for block, ctl, n in (
            ("length_balanced", "length filter", n_filter),
            ("length_covariate", "covariate", n_cov),
        ):
            m, lo, hi = delta(report, block, "T, F, I над T, F")
            rows.append(["indeterminacy", f"{label}, {ctl} (n = {n:,})", m, lo, hi])
    shp = json.loads((PREF / "rep_shp_natural.json").read_text(encoding="utf-8"))
    ratio = {
        r["pair_id"]: float(r["score_ratio"])
        for r in read_csv(SCORES / "preference_pair_ids.csv")
        if r["sample"] == "shp"
    }
    s = [
        r
        for r in read_jsonl(SCORES / "ns_logprob_shp.jsonl")
        if ratio[r["id"]] < 1.5 or ratio[r["id"]] >= 3.0
    ]
    n_shp = {
        "b::length-balanced": sum(rel_len(r) <= LEN_REL for r in s),
        "b::все данные, длина-ковариата": len(s),
    }
    for panel, key in (
        ("indeterminacy", "|T-F|, T+F, I над |T-F|, T+F"),
        ("evidence", "|T-F|, T+F над |T-F|"),
    ):
        for block, ctl in (
            ("b::length-balanced", "length filter"),
            ("b::все данные, длина-ковариата", "covariate"),
        ):
            m, lo, hi = delta(shp, block, key)
            rows.append([panel, f"SHP near-ties, {ctl} (n = {n_shp[block]:,})", m, lo, hi])
    write_csv(PREF / "preference_steps.csv", ["panel", "label", "delta", "ci_lo", "ci_hi"], rows)


if __name__ == "__main__":
    hsi_inputs()
    preference_inputs()
