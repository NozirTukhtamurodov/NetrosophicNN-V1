"""Revision analyses (post hoc), comment 14 (A3): list of all 224 comparisons (196 per run + 28
pooled)."""

import csv

import common as C

with open(C.RESULTS_DIR / "q1" / "C3_deltas.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
out = []
for i, r in enumerate(rows, 1):
    pooled = r["split_seed"] == "pooled"
    fam = (
        "HybridSN"
        if r["model"].startswith("hybrid")
        else "ensemble"
        if r["model"] == "ensemble5"
        else "CNN"
    )
    degen = r["model"] == "hybrid_evidential"
    out.append(
        [
            i,
            "pooled" if pooled else "per-run",
            fam,
            r["scene"],
            r["model"],
            r["ladder"],
            r["split_seed"],
            r["train_seed"],
            r["step"],
            "yes" if degen else "",
            r["delta_boot_mean"],
            r["ci_lo"],
            r["ci_hi"],
            r["pass_bootmean_rule"],
        ]
    )
C.write_csv(
    C.REV / "c14_all_comparisons.csv",
    [
        "id",
        "kind",
        "family",
        "scene",
        "model",
        "ladder",
        "split_seed",
        "train_seed",
        "step",
        "degenerate_evidential_hybridsn",
        "delta_mean",
        "ci_lo",
        "ci_hi",
        "rule_pass",
    ],
    out,
)
pr = [o for o in out if o[1] == "per-run"]
print(
    f"{len(out)} comparisons: {len(pr)} per run, {len(out) - len(pr)} pooled; "
    f"rule (10) passes: {sum(o[13] == 'yes' for o in pr)} per run"
)
