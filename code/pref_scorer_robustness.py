"""Section 7.2: the two smaller samples (orca, 487 pairs; UltraFeedback gap >= 4, 2,000 pairs)
scored with
Qwen2.5-1.5B-Instruct, Qwen2.5-0.5B-Instruct and SmolLM2-1.7B-Instruct: count of ladder
comparisons and of those
satisfying rule (10), from the stored detector reports. Writes
results/preference/scorer_robustness.csv."""

import json

from common import RESULTS_DIR, write_csv

REPORTS = RESULTS_DIR / "preference"
FILES = [
    ("orca n487-base", "Qwen2.5-1.5B-Instruct", "ns_logprob_report.json"),
    ("orca n487-base", "Qwen2.5-0.5B-Instruct", "rep_orca__qwen05.json"),
    ("orca n487-base", "SmolLM2-1.7B-Instruct", "rep_orca__smol17.json"),
    ("UF gap>=4 (2000)", "Qwen2.5-1.5B-Instruct", "ns_logprob_report_uf_gap4.json"),
    ("UF gap>=4 (2000)", "Qwen2.5-0.5B-Instruct", "rep_ultrafeedback_gap4__qwen05.json"),
    ("UF gap>=4 (2000)", "SmolLM2-1.7B-Instruct", "rep_ultrafeedback_gap4__smol17.json"),
]


def main():
    rows = []
    for sample, scorer, fn in FILES:
        with open(REPORTS / fn, encoding="utf-8") as f:
            rep = json.load(f)
        for blk, v in rep.items():
            for k, d in v.items():
                if k.startswith("delta::"):
                    rows.append(
                        [
                            sample,
                            scorer,
                            blk,
                            k[7:],
                            f"{d['mean']:+.4f}",
                            f"{d['lo']:+.4f}",
                            f"{d['hi']:+.4f}",
                            "yes" if (d["mean"] > 0.03 and d["lo"] > 0) else "no",
                        ]
                    )
    write_csv(
        REPORTS / "scorer_robustness.csv",
        ["sample", "scorer", "control", "step", "delta_mean", "ci_lo", "ci_hi", "rule_pass"],
        rows,
    )
    print(f"{len(rows)} comparisons, {sum(r[-1] == 'yes' for r in rows)} satisfy rule (10)")


if __name__ == "__main__":
    main()
