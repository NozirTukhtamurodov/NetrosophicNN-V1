"""Table 5: swapped-pair detection on the preference data. Ladder margin T - F -> (T, F) -> (T, F,
I), with
T, F = per-token mean log-probabilities of the presented chosen/rejected response and I = mean of
their per-token
entropies; on all pairs, under the length filter, and with length covariates. Writes the report
JSON.
JSON keys are kept as in the original runs ("над" = "over", "одна" = "alone", "справ." =
"reference")."""

import argparse
import json

import numpy as np
from common import LEN_BALANCED_MAX_REL, boot_delta, oof_scores, read_jsonl
from sklearn.metrics import roc_auc_score


def block(rows, len_covariate=False):
    """len_covariate: [l_c - l_r, |l_c - l_r|] appended to every rung instead of filtering by
    length."""
    y = np.array([r["corrupt"] for r in rows], int)
    n, npos = len(rows), int(y.sum())
    if npos < 10 or n - npos < 10:
        return {}

    T = np.array([r["chosen"]["mean_lp"] for r in rows])
    F = np.array([r["rejected"]["mean_lp"] for r in rows])
    I = np.array([(r["chosen"]["mean_ent"] + r["rejected"]["mean_ent"]) / 2 for r in rows])
    leak = np.array([r["len_chosen"] - r["len_rejected"] for r in rows], dtype=float)

    feats = {
        "margin = T-F  [1 dof]": (T - F).reshape(-1, 1),
        "T, F          [2 dof]": np.c_[T, F],
        "T, F, I       [3 dof]": np.c_[T, F, I],
        "I одна        [справ.]": I.reshape(-1, 1),
    }
    if len_covariate:
        L = np.c_[leak, np.abs(leak)]
        feats = {k: np.c_[v, L] for k, v in feats.items()}
        feats["только длина [справ.]"] = L

    scores, res = {}, {}
    for name, X in feats.items():
        s = oof_scores(X, y)
        scores[name] = s
        res[name] = float(roc_auc_score(y, s))

    res["length_leak"] = float(roc_auc_score(y, leak))

    keys = list(feats)
    pairs = [(keys[1], keys[0]), (keys[2], keys[1]), (keys[2], keys[0])]
    for a_k, b_k in pairs:
        b = boot_delta(y, scores[a_k], scores[b_k], a_k, b_k)
        short = f"{a_k.split('[')[0].strip()} над {b_k.split('[')[0].strip()}"
        res[f"delta::{short}"] = b
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data", required=True, help="numbers-only score file (preference_scores/*.jsonl)"
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.data)

    def rel(r):
        return abs(r["len_chosen"] - r["len_rejected"]) / max(r["len_chosen"], r["len_rejected"], 1)

    bal = [r for r in rows if rel(r) <= LEN_BALANCED_MAX_REL]
    report = {
        "all": block(rows),
        "length_balanced": block(bal),
        "length_covariate": block(rows, len_covariate=True),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    for blk, v in report.items():
        for k, d in v.items():
            if k.startswith("delta::"):
                print(f"{blk:<17} {k[7:]:<34} {d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]")


if __name__ == "__main__":
    main()
