"""Table 6 and Fig. 4: SHP. (a) recovery of the presented order by vote-ratio bin (length filter);
(b) near ties (vote ratio < 1.5) vs decisive pairs (>= 3) with the permutation-invariant ladder
|T - F| -> (|T - F|, T + F) -> (|T - F|, T + F, I), under the length filter and with length
covariates.
JSON keys are kept as in the original runs ("над" = "over"; "все данные, длина-ковариата" = "all
data, length covariate")."""

import argparse
import csv
import json

import numpy as np
from common import LEN_BALANCED_MAX_REL, RESULTS_DIR, ROOT, boot_delta, oof_scores, read_jsonl
from sklearn.metrics import roc_auc_score

LOW, HIGH = 1.5, 3.0


def load(scored, ids):
    """Score records with the SHP vote ratio from preference_pair_ids.csv."""
    with open(ids, encoding="utf-8") as f:
        ratio = {
            r["pair_id"]: float(r["score_ratio"]) for r in csv.DictReader(f) if r["sample"] == "shp"
        }
    rows = []
    for r in read_jsonl(scored):
        if r["id"] not in ratio:
            continue
        r["score_ratio"] = ratio[r["id"]]
        rows.append(r)
    return rows


def feats_of(rows, invariant):
    T = np.array([r["chosen"]["mean_lp"] for r in rows])
    F = np.array([r["rejected"]["mean_lp"] for r in rows])
    I = np.array([(r["chosen"]["mean_ent"] + r["rejected"]["mean_ent"]) / 2 for r in rows])
    if invariant:
        m = np.abs(T - F)
        return {
            "|T-F|          [1 dof]": m.reshape(-1, 1),
            "|T-F|, T+F     [2 dof]": np.c_[m, T + F],
            "|T-F|, T+F, I  [3 dof]": np.c_[m, T + F, I],
        }
    return {
        "margin = T-F   [1 dof]": (T - F).reshape(-1, 1),
        "T, F           [2 dof]": np.c_[T, F],
        "T, F, I        [3 dof]": np.c_[T, F, I],
    }


def ladder(rows, y, invariant, len_cov=False):
    n, npos = len(rows), int(y.sum())
    if npos < 25 or n - npos < 25:
        return {}

    feats = feats_of(rows, invariant)
    if len_cov:
        dl = np.array([r["len_chosen"] - r["len_rejected"] for r in rows], float)
        L = np.c_[dl, np.abs(dl)]
        feats = {k: np.c_[v, L] for k, v in feats.items()}

    scores, res = {}, {}
    for name, X in feats.items():
        s = oof_scores(X, y)
        scores[name] = s
        res[name] = float(roc_auc_score(y, s))

    keys = list(feats)
    for a_k, b_k in [(keys[1], keys[0]), (keys[2], keys[1]), (keys[2], keys[0])]:
        b = boot_delta(y, scores[a_k], scores[b_k], a_k, b_k)
        short = f"{a_k.split('[')[0].strip()} над {b_k.split('[')[0].strip()}"
        res[f"delta::{short}"] = b
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default=str(ROOT / "preference_scores" / "ns_logprob_shp.jsonl"))
    ap.add_argument("--ids", default=str(ROOT / "preference_scores" / "preference_pair_ids.csv"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "preference" / "rep_shp_natural.json"))
    args = ap.parse_args()

    rows = load(args.scored, args.ids)

    def rel(r):
        return abs(r["len_chosen"] - r["len_rejected"]) / max(r["len_chosen"], r["len_rejected"], 1)

    bal = [r for r in rows if rel(r) <= LEN_BALANCED_MAX_REL]

    report = {}
    for lo, hi in [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 1e9)]:
        sub = [r for r in bal if lo <= r["score_ratio"] < hi]
        y = np.array([r["corrupt"] for r in sub], int)
        report[f"a::{lo}"] = ladder(sub, y, False)

    y_all = np.array([r["corrupt"] for r in bal], int)
    report["a::all"] = ladder(bal, y_all, False)

    for name, src in [("length-balanced", bal), ("все данные, длина-ковариата", rows)]:
        sub = [r for r in src if r["score_ratio"] < LOW or r["score_ratio"] >= HIGH]
        y = np.array([r["score_ratio"] < LOW for r in sub], int)
        report[f"b::{name}"] = ladder(sub, y, True, len_cov=(name != "length-balanced"))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
