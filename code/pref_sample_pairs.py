"""Preference samples for Table 5: draw pairs from orca_dpo_pairs, UltraFeedback or
Anthropic/hh-rlhf
(random.Random(1337) shuffle of the row indices) and swap chosen/rejected in 30% of them (target
y = swapped).
Writes data/preference/train_noisy_<tag><suffix>.jsonl (contains text; not redistributed)."""

import argparse
import json
import os
import random

from common import DATA_DIR

FLIP_RATE = 0.30
SEED = 1337


def _last_turn(text, marker="\n\nAssistant:"):
    """HH-RLHF dialogue string -> (context, last assistant turn)."""
    i = text.rfind(marker)
    if i < 0:
        return None, None
    return text[:i], text[i + len(marker) :].strip()


def load_pairs(tag, n_target, rng, min_gap=0.0):
    from datasets import load_dataset

    out = []
    if tag == "ultrafeedback":
        d = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
        idx = list(range(len(d)))
        rng.shuffle(idx)
        for i in idx:
            r = d[i]
            if r["score_chosen"] is None or r["score_rejected"] is None:
                continue
            gap = r["score_chosen"] - r["score_rejected"]
            if gap <= 0 or gap < min_gap:  # ties are not preference pairs
                continue
            ch = [m for m in r["chosen"] if m["role"] == "assistant"]
            rj = [m for m in r["rejected"] if m["role"] == "assistant"]
            if not ch or not rj:
                continue
            out.append((r["prompt"], ch[-1]["content"], rj[-1]["content"], gap))
            if len(out) >= n_target:
                break

    elif tag == "orca":
        d = load_dataset("Intel/orca_dpo_pairs", split="train")
        idx = list(range(len(d)))
        rng.shuffle(idx)
        for i in idx:
            r = d[i]
            p = f"{r['system']}\n\n{r['question']}" if r.get("system") else r["question"]
            if not r["chosen"] or not r["rejected"] or r["chosen"] == r["rejected"]:
                continue
            out.append((p, r["chosen"], r["rejected"], None))
            if len(out) >= n_target:
                break

    elif tag == "hh_rlhf":
        d = load_dataset("Anthropic/hh-rlhf", split="train")
        idx = list(range(len(d)))
        rng.shuffle(idx)
        for i in idx:
            r = d[i]
            pc, ac = _last_turn(r["chosen"])
            pr, ar = _last_turn(r["rejected"])
            # keep only pairs that share the dialogue and differ in the last assistant turn
            if pc is None or pr is None or pc != pr:
                continue
            if not ac or not ar or ac == ar:
                continue
            out.append((pc.strip(), ac, ar, None))
            if len(out) >= n_target:
                break
    else:
        raise ValueError(f"unknown tag: {tag}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["orca", "ultrafeedback", "hh_rlhf"])
    ap.add_argument("-n", type=int, default=2000)
    ap.add_argument("--flip-rate", type=float, default=FLIP_RATE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--min-gap", type=float, default=0.0, help="ultrafeedback only: minimum rating gap"
    )
    ap.add_argument("--suffix", default="", help="file-name suffix")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pairs = load_pairs(args.tag, args.n, rng, args.min_gap)
    print(f"[data] {args.tag}: {len(pairs)} pairs (min_gap={args.min_gap})")

    rows = [
        {
            "id": f"{args.tag[:2]}{i:05d}",
            "prompt": p,
            "chosen": c,
            "rejected": j,
            "score_gap": g,
            "I_consistency": 0.0,  # unused placeholder kept for file-format compatibility
        }
        for i, (p, c, j, g) in enumerate(pairs)
    ]

    n_flip = round(args.flip_rate * len(rows))
    flip = set(rng.sample([r["id"] for r in rows], n_flip))
    for r in rows:
        r["corrupt"] = r["id"] in flip
        if r["corrupt"]:
            r["chosen"], r["rejected"] = r["rejected"], r["chosen"]

    out_dir = os.path.join(DATA_DIR, "preference")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"train_noisy_{args.tag}{args.suffix}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] {path}: {len(rows)} pairs, swapped {n_flip} ({n_flip / len(rows):.0%})")


if __name__ == "__main__":
    main()
