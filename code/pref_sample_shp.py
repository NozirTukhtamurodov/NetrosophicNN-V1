"""SHP sample for Table 6: one pair per post, both comments 40-4000 characters (random.Random(1337)
shuffle);
presentation order swapped for 50% of the pairs to balance position. score_ratio = human vote ratio.
Writes data/preference/train_noisy_shp<suffix>.jsonl (contains text; not redistributed)."""

import argparse
import json
import os
import random

from common import DATA_DIR

SEED = 1337
MIN_LEN = 40
MAX_LEN = 4000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)

    d = load_dataset("stanfordnlp/SHP", split="train")
    idx = list(range(len(d)))
    rng.shuffle(idx)

    rows, seen_posts = [], set()
    for i in idx:
        r = d[i]
        a, b = r["human_ref_A"], r["human_ref_B"]
        if not a or not b:
            continue
        if not (MIN_LEN <= len(a) <= MAX_LEN and MIN_LEN <= len(b) <= MAX_LEN):
            continue
        if r["post_id"] in seen_posts:
            continue
        seen_posts.add(r["post_id"])

        better, worse = (a, b) if r["labels"] == 1 else (b, a)
        sb, sw = (r["score_A"], r["score_B"]) if r["labels"] == 1 else (r["score_B"], r["score_A"])

        rows.append(
            {
                "id": f"shp{len(rows):06d}",
                "prompt": r["history"],
                "_better": better,
                "_worse": worse,
                "score_better": int(sb),
                "score_worse": int(sw),
                "score_ratio": float(r["score_ratio"]),
                "domain": r["domain"].replace("_train", ""),
                "I_consistency": 0.0,
            }
        )
        if len(rows) >= args.n:
            break

    for r in rows:
        swap = rng.random() < 0.5
        r["corrupt"] = swap
        r["chosen"] = r["_worse"] if swap else r["_better"]
        r["rejected"] = r["_better"] if swap else r["_worse"]
        del r["_better"], r["_worse"]

    out_dir = os.path.join(DATA_DIR, "preference")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"train_noisy_shp{args.suffix}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] {path}: {len(rows)} pairs from {len(seen_posts)} posts")


if __name__ == "__main__":
    main()
