"""Preference scoring (Section 7, Tables 5-6): teacher-forced per-token log-probabilities and
entropies of each
response given the prompt, with a fixed scorer (default Qwen/Qwen2.5-1.5B-Instruct). No chat
template;
only response tokens are used. Writes one numbers-only record per pair (resumable)."""

import argparse
import json
import os

import torch
import torch.nn.functional as F_
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_PROMPT = 512
MAX_ANSWER = 512


@torch.no_grad()
def score(model, tok, prompt, answer, device):
    """Per-response mean log p(token), its lower quartile, mean and 90th-percentile next-token
    entropy (nats)
    over the response tokens; None if the response has fewer than 2 tokens (pair excluded)."""
    p_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT).input_ids
    a_ids = tok(
        answer,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_ANSWER,
        add_special_tokens=False,
    ).input_ids
    if a_ids.shape[1] < 2:
        return None

    ids = torch.cat([p_ids, a_ids], dim=1).to(device)
    logits = model(ids).logits[0].float()  # (L, V)

    start = p_ids.shape[1]
    tgt = ids[0, start:]  # response tokens
    lg = logits[start - 1 : -1]  # position t-1 predicts token t
    logp = F_.log_softmax(lg, dim=-1)

    tok_lp = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)  # (n,)
    ent = -(logp.exp() * logp).sum(dim=-1)  # (n,)

    return {
        "mean_lp": float(tok_lp.mean()),
        "q25_lp": float(tok_lp.quantile(0.25)),
        "mean_ent": float(ent.mean()),
        "q90_ent": float(ent.quantile(0.90)),
        "n_tok": int(tgt.numel()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--data", required=True, help="train_noisy_*.jsonl written by pref_sample_*.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except json.JSONDecodeError:
                    pass
    todo = [r for r in rows if r["id"] not in done]
    print(f"[data] {len(rows)} pairs, done {len(done)}, to do {len(todo)}")
    if not todo:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(device).eval()

    with open(args.out, "a", encoding="utf-8") as f:
        for r in todo:
            sc = score(model, tok, r["prompt"], r["chosen"], device)
            sr = score(model, tok, r["prompt"], r["rejected"], device)
            if sc is None or sr is None:
                continue
            rec = {
                "id": r["id"],
                "corrupt": r["corrupt"],
                "I_consistency": r["I_consistency"],
                "len_chosen": len(r["chosen"]),
                "len_rejected": len(r["rejected"]),
                "chosen": {k2: v for k2, v in sc.items()},
                "rejected": {k2: v for k2, v in sr.items()},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
