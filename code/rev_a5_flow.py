"""Revision analyses (post hoc), comment 20 (A5): preference data flow (drawn -> scored -> length
filter),
exclusions, UltraFeedback overlap and SHP selection (Table S16). Needs the rebuilt pair texts
(scripts/download_data.py --rebuild-pairs) and the numbers-only score files."""

import json
import os

import common as C
from transformers import AutoTokenizer

TEXT_DIR = os.path.join(C.DATA_DIR, "preference")
SCORE_DIR = os.path.join(C.ROOT, "preference_scores")
REL = 0.25


def jl(name):
    folder = TEXT_DIR if name.startswith("train_noisy") else SCORE_DIR
    with open(os.path.join(folder, name + ".jsonl"), encoding="utf-8") as f:
        return [json.loads(x) for x in f]


tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")


def rel(r):
    return abs(r["len_chosen"] - r["len_rejected"]) / max(r["len_chosen"], r["len_rejected"], 1)


def ntok(t):
    """Response length in tokens, as in pref_score.score (no special tokens, first 512)."""
    return len(tok(t[:20000], add_special_tokens=False, truncation=True, max_length=512).input_ids)


SRC = [
    ("orca_dpo_pairs", "train_noisy_orca_big", "ns_logprob_orca_big"),
    (
        "UltraFeedback gap>=4",
        "train_noisy_ultrafeedback_gap4big",
        "ns_logprob_ultrafeedback_gap4big",
    ),
    (
        "UltraFeedback gap>=2",
        "train_noisy_ultrafeedback_gap2big",
        "ns_logprob_ultrafeedback_gap2big",
    ),
    ("Anthropic/hh-rlhf", "train_noisy_hh_rlhf_big", "ns_logprob_hh_rlhf_big"),
    ("SHP", "train_noisy_shp", "ns_logprob_shp"),
]
flow, excl, keys = [], [], {}
tot = dict(drawn=0, scored=0, excl=0, bal=0)
for name, src, sco in SRC:
    S, Cs = jl(src), jl(sco)
    done = {r["id"]: r for r in Cs}
    miss = [r for r in S if r["id"] not in done]
    reasons = {}
    for r in miss:
        a, b = ntok(r["chosen"]), ntok(r["rejected"])
        why = (
            ("chosen<2 tokens" if a < 2 else "")
            + ("; " if a < 2 and b < 2 else "")
            + ("rejected<2 tokens" if b < 2 else "")
        )
        why = why or "OTHER (not explained by the <2-token rule)"
        reasons[why] = reasons.get(why, 0) + 1
    for w, n in sorted(reasons.items()):
        excl.append([name, w, n])
    nbal = sum(rel(r) <= REL for r in Cs)
    flow.append(
        [
            name,
            src,
            len(S),
            len(miss),
            len(Cs),
            nbal,
            len(Cs) - nbal,
            round(nbal / len(Cs), 4),
            sum(r["corrupt"] for r in done.values()),
        ]
    )
    for k in tot:
        tot[k] += {"drawn": len(S), "scored": len(Cs), "excl": len(miss), "bal": nbal}[k]
    if name.startswith("UltraFeedback"):
        # pair key independent of presentation order (a swap exchanges chosen and rejected)
        keys[name] = {(r["prompt"], frozenset((r["chosen"], r["rejected"]))): r["id"] for r in S}
        keys[name + " scored"] = {
            (r["prompt"], frozenset((r["chosen"], r["rejected"]))) for r in S if r["id"] in done
        }
    if name == "SHP":
        meta = {r["id"]: r for r in S}
        rows = [(meta[i], done[i]) for i in done]
        shp = []
        for lab, f in (
            ("near tie: ratio < 1.5", lambda x: x < 1.5),
            ("decisive: ratio >= 3", lambda x: x >= 3),
            ("between: 1.5 <= ratio < 3 (excluded from E4b)", lambda x: 1.5 <= x < 3),
        ):
            sel = [c for m, c in rows if f(m["score_ratio"])]
            shp.append([lab, len(sel), sum(rel(c) <= REL for c in sel)])
        C.write_csv(
            C.REV / "c20_shp_selection.csv", ["subset", "n_scored", "n_length_balanced"], shp
        )

g4, g2 = keys["UltraFeedback gap>=4"], keys["UltraFeedback gap>=2"]
ov = set(g4) & set(g2)
ovs = keys["UltraFeedback gap>=4 scored"] & keys["UltraFeedback gap>=2 scored"]
flow.append(
    [
        "TOTAL (rows above summed)",
        "",
        tot["drawn"],
        tot["excl"],
        tot["scored"],
        tot["bal"],
        tot["scored"] - tot["bal"],
        "",
        "",
    ]
)
C.write_csv(
    C.REV / "c20_flow_by_source.csv",
    [
        "source",
        "file",
        "drawn",
        "excluded",
        "scored",
        "length_balanced_le0.25",
        "removed_by_length_filter",
        "retained_share",
        "corrupt_in_scored",
    ],
    flow,
)
C.write_csv(C.REV / "c20_exclusions.csv", ["source", "reason", "n_pairs"], excl)
uf = [
    ["gap>=4 draw: pairs", len(g4)],
    ["gap>=2 draw: pairs", len(g2)],
    ["pairs present in BOTH draws (drawn)", len(ov)],
    ["pairs present in BOTH draws (both scored)", len(ovs)],
    ["unique UltraFeedback pairs drawn (union)", len(set(g4) | set(g2))],
    [
        "unique UltraFeedback pairs scored (union)",
        len(keys["UltraFeedback gap>=4 scored"] | keys["UltraFeedback gap>=2 scored"]),
    ],
]
C.write_csv(C.REV / "c20_ultrafeedback_overlap.csv", ["item", "n"], uf)
print(
    f"drawn {tot['drawn']}, scored {tot['scored']}, excluded {tot['excl']}; "
    f"UltraFeedback overlap (scored) {len(ovs)}"
)
