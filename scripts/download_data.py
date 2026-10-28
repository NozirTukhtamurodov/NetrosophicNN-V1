"""Download and verify the third-party data used by the paper (not redistributed here).

  hyperspectral scenes  Pavia University, Salinas (.mat files; Hugging Face mirrors
                        danaroth/pavia and danaroth/salinas)
  CIFAR-10H             vote counts from github.com/jcpeterson/cifar-10h (pinned commit);
                        CIFAR-10 via torchvision
  preference pairs      rebuilt from preference_scores/preference_pair_ids.csv and the pinned
                        Hugging Face revisions of Intel/orca_dpo_pairs,
                        HuggingFaceH4/ultrafeedback_binarized, Anthropic/hh-rlhf, stanfordnlp/SHP

Every file is checked against its SHA-256. Usage:
  python scripts/download_data.py                  download and rebuild into $NSDOF_DATA_DIR
  python scripts/download_data.py --checksum-only  only verify files already present
  python scripts/download_data.py --skip-preference
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NSDOF_DATA_DIR", ROOT / "data"))
IDS = ROOT / "preference_scores" / "preference_pair_ids.csv"

HSI = {  # file: (Hugging Face repo, revision, sha256)
    "PaviaU.mat": (
        "danaroth/pavia",
        "8582000f3769db8e19720efabd0692ad2edfde41",
        "28447fa87f7a5797845e9a189c0da85e23b1d06a4ba7361e5ff44efbf834d2fb",
    ),
    "PaviaU_gt.mat": (
        "danaroth/pavia",
        "8582000f3769db8e19720efabd0692ad2edfde41",
        "23f6a426928f9b32984adffe659e29f554f9fb6c93b5a107528d308d5087a829",
    ),
    "Salinas_corrected.mat": (
        "danaroth/salinas",
        "fc368ffdddd47bed4a3266218f5d315fafd49cb3",
        "5ec1c0d22f56d18ecd336f8e35735863c0f160682e04e0c18ef3f89a3334d87d",
    ),
    "Salinas_gt.mat": (
        "danaroth/salinas",
        "fc368ffdddd47bed4a3266218f5d315fafd49cb3",
        "ecfab4d31ef5553f097943235d8ea502038eb4a2067b2ad10b33e37c949955e2",
    ),
}
CIFAR10H = (
    "https://raw.githubusercontent.com/jcpeterson/cifar-10h/"
    "389f22d9d300ec71a4c44735d46f550c94b957ab/data/cifar10h-counts.npy",
    "9b36b4550349af56c79083356312eb4af856cc5fd3caf1f669d40b9d0bbb69ff",
)
CIFAR10_TAR = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
CRLF, LF = "\r\n", "\n"  # line endings of the original files; the SHA-256 pins depend on them
PAIRS = {  # sample in preference_pair_ids.csv: (file, sha256, line ending)
    "orca_big": (
        "train_noisy_orca_big.jsonl",
        "70774eea1824a46fe82bee0111a7b4470a5930f7b63345470004ebcd1ed7499a",
        CRLF,
    ),
    "uf_gap4big": (
        "train_noisy_ultrafeedback_gap4big.jsonl",
        "54fe0b699253ab9d087c4fe35064e31d1813da039f469781a4cffd45abed6f8c",
        CRLF,
    ),
    "uf_gap2big": (
        "train_noisy_ultrafeedback_gap2big.jsonl",
        "a452e34ca04c181e63b48cccdd427a55073c72a41881f13f2ae107fbdbc28be1",
        CRLF,
    ),
    "hh_big": (
        "train_noisy_hh_rlhf_big.jsonl",
        "6d864a70338d095f2509b713aa1ff0e4f5535802f6b27a9fadd10e3de4fcb941",
        CRLF,
    ),
    "shp": (
        "train_noisy_shp.jsonl",
        "036025357671a82a0333849e3d21ce120aa39b25c288acb77cc1f41b0c16b560",
        CRLF,
    ),
    "uf_gap4_2000": (
        "train_noisy_ultrafeedback_gap4.jsonl",
        "5cbac09af56163835b17725b5be9f93ccc82a61cd9691d77c94003e8f623dc17",
        CRLF,
    ),
    "orca_487": (
        "train_noisy.jsonl",
        "caccbd66d8d992dc0201530badd01af03db3b098747b5e09091d7de8eba7674c",
        LF,
    ),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(path, expected):
    if not path.exists():
        print(f"MISSING  {path}")
        return False
    ok = sha256(path) == expected
    print(f"{'OK      ' if ok else 'MISMATCH'} {path}")
    return ok


def last_turn(text, marker="\n\nAssistant:"):
    i = text.rfind(marker)
    return (None, None) if i < 0 else (text[:i], text[i + len(marker) :].strip())


def pair_record(sample, r, row):
    """One record of the pair file, in the field order written by pref_sample_*.py."""
    swap = row["presented_order"] == "swapped"
    if sample == "shp":
        a, b = r["human_ref_A"], r["human_ref_B"]
        better, worse = (a, b) if r["labels"] == 1 else (b, a)
        sb, sw = (r["score_A"], r["score_B"]) if r["labels"] == 1 else (r["score_B"], r["score_A"])
        return {
            "id": row["pair_id"],
            "prompt": r["history"],
            "score_better": int(sb),
            "score_worse": int(sw),
            "score_ratio": float(r["score_ratio"]),
            "domain": r["domain"].replace("_train", ""),
            "I_consistency": 0.0,
            "corrupt": swap,
            "chosen": worse if swap else better,
            "rejected": better if swap else worse,
        }
    if row["hf_dataset"] == "HuggingFaceH4/ultrafeedback_binarized":
        p = r["prompt"]
        c = [m for m in r["chosen"] if m["role"] == "assistant"][-1]["content"]
        j = [m for m in r["rejected"] if m["role"] == "assistant"][-1]["content"]
        gap = r["score_chosen"] - r["score_rejected"]
    elif row["hf_dataset"] == "Anthropic/hh-rlhf":
        pc, c = last_turn(r["chosen"])
        _, j = last_turn(r["rejected"])
        p, gap = pc.strip(), None
    else:
        p = f"{r['system']}\n\n{r['question']}" if r.get("system") else r["question"]
        c, j, gap = r["chosen"], r["rejected"], None
    if swap:
        c, j = j, c
    if sample == "orca_487":
        return {
            "id": row["pair_id"],
            "prompt": p,
            "chosen": c,
            "rejected": j,
            "I_consistency": float(row["i_consistency"]),
            "corrupt": swap,
        }
    return {
        "id": row["pair_id"],
        "prompt": p,
        "chosen": c,
        "rejected": j,
        "score_gap": gap,
        "I_consistency": 0.0,
        "corrupt": swap,
    }


def rebuild_pairs():
    from datasets import load_dataset

    with open(IDS, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cache = {}
    out_dir = DATA / "preference"
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for sample, (fn, expected, eol) in PAIRS.items():
        sel = [r for r in rows if r["sample"] == sample]
        key = (sel[0]["hf_dataset"], sel[0]["hf_revision"], sel[0]["hf_split"])
        if key not in cache:
            cache[key] = load_dataset(key[0], split=key[2], revision=key[1])
        d = cache[key]
        with open(out_dir / fn, "w", encoding="utf-8", newline="") as f:
            for row in sel:
                rec = pair_record(sample, d[int(row["row_index"])], row)
                f.write(json.dumps(rec, ensure_ascii=False) + eol)
        ok &= check(out_dir / fn, expected)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checksum-only", action="store_true")
    ap.add_argument("--skip-preference", action="store_true")
    a = ap.parse_args()
    ok = True
    hsi_dir = DATA / "hsi"
    for fn, (repo, rev, digest) in HSI.items():
        if not a.checksum_only and not (hsi_dir / fn).exists():
            from huggingface_hub import hf_hub_download

            hsi_dir.mkdir(parents=True, exist_ok=True)
            p = hf_hub_download(repo, fn, repo_type="dataset", revision=rev)
            shutil.copy(p, hsi_dir / fn)
        ok &= check(hsi_dir / fn, digest)
    counts = DATA / "cifar10h" / "cifar10h-counts.npy"
    if not a.checksum_only and not counts.exists():
        counts.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CIFAR10H[0], counts)
    ok &= check(counts, CIFAR10H[1])
    tar = DATA / "torchvision" / "cifar-10-python.tar.gz"
    if not a.checksum_only and not tar.exists():
        import torchvision

        torchvision.datasets.CIFAR10(DATA / "torchvision", train=False, download=True)
    ok &= check(tar, CIFAR10_TAR)
    if not a.skip_preference:
        if a.checksum_only:
            for fn, digest, _ in PAIRS.values():
                ok &= check(DATA / "preference" / fn, digest)
        else:
            ok &= rebuild_pairs()
    print("all files verified" if ok else "some files are missing or do not match")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
