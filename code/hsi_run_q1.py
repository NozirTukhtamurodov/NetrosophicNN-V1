"""All 70 prespecified hyperspectral runs, one process each (resumable; finished .npz files are
skipped).
Wall-clock times go to outputs/logs_q1/runs_hsi.csv."""

import csv
import os
import subprocess
import sys
import time

from common import HSI_Q1, OUTPUTS_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(OUTPUTS_DIR, "logs_q1")

cfgs = []
for sc in ["pavia_u", "salinas"]:
    for m in ["softmax", "evidential", "nsu"]:
        for s in (0, 1, 2):
            for t in (0, 1, 2):
                cfgs.append((sc, m, s, t))
    cfgs += [(sc, "softmax", 0, 3), (sc, "softmax", 0, 4)]  # extra deep-ensemble members (split 0)
    for m in ["hybrid_softmax", "hybrid_evidential"]:
        for s in (0, 1, 2):
            cfgs.append((sc, m, s, 0))

os.makedirs(LOG, exist_ok=True)
book = os.path.join(LOG, "runs_hsi.csv")
new = not os.path.exists(book)
with open(book, "a", newline="", encoding="utf-8") as fb:
    w = csv.writer(fb)
    if new:
        w.writerow(["tag", "start_iso", "wall_s", "returncode"])
    for i, (sc, m, s, t) in enumerate(cfgs):
        tag = f"{sc}_{m}_s{s}_t{t}"
        if os.path.exists(os.path.join(HSI_Q1, tag + ".npz")):
            continue
        st = time.time()
        with open(os.path.join(LOG, tag + ".log"), "w", encoding="utf-8") as lf:
            rc = subprocess.call(
                [
                    sys.executable,
                    "-u",
                    os.path.join(HERE, "hsi_train_q1.py"),
                    "--scene",
                    sc,
                    "--model",
                    m,
                    "--split",
                    str(s),
                    "--tseed",
                    str(t),
                ],
                stdout=lf,
                stderr=subprocess.STDOUT,
            )
        wall = time.time() - st
        w.writerow([tag, time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st)), f"{wall:.1f}", rc])
        fb.flush()
        print(f"[{i + 1}/{len(cfgs)}] {tag} rc={rc} {wall:.0f}s", flush=True)
