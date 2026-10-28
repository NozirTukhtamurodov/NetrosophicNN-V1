"""Revision analyses (post hoc): all revision training runs (70 replicas, 72 C1(b) runs, 36 C2
runs), one process
each, resumable (finished .npz files are skipped). Log:
outputs/hsi_outputs_revision/run_train.log."""

import os
import subprocess
import sys
import time

from common import HSI_REV

HERE = os.path.dirname(os.path.abspath(__file__))
SC = ["pavia_u", "salinas"]
jobs = []
for sc in SC:
    for m in ["softmax", "evidential", "nsu"]:
        for s in range(3):
            for t in range(3):
                jobs.append((sc, m, s, t, "replica", 0.0))
    for t in (3, 4):
        jobs.append((sc, "softmax", 0, t, "replica", 0.0))
    for m in ["hybrid_softmax", "hybrid_evidential"]:
        for s in range(3):
            jobs.append((sc, m, s, 0, "replica", 0.0))
for lam in (0.0, 1e-4, 1e-3, 1e-2):
    for sc in SC:
        for s in range(3):
            for t in range(3):
                jobs.append((sc, "nsu", s, t, "nsu_pen", lam))
for v in ("evid_log", "evid_mse_anneal20"):
    for sc in SC:
        for s in range(3):
            for t in range(3):
                jobs.append((sc, "evidential", s, t, v, 0.0))


def out_of(j):
    sc, m, s, t, v, lam = j
    vd = v + (f"_lam{lam:g}" if v == "nsu_pen" else "")
    return os.path.join(HSI_REV, vd, f"{sc}_{m}_s{s}_t{t}.npz"), vd


os.makedirs(HSI_REV, exist_ok=True)
with open(os.path.join(HSI_REV, "run_train.log"), "a", encoding="utf-8") as log:
    for j in jobs:
        out, vd = out_of(j)
        if os.path.exists(out):
            continue
        sc, m, s, t, v, lam = j
        tag = f"{vd}_{sc}_{m}_s{s}_t{t}"
        for attempt in (1, 2):
            with open(os.path.join(HSI_REV, f"train_{tag}.log"), "w", encoding="utf-8") as f:
                rc = subprocess.call(
                    [
                        sys.executable,
                        "-u",
                        os.path.join(HERE, "rev_train.py"),
                        "--scene",
                        sc,
                        "--model",
                        m,
                        "--split",
                        str(s),
                        "--tseed",
                        str(t),
                        "--variant",
                        v,
                        "--lam",
                        str(lam),
                    ],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )
            log.write(time.strftime("[%Y-%m-%d %H:%M:%S] ") + f"{tag} rc={rc} attempt {attempt}\n")
            log.flush()
            if rc == 0:
                break
    done = sum(os.path.exists(out_of(j)[0]) for j in jobs)
    log.write(time.strftime("[%Y-%m-%d %H:%M:%S] ") + f"finished: {done}/{len(jobs)}\n")
