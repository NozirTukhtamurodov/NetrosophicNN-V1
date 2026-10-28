"""Table 3 (C2_summary: OA, AA, kappa mean and SD), Section 6.4 best rung per model
(C4_cross_model),
SD of ladder contrasts (C3_delta_mean_sd), classes without training pixels (C5) and Section 6.5
transductive vs train-only preprocessing (C6_old_vs_new_pca). Input and output: results/q1."""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
from common import RESULTS_DIR


def ms(v):
    """Mean and SD (ddof = 1) as 4-decimal strings."""
    v = np.array(v, float)
    return (f"{v.mean():.4f}", f"{v.std(ddof=1):.4f}" if len(v) > 1 else "") if len(v) else ("", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(RESULTS_DIR / "q1"))
    T = ap.parse_args().dir

    def rd(fn):
        with open(os.path.join(T, fn), encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def write(fn, hdr, rows):
        with open(os.path.join(T, fn), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(hdr)
            w.writerows(rows)

    runs = [r for r in rd("C1_runs.csv") if r["note"] == ""]
    g = defaultdict(list)
    for r in runs:
        g[(r["scene"], r["model"])].append(r)
    c2 = []
    for (sc, m), rr in sorted(g.items()):
        row = [sc, m, len(rr)]
        for k in ("OA", "AA", "kappa", "errors"):
            row += list(ms([float(x[k]) for x in rr]))
        if m == "softmax":
            row += list(ms([float(x["OA_mc_dropout_mean"]) for x in rr]))
        else:
            row += ["", ""]
        c2.append(row)
    write(
        "C2_summary.csv",
        [
            "scene",
            "model",
            "n_runs",
            "OA_mean",
            "OA_sd",
            "AA_mean",
            "AA_sd",
            "kappa_mean",
            "kappa_sd",
            "errors_mean",
            "errors_sd",
            "OA_mcmean_mean",
            "OA_mcmean_sd",
        ],
        c2,
    )

    D = [
        r for r in rd("C3_deltas.csv") if r["split_seed"] != "pooled" and r["model"] != "ensemble5"
    ]
    agg = []
    for level in ("per_split", "all_runs"):
        gg = defaultdict(list)
        for r in D:
            key = (
                r["scene"],
                r["model"],
                r["ladder"],
                r["step"],
                r["split_seed"] if level == "per_split" else "all",
            )
            gg[key].append(r)
        for key, rr in sorted(gg.items()):
            row = [*key, len(rr)]
            for k in ("auc_upper", "auc_lower", "delta_point", "delta_boot_mean", "ci_lo"):
                row += list(ms([float(x[k]) for x in rr]))
            row += [sum(x["pass_bootmean_rule"] == "yes" for x in rr)]
            agg.append(row)
    write(
        "C3_delta_mean_sd.csv",
        [
            "scene",
            "model",
            "ladder",
            "step",
            "split_seed_or_all",
            "n_runs",
            "auc_upper_mean",
            "auc_upper_sd",
            "auc_lower_mean",
            "auc_lower_sd",
            "delta_point_mean",
            "delta_point_sd",
            "delta_bootmean_mean",
            "delta_bootmean_sd",
            "ci_lo_mean",
            "ci_lo_sd",
            "n_runs_passing",
        ],
        agg,
    )

    A = [r for r in rd("C3_aucs.csv") if r["split_seed"] != "pooled"]
    LAB = {
        ("softmax", "S"): "softmax CNN",
        ("softmax", "M"): "MC dropout (softmax CNN)",
        ("evidential", "R"): "evidential CNN",
        ("nsu", "N"): "neutrosophic NS-U CNN",
        ("ensemble5", "E"): "deep ensemble (5, split 0)",
        ("hybrid_softmax", "S"): "HybridSN softmax",
        ("hybrid_evidential", "R"): "HybridSN evidential",
    }
    c4 = []
    for sc in ("pavia_u", "salinas"):
        for (m, lad), lab in LAB.items():
            rr = [r for r in A if r["scene"] == sc and r["model"] == m and r["ladder"] == lad]
            if not rr:
                continue
            for kind in ("single(OOF)", "rung"):
                per = defaultdict(list)
                for r in rr:
                    if r["kind"] == kind:
                        per[r["feature_or_rung"]].append(float(r["auc"]))
                best = max(per, key=lambda k: np.mean(per[k]))  # noqa: B023
                v = per[best]
                allf = "; ".join(f"{k}={np.mean(x):.4f}" for k, x in per.items())
                c4.append([sc, lab, kind, best, len(v), *ms(v), allf])
    write(
        "C4_cross_model.csv",
        ["scene", "model", "kind", "best", "n_runs", "auc_mean", "auc_sd", "all_mean_aucs"],
        c4,
    )

    PC = rd("C5_perclass.csv")
    zero = defaultdict(set)
    for r in PC:
        if r["n_train_class"] == "0" and r["model"] == "evidential":
            zero[(r["scene"], r["split_seed"])].add(int(r["class"]))
    write(
        "C5_classes_no_training.csv",
        ["scene", "split_seed", "classes_with_0_train_px"],
        [[k[0], k[1], ";".join(map(str, sorted(v)))] for k, v in sorted(zero.items())],
    )

    # Section 6.5: transductive preprocessing, evidential network, training seed = split seed
    with open(os.path.join(T, "transductive_b6.json"), encoding="utf-8") as f:
        b6 = json.load(f)
    with open(os.path.join(T, "transductive_a.json"), encoding="utf-8") as f:
        old_a = json.load(f)
    dd = {
        (r["scene"], r["split_seed"], r["train_seed"], r["step"]): r
        for r in rd("C3_deltas.csv")
        if r["model"] == "evidential"
    }
    rr = {
        (r["scene"], r["split_seed"], r["train_seed"]): r
        for r in runs
        if r["model"] == "evidential"
    }
    c6 = []
    KO = {
        "R2-R1": "R2 - R1 (свидетельство)",
        "R3-R2": "невложенный R3 - R2",
        "R3plus-R2": "ВЛОЖЕННЫЙ R3+ - R2",
    }
    for sc in ("pavia_u", "salinas"):
        for s in ("0", "1", "2"):
            new = rr.get((sc, s, s))
            if new is None:
                continue
            row = [
                sc,
                s,
                f"{old_a[f'{sc}_s{s}']['oa']:.4f}",
                new["OA"],
                f"{old_a[f'{sc}_s{s}']['aa']:.4f}",
                new["AA"],
            ]
            for st, ko in KO.items():
                o = b6[f"{sc}_s{s}"]["delta"][ko]
                n = dd[(sc, s, s, st)]
                row += [
                    f"{o['mean']:+.4f}",
                    f"[{o['lo']:+.4f}; {o['hi']:+.4f}]",
                    n["delta_boot_mean"],
                    f"[{n['ci_lo']}; {n['ci_hi']}]",
                ]
            c6.append(row)
    write(
        "C6_old_vs_new_pca.csv",
        [
            "scene",
            "split_seed",
            "OA_old_transductive",
            "OA_new_trainonly",
            "AA_old",
            "AA_new",
            "dR2R1_old",
            "ci_old",
            "dR2R1_new",
            "ci_new",
            "dR3R2_old",
            "ci_old",
            "dR3R2_new",
            "ci_new",
            "dR3plusR2_old",
            "ci_old",
            "dR3plusR2_new",
            "ci_new",
        ],
        c6,
    )


if __name__ == "__main__":
    main()
