"""Revision analyses (post hoc): revised tables in the format of Table 4 (11 prespecified rows, 166
per-run
comparisons = 162 CNN + 4 ensemble) with the single-coordinate rows below them (Tables S1-S5).
Cell: "mean delta-AUC (SD over runs); runs passing". Writes t4_*.csv and t4_tables.md."""

import csv
import os

import common as C
import numpy as np

ROWS = [  # model, step, adds, type, model key, step code, block
    ("Softmax", "S2 − S1", "p_max", "nested", "softmax", "S2-S1", "T4"),
    ("Softmax", "S3 − S2", "p_(2)", "nested", "softmax", "S3-S2", "T4"),
    ("MC dropout", "M2 − M1", "MI", "nested", "softmax", "M2-M1", "T4"),
    ("MC dropout", "M3 − M2", "p_max", "nested", "softmax", "M3-M2", "T4"),
    ("Ensemble", "E2 − E1", "MI", "nested", "ensemble5", "E2-E1", "T4"),
    ("Ensemble", "E3 − E2", "p_max", "nested", "ensemble5", "E3-E2", "T4"),
    ("Evidential", "R2 − R1", "u", "nested", "evidential", "R2-R1", "T4"),
    (
        "Evidential",
        "R3+ − R2",
        "(b_max, b_(2))",
        "nested, 2-D (combined increment)",
        "evidential",
        "R3plus-R2",
        "T4",
    ),
    ("Evidential", "R3 − R2", "triple vs pair", "head-to-head", "evidential", "R3-R2", "T4"),
    ("NS-U", "N2 − N1", "F given T − F", "nested", "nsu", "N2-N1", "T4"),
    (
        "NS-U",
        "N2+ − N2base",
        "(T, F)",
        "nested, 2-D (combined increment)",
        "nsu",
        "N2plus-N2base",
        "T4",
    ),
    ("Evidential", "(H,u,b_max) − R2", "b_max", "nested, 1-D", "evidential", "R2_bmax-R2", "B6"),
    (
        "Evidential",
        "R3+ − (H,u,b_max)",
        "b_(2) given b_max",
        "nested, 1-D",
        "evidential",
        "R3plus-R2_bmax",
        "B6",
    ),
    ("Evidential", "(H,u,b_(2)) − R2", "b_(2)", "nested, 1-D", "evidential", "R2_b2-R2", "B6"),
    (
        "Evidential",
        "BF − R2",
        "binary-frame triple (b_max, 1−u−b_max, u) vs (H,u)",
        "head-to-head",
        "evidential",
        "BF-R2",
        "B6",
    ),
    (
        "Evidential",
        "BF − (b_max,u)",
        "b_other = 1−u−b_max",
        "nested, affine-dependent",
        "evidential",
        "BF-BMU",
        "B6",
    ),
    ("NS-U", "N2base+T − N2base", "T", "nested, 1-D", "nsu", "N2base_T-N2base", "B6"),
    ("NS-U", "N2+ − (N2base+T)", "F given T", "nested, 1-D", "nsu", "N2plus-N2base_T", "B6"),
    ("NS-U", "N2base+F − N2base", "F", "nested, 1-D", "nsu", "N2base_F-N2base", "B6"),
]
SC = ["pavia_u", "salinas"]
SCN = {"pavia_u": "Pavia University", "salinas": "Salinas"}


def rd(f):
    with open(os.path.join(C.REV, f), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


S = rd("engine_steps.csv") + (
    rd("engine_steps_b4_b6.csv")
    if os.path.exists(os.path.join(C.REV, "engine_steps_b4_b6.csv"))
    else []
)
IDX = {}
for r in S:
    IDX.setdefault((r["variant"], r["scene"], r["model"], r["step"]), []).append(r)


def runs_of(variant, sc, mk, step, split=None):
    v = IDX.get((variant, sc, mk, step), [])
    return [r for r in v if split is None or r["split_seed"] == str(split)]


def cell(v, thr=0.03, key=None):
    if not v:
        return "n/a"
    d = np.array([float(r["delta_boot_mean"]) for r in v])
    p = sum(
        (key(r) if key else C.rule(float(r["delta_boot_mean"]), float(r["ci_lo"]), thr) == "yes")
        for r in v
    )
    sd = f" ({d.std(ddof=1):.3f})" if len(d) > 1 else ""
    return f"{d.mean():+.3f}{sd}; {p}/{len(v)}"


def label(row):
    return list(row[:4]) + ["Table 4" if row[6] == "T4" else "revision, post hoc"]


HDR = ["Model", "Step", "Adds", "Type", "Row source"]
out_md = []


def emit(name, title, note, hdr, rows):
    C.write_csv(C.REV / f"t4_{name}.csv", hdr, rows)
    out_md.append(
        f"### {title}\n\n{note}\n\n| "
        + " | ".join(hdr)
        + " |\n|"
        + "---|" * len(hdr)
        + "\n"
        + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in rows)
        + "\n"
    )


# Table 4 as published, with the B6 rows
t0 = [label(r) + [cell(runs_of("orig", sc, r[4], r[5])) for sc in SC] for r in ROWS]
emit(
    "table4_original",
    "Table 4 (original rule (10), Δ > 0.03 and CI_lo > 0) with the B6 rows appended",
    "Cell: mean ΔAUC (SD over runs); runs passing. Recomputed by the revision engine "
    "(equals q1 within rounding).",
    HDR + [SCN[s] for s in SC],
    t0,
)

# B2
TH = [0.01, 0.02, 0.03, 0.05]
t2 = []
for r in ROWS:
    row = label(r)
    for sc in SC:
        v = runs_of("orig", sc, r[4], r[5])
        row += [cell(v, 0.03).split(";")[0]] + [
            f"{sum(C.rule(float(x['delta_boot_mean']), float(x['ci_lo']), t) == 'yes' for x in v)}"
            f"/{len(v)}"
            for t in TH
        ]
    t2.append(row)
h2 = HDR + [
    f"{SCN[sc]}: {c}"
    for sc in SC
    for c in ["mean (SD)", "pass thr 0.01", "pass thr 0.02", "pass thr 0.03", "pass thr 0.05"]
]
emit(
    "B2_thresholds",
    "B2 (comment 15): verdict at Δ thresholds 0.01 / 0.02 / 0.03 / 0.05 (always with CI_lo > 0)",
    "Runs passing per threshold; mean (SD) of ΔAUC is the same for all thresholds.",
    h2,
    t2,
)


# B3: BH family = the 166 Table 4 comparisons; B6 rows form a separate family
def bh(p):
    p = np.asarray(p)
    n = len(p)
    o = np.argsort(p)
    adj = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(adj, 1)
    return out


pv = {}
per_run = []
for blk in ("T4", "B6"):
    fam = [
        (r, sc, x)
        for r in ROWS
        if r[6] == blk
        for sc in SC
        for x in runs_of("orig", sc, r[4], r[5])
    ]
    p = [min(1.0, 2 * min(float(x["share_le0"]), float(x["share_ge0"]))) for _, _, x in fam]
    a = bh(p)
    for (r, sc, x), pi, ai in zip(fam, p, a, strict=True):
        pv[id(x)] = (pi, ai)
        per_run.append(
            [
                blk,
                len(fam),
                r[0],
                r[1],
                SCN[sc],
                x["split_seed"],
                x["train_seed"],
                x["delta_boot_mean"],
                x["ci_lo"],
                x["ci_hi"],
                x["B_eff"],
                f"{pi:.5f}",
                f"{ai:.5f}",
                "yes" if ai <= 0.05 else "no",
                C.rule(float(x["delta_boot_mean"]), float(x["ci_lo"])),
            ]
        )
C.write_csv(
    C.REV / "t4_B3_pvalues_per_run.csv",
    [
        "family",
        "family_size",
        "model",
        "step",
        "scene",
        "split_seed",
        "train_seed",
        "delta_mean",
        "ci_lo",
        "ci_hi",
        "B_eff",
        "p_boot_two_sided",
        "p_BH",
        "BH_reject_q0.05",
        "rule10_pass",
    ],
    per_run,
)
t3 = []
for r in ROWS:
    row = label(r)
    for sc in SC:
        v = runs_of("orig", sc, r[4], r[5])
        rej = sum(pv[id(x)][1] <= 0.05 for x in v)
        rej3 = sum(pv[id(x)][1] <= 0.05 and float(x["delta_boot_mean"]) > 0.03 for x in v)
        row += [cell(v), f"{rej}/{len(v)}", f"{rej3}/{len(v)}"]
    t3.append(row)
n4 = sum(1 for x in per_run if x[0] == "T4")
n6 = sum(1 for x in per_run if x[0] == "B6")
h3 = HDR + [
    f"{SCN[sc]}: {c}"
    for sc in SC
    for c in ["rule (10): mean (SD); passing", "BH q=0.05 rejections", "BH rejection and Δ > 0.03"]
]
emit(
    "B3_BH",
    "B3 (comment 14): two-sided bootstrap p-values and Benjamini–Hochberg (q = 0.05)",
    f"p = 2·min(share Δ* ≤ 0, share Δ* ≥ 0) from the same B = 3000 resamples. "
    f"BH family for the Table 4 rows = the "
    f"{n4} per-run comparisons of Table 4; the B6 rows form a separate post hoc family "
    f"({n6} comparisons).",
    h3,
    t3,
)

# B4
VB = [
    ("orig", "original (pixel CV, pixel bootstrap)"),
    ("origcv_blockboot16", "original OOF + 16×16 block bootstrap"),
    ("blockcv16", "16×16 block CV + block bootstrap"),
    ("blockcv32", "32×32 block CV + block bootstrap"),
]
t4 = []
for r in ROWS:
    row = label(r)
    for sc in SC:
        row += [cell(runs_of(v, sc, r[4], r[5])) for v, _ in VB]
    t4.append(row)
emit(
    "B4_spatial",
    "B4 (comment 12): original vs spatially blocked detector and bootstrap",
    "Cell: mean ΔAUC (SD over runs); runs passing rule (10). GroupKFold(5) by blocks; "
    "bootstrap resamples test blocks "
    "with replacement (B = 3000, seed 1337).",
    HDR + [f"{SCN[sc]}: {lab}" for sc in SC for _, lab in VB],
    t4,
)

# B5
VC = [
    ("C=0.01", "C = 0.01"),
    ("C=0.1", "C = 0.1"),
    ("orig", "C = 1 (original)"),
    ("C=10", "C = 10"),
    ("C=100", "C = 100"),
    ("C=inf", "unpenalised"),
]
t5 = []
for r in ROWS:
    row = label(r)
    for sc in SC:
        cells = [cell(runs_of(v, sc, r[4], r[5])) for v, _ in VC]
        base = {(x["split_seed"], x["train_seed"]): x for x in runs_of("orig", sc, r[4], r[5])}
        ch = 0
        for k in base:
            vs = set()
            for v, _ in VC:
                x = next(
                    y for y in runs_of(v, sc, r[4], r[5]) if (y["split_seed"], y["train_seed"]) == k
                )
                vs.add(C.rule(float(x["delta_boot_mean"]), float(x["ci_lo"])))
            ch += len(vs) > 1
        row += cells + [f"{ch}/{len(base)}"]
    t5.append(row)
emit(
    "B5_C_sensitivity",
    "B5 (comment 9): detector regularisation C",
    "Cell: mean ΔAUC (SD over runs); runs passing rule (10). "
    "Last column per scene: runs whose verdict changes across C.",
    HDR + [f"{SCN[sc]}: {lab}" for sc in SC for _, lab in VC + [("", "runs with verdict change")]],
    t5,
)

# B7
SPL = {"pavia_u": 0, "salinas": 1}
t7 = []
for r in ROWS:
    row = label(r)
    for sc in SC:
        s = SPL[sc]
        if r[4] == "ensemble5":
            allc = runs_of("orig", sc, r[4], r[5], 0) if s == 0 else []
            rem = runs_of("absent_removed", sc, r[4], r[5], 0) if s == 0 else []
        else:
            allc, rem = (
                runs_of("orig", sc, r[4], r[5], s),
                runs_of("absent_removed", sc, r[4], r[5], s),
            )
        row += [cell(allc), cell(rem)]
    t7.append(row)
emit(
    "B7_unseen_classes",
    "B7 (comment 13): test pixels of training-absent classes removed",
    "Only the splits with training-absent classes: Pavia University split 0 (class 5) "
    "and Salinas split 1 (classes 14, 16); "
    "3 runs each; ensemble only exists for split 0 (Pavia: 1 run; Salinas: n/a). "
    "The detector is refit on the subset.",
    HDR
    + [
        f"{SCN[sc]} split {SPL[sc]}: {c}"
        for sc in SC
        for c in ["all classes (same runs)", "unseen classes removed"]
    ],
    t7,
)

open(os.path.join(C.REV, "t4_tables.md"), "w", encoding="utf-8").write(
    "# Revised tables in the format of Table 4 (revision analyses, post hoc)\n\n"
    "Rows 1–11 = Table 4 (primary family, 166 per-run comparisons: 162 CNN + 4 ensemble), "
    "in the manuscript order. "
    'Rows below are the single-coordinate steps of B6, marked "revision, post hoc". '
    "HybridSN is secondary and not shown "
    "(its rows are in the per-run CSVs).\n\n" + "\n".join(out_md)
)
for r in t0:
    print(r[0], r[1], "|", r[5], "|", r[6])
