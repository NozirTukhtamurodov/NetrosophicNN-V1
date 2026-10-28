"""Revision analyses (post hoc), comments 9, 10-11, 12, 13, 14, 15: tables B2-B8 from engine_*.csv
(thresholds, p-values and BH, spatial blocking, C sensitivity and rank, single-coordinate steps,
training-absent
classes, AUPRC/AURC and risk-coverage curves), and the check against the q1 table C3_deltas.csv."""

import csv
import os

import common as C
import matplotlib
import numpy as np
from rev_engine import ensemble_data, extend, oof

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def rd(f):
    with open(os.path.join(C.REV, f), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


S = rd("engine_steps.csv")
RG = rd("engine_rungs.csv")
FA = rd("engine_foldauc.csv")
orig = [r for r in S if r["variant"] == "orig" and r["step_kind"] == "orig"]


def fam(r):
    if r["model"].startswith("hybrid"):
        return "HybridSN"
    return "ensemble" if r["model"] == "ensemble5" else "CNN"


def key(r):
    return (r["scene"], r["model"], r["split_seed"], r["train_seed"], r["step"])


q = {
    key(r): r
    for r in csv.DictReader(open(C.RESULTS_DIR / "q1" / "C3_deltas.csv", encoding="utf-8"))
    if r["split_seed"] != "pooled"
}
dmax = max(
    max(
        abs(float(r[a]) - float(q[key(r)][b]))
        for a, b in (("delta_boot_mean", "delta_boot_mean"), ("ci_lo", "ci_lo"), ("ci_hi", "ci_hi"))
    )
    for r in orig
)
nmatch = sum(key(r) in q for r in orig)
open(os.path.join(C.REV, "check_engine_vs_table4.txt"), "w", encoding="utf-8").write(
    f"steps matched {nmatch}/{len(orig)}; "
    f"max abs difference in delta mean / CI bounds = {dmax:.2e} "
    "(q1 values are rounded to 4 decimals)\n"
)

# B2: decision thresholds
TH = [0.01, 0.02, 0.03, 0.05]
b2 = []
for r in orig:
    m, lo = float(r["delta_boot_mean"]), float(r["ci_lo"])
    b2.append(
        [
            fam(r),
            r["scene"],
            r["model"],
            r["split_seed"],
            r["train_seed"],
            r["step"],
            r["delta_boot_mean"],
            r["ci_lo"],
            r["ci_hi"],
            *[C.rule(m, lo, t) for t in TH],
        ]
    )
for r in csv.DictReader(open(C.RESULTS_DIR / "q1" / "C3_deltas.csv", encoding="utf-8")):
    if r["split_seed"] == "pooled":
        m, lo = float(r["delta_boot_mean"]), float(r["ci_lo"])
        b2.append(
            [
                ("HybridSN" if r["model"].startswith("hybrid") else "CNN") + " (pooled, q1 values)",
                r["scene"],
                r["model"],
                "pooled",
                r["train_seed"],
                r["step"],
                r["delta_boot_mean"],
                r["ci_lo"],
                r["ci_hi"],
                *[C.rule(m, lo, t) for t in TH],
            ]
        )
C.write_csv(
    C.REV / "c15_threshold_sensitivity.csv",
    [
        "family",
        "scene",
        "model",
        "split_seed",
        "train_seed",
        "step",
        "delta_mean",
        "ci_lo",
        "ci_hi",
        *[f"pass_thr_{t}" for t in TH],
    ],
    b2,
)
cnt = []
for f in ["CNN", "ensemble", "HybridSN", "CNN (pooled, q1 values)", "HybridSN (pooled, q1 values)"]:
    rr = [r for r in b2 if r[0] == f]
    cnt.append([f, len(rr), *[sum(r[9 + i] == "yes" for r in rr) for i in range(4)]])
C.write_csv(
    C.REV / "c15_threshold_counts.csv",
    ["family", "n_contrasts", *[f"n_pass_thr_{t}" for t in TH]],
    cnt,
)

# B3: bootstrap p-values and Benjamini-Hochberg
prim = [r for r in orig if fam(r) == "CNN"]


def bh(p):
    p = np.asarray(p)
    n = len(p)
    o = np.argsort(p)
    adj = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(adj, 1)
    return out


b3 = []
for famname, rows in (("primary: 162 CNN per-run steps", prim), ("all 196 per-run steps", orig)):
    p = [min(1.0, 2 * min(float(r["share_le0"]), float(r["share_ge0"]))) for r in rows]
    adj = bh(p)
    for r, pi, ai in zip(rows, p, adj, strict=True):
        b3.append(
            [
                famname,
                fam(r),
                r["scene"],
                r["model"],
                r["split_seed"],
                r["train_seed"],
                r["step"],
                r["delta_boot_mean"],
                r["ci_lo"],
                r["ci_hi"],
                r["B_eff"],
                f"{pi:.5f}",
                f"{ai:.5f}",
                "yes" if ai <= 0.05 else "no",
                "yes" if (ai <= 0.05 and float(r["delta_boot_mean"]) > 0.03) else "no",
                C.rule(float(r["delta_boot_mean"]), float(r["ci_lo"])),
            ]
        )
C.write_csv(
    C.REV / "c14_pvalues_bh.csv",
    [
        "family_for_BH",
        "model_family",
        "scene",
        "model",
        "split_seed",
        "train_seed",
        "step",
        "delta_mean",
        "ci_lo",
        "ci_hi",
        "B_eff",
        "p_boot_two_sided",
        "p_BH",
        "BH_reject_q0.05",
        "BH_reject_and_delta_gt_0.03",
        "original_rule_pass",
    ],
    b3,
)
bs = []
for famname in ("primary: 162 CNN per-run steps", "all 196 per-run steps"):
    rr = [r for r in b3 if r[0] == famname]
    bs.append(
        [
            famname,
            len(rr),
            sum(r[13] == "yes" for r in rr),
            sum(r[14] == "yes" for r in rr),
            sum(r[15] == "yes" for r in rr),
            sum(float(r[11]) == 0 for r in rr),
        ]
    )
C.write_csv(
    C.REV / "c14_bh_counts.csv",
    [
        "family",
        "n",
        "n_BH_reject",
        "n_BH_reject_and_delta_gt_0.03",
        "n_original_rule_pass",
        "n_p_equal_0_at_resolution_1_over_B",
    ],
    bs,
)
lvl = []
groups = {}
for r in orig:
    groups.setdefault((fam(r), r["scene"], r["model"], r["step"]), []).append(r)
for (f, sc, m, st), rr in sorted(groups.items()):
    v = np.array([float(r["delta_boot_mean"]) for r in rr])
    row = [
        f,
        sc,
        m,
        st,
        len(v),
        f"{v.mean():+.4f}",
        f"{v.std(ddof=1):.4f}" if len(v) > 1 else "",
        f"{v.min():+.4f}",
        f"{v.max():+.4f}",
    ]
    for s in ("0", "1", "2"):
        vs = [float(r["delta_boot_mean"]) for r in rr if r["split_seed"] == s]
        row.append(f"{np.mean(vs):+.4f}" if vs else "")
        row.append(f"{np.std(vs, ddof=1):.4f}" if len(vs) > 1 else "")
    for t in ("0", "1", "2"):
        vt = [float(r["delta_boot_mean"]) for r in rr if r["train_seed"] == t]
        row.append(f"{np.mean(vt):+.4f}" if vt else "")
    lvl.append(row)
C.write_csv(
    C.REV / "c14_delta_by_level.csv",
    [
        "family",
        "scene",
        "model",
        "step",
        "n_runs",
        "mean_over_runs",
        "sd_over_runs",
        "min",
        "max",
        "split0_mean",
        "split0_sd",
        "split1_mean",
        "split1_sd",
        "split2_mean",
        "split2_sd",
        "tseed0_mean",
        "tseed1_mean",
        "tseed2_mean",
    ],
    lvl,
)

# B4: spatial blocking
bl = {(key(r), r["variant"]): r for r in S if r["step_kind"] == "orig"}
b4 = []
for r in orig:
    k = key(r)
    row = [
        fam(r),
        *k,
        r["delta_boot_mean"],
        r["ci_lo"],
        r["ci_hi"],
        C.rule(float(r["delta_boot_mean"]), float(r["ci_lo"])),
    ]
    for v in ("origcv_blockboot16", "blockcv16", "blockcv32"):
        x = bl.get((k, v))
        row += (
            [
                x["auc_upper"],
                x["auc_lower"],
                x["delta_boot_mean"],
                x["ci_lo"],
                x["ci_hi"],
                C.rule(float(x["delta_boot_mean"]), float(x["ci_lo"])),
            ]
            if x
            else ["", "", "", "", "", "not run"]
        )
    b4.append(row)
hdr = [
    "family",
    "scene",
    "model",
    "split_seed",
    "train_seed",
    "step",
    "orig_delta",
    "orig_lo",
    "orig_hi",
    "orig_rule",
]
for v in ("origcvBlockBoot16", "blockCV16_blockBoot16", "blockCV32_blockBoot32"):
    hdr += [f"{v}_auc_upper", f"{v}_auc_lower", f"{v}_delta", f"{v}_lo", f"{v}_hi", f"{v}_rule"]
C.write_csv(C.REV / "c12_blocked_vs_original.csv", hdr, b4)
ch = [
    [
        "",
        "n",
        "orig pass",
        "origcv+blockboot16 pass",
        "blockcv16 pass",
        "blockcv32 pass",
        "verdict changed vs orig (any)",
    ]
]
for f in ("CNN", "ensemble", "HybridSN"):
    rr = [r for r in b4 if r[0] == f]
    ch.append(
        [
            f,
            len(rr),
            sum(r[9] == "yes" for r in rr),
            sum(r[15] == "yes" for r in rr),
            sum(r[21] == "yes" for r in rr),
            sum(r[27] == "yes" for r in rr),
            sum(len({r[9], r[15], r[21], r[27]} - {"not run"}) > 1 for r in rr),
        ]
    )
C.write_csv(C.REV / "c12_blocked_counts.csv", ch[0], ch[1:])
C.write_csv(
    C.REV / "c12_fold_auc_diagnostic.csv", list(FA[0].keys()), [list(r.values()) for r in FA]
)

# B5: regularisation and rank
CV = ["C=0.01", "C=0.1", "orig", "C=10", "C=100", "C=inf"]
bc = {(key(r), r["variant"]): r for r in S}
b5 = []
for r in [x for x in S if x["variant"] == "orig"]:
    k = key(r)
    row = [fam(r), r["step_kind"], *k]
    verd = []
    for v in CV:
        x = bc[(k, v)]
        vv = C.rule(float(x["delta_boot_mean"]), float(x["ci_lo"]))
        verd.append(vv)
        row += [x["delta_boot_mean"], x["ci_lo"], x["ci_hi"], vv, x["convergence_warnings"]]
    ch_ = len(set(verd)) > 1
    b5.append(row + ["CHANGES" if ch_ else ""])
hdr = ["family", "step_kind", "scene", "model", "split_seed", "train_seed", "step"]
for v in ("C0.01", "C0.1", "C1", "C10", "C100", "Cinf_unpenalised"):
    hdr += [f"{v}_delta", f"{v}_lo", f"{v}_hi", f"{v}_rule", f"{v}_conv_warn"]
C.write_csv(C.REV / "c09_C_sensitivity.csv", hdr + ["verdict_changes_with_C"], b5)
rk = [
    [
        r["scene"],
        r["model"],
        r["split_seed"],
        r["train_seed"],
        r["ladder"],
        r["rung"],
        r["n_cols"],
        r["rank"],
        r["cond_number"],
    ]
    for r in RG
    if r["variant"] == "orig"
]
C.write_csv(
    C.REV / "c09_design_rank.csv",
    [
        "scene",
        "model",
        "split_seed",
        "train_seed",
        "ladder",
        "rung",
        "n_cols",
        "numerical_rank",
        "condition_number_standardised",
    ],
    rk,
)

# ---- B6
b6 = [
    [
        fam(r),
        *key(r),
        r["auc_upper"],
        r["auc_lower"],
        r["delta_boot_mean"],
        r["ci_lo"],
        r["ci_hi"],
        C.rule(float(r["delta_boot_mean"]), float(r["ci_lo"])),
    ]
    for r in S
    if r["variant"] == "orig" and r["step_kind"] == "B6"
]
C.write_csv(
    C.REV / "c10_11_single_coordinate_steps.csv",
    [
        "family",
        "scene",
        "model",
        "split_seed",
        "train_seed",
        "step",
        "auc_upper",
        "auc_lower",
        "delta_mean",
        "ci_lo",
        "ci_hi",
        "rule_pass",
    ],
    b6,
)
au = [
    [
        r["scene"],
        r["model"],
        r["split_seed"],
        r["train_seed"],
        r["rung"],
        r["auc"],
        r["n_cols"],
        r["rank"],
    ]
    for r in RG
    if r["variant"] == "orig"
    and r["rung"] in ("BF", "BMU", "R2", "R3", "R3plus", "R2_bmax", "R2_b2")
]
C.write_csv(
    C.REV / "c10_evidential_binary_frame_aucs.csv",
    ["scene", "model", "split_seed", "train_seed", "rung", "auc", "n_cols", "numerical_rank"],
    au,
)

# ---- B7
b7 = []
for r in [x for x in S if x["variant"] == "absent_removed"]:
    o = bc[(key(r), "orig")]
    b7.append(
        [
            fam(r),
            r["step_kind"],
            *key(r),
            o["n"],
            o["errors"],
            o["delta_boot_mean"],
            o["ci_lo"],
            o["ci_hi"],
            C.rule(float(o["delta_boot_mean"]), float(o["ci_lo"])),
            r["n"],
            r["errors"],
            r["delta_boot_mean"],
            r["ci_lo"],
            r["ci_hi"],
            C.rule(float(r["delta_boot_mean"]), float(r["ci_lo"])),
        ]
    )
C.write_csv(
    C.REV / "c13_absent_classes_removed.csv",
    [
        "family",
        "step_kind",
        "scene",
        "model",
        "split_seed",
        "train_seed",
        "step",
        "all_n",
        "all_errors",
        "all_delta",
        "all_lo",
        "all_hi",
        "all_rule",
        "removed_n",
        "removed_errors",
        "removed_delta",
        "removed_lo",
        "removed_hi",
        "removed_rule",
    ],
    b7,
)

# ---- B8
b8 = [
    [
        r["scene"],
        r["model"],
        r["split_seed"],
        r["train_seed"],
        r["ladder"],
        r["rung"],
        r["auc"],
        r["auprc_errors_positive"],
        r["aurc"],
    ]
    for r in RG
    if r["variant"] == "orig"
]
C.write_csv(
    C.REV / "c15_auprc_aurc_by_run.csv",
    ["scene", "model", "split_seed", "train_seed", "ladder", "rung", "auc", "auprc", "aurc"],
    b8,
)
agg = {}
for r in b8:
    agg.setdefault((r[0], r[1], r[4], r[5]), []).append([float(x) for x in r[6:]])
b8s = []
for (sc, m, la, rg), v in sorted(agg.items()):
    v = np.array(v)
    sd = v.std(0, ddof=1) if len(v) > 1 else [np.nan] * 3
    b8s.append([sc, m, la, rg, len(v), *[f"{a:.4f}" for a in v.mean(0)], *[f"{a:.4f}" for a in sd]])
C.write_csv(
    C.REV / "c15_auprc_aurc_summary.csv",
    [
        "scene",
        "model",
        "ladder",
        "rung",
        "n_runs",
        "auc_mean",
        "auprc_mean",
        "aurc_mean",
        "auc_sd",
        "auprc_sd",
        "aurc_sd",
    ],
    b8s,
)
# risk-coverage curves: rung with the lowest mean AURC per model and ladder
# (chosen after seeing the values)
os.makedirs(C.REV_FIG, exist_ok=True)
grid = np.linspace(0.01, 1, 100)
curves = []
for sc in C.SCENE_LIST:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m in ("softmax", "evidential", "nsu", "hybrid_softmax", "hybrid_evidential", "ensemble5"):
        cand = [r for r in b8s if r[0] == sc and r[1] == m]
        if not cand:
            continue
        lad_rung = {(r[2], r[3]): float(r[7]) for r in cand}
        for la in sorted({r[2] for r in cand}):
            best = min((k for k in lad_rung if k[0] == la), key=lambda k: lad_rung[k])
            rc = []
            for sc2, _m2, _s, t, p in (
                [(sc, m, 0, "0-4", None)] if m == "ensemble5" else C.runs([m])
            ):
                if sc2 != sc or (m == "softmax" and t != "0-4" and int(t) >= 3):
                    continue
                d = ensemble_data(sc) if m == "ensemble5" else dict(np.load(p))
                lads = d["ens_ladders"] if m == "ensemble5" else C.ladders_of(m, d)[0]
                lad = next(x for x in lads if x[0] == la)
                feats = extend(m, lad)[0]
                y = lad[1]
                s_ = oof(feats[best[1]], y)[0]
                o = np.argsort(s_, kind="mergesort")
                risk = np.cumsum(y[o]) / np.arange(1, len(y) + 1)
                rc.append(np.interp(grid, np.arange(1, len(y) + 1) / len(y), risk))
            mean_rc = np.mean(rc, 0)
            lab = f"{m} {la}:{best[1]} (AURC {lad_rung[best]:.4f}, {len(rc)} runs)"
            ax.plot(grid, mean_rc, lw=1.2, label=lab)
            curves += [
                [sc, m, la, best[1], f"{g:.2f}", f"{v:.6f}"]
                for g, v in zip(grid, mean_rc, strict=True)
            ]
    ax.set_xlabel("coverage")
    ax.set_ylabel("risk (error rate among accepted)")
    ax.set_title(f"{sc}: risk-coverage, best rung per model (mean over runs; post hoc)", fontsize=9)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(C.REV_FIG, f"c15_risk_coverage_{sc}.png"), dpi=300)
    plt.close(fig)
C.write_csv(
    C.REV / "c15_risk_coverage_curves.csv",
    ["scene", "model", "ladder", "best_rung", "coverage", "mean_risk"],
    curves,
)
print(f"engine vs q1 C3_deltas: {nmatch}/{len(orig)} steps matched, max |difference| {dmax:.2e}")
