"""Figs. 2 and 3: NS-U head sum and decomposition; ladder steps per run (hyperspectral runs).

Inputs (results/q1):
  ladders_runs.csv  per-run ladder contrasts: scene, model, step, split_seed, delta_boot_mean
  nsu_explore.csv   exploratory NS-U steps: scene, step, split, delta_boot_mean
  nsu_sum_by_run.csv  per NS-U run: scene, share_TplusF_gt_1, mean_TplusF (predicted class, test
  pixels)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from style import BLUE, GREY, ORANGE, PURPLE, plt

plt.rcParams.update({"xtick.major.width": 0.8, "ytick.major.width": 0.8})

LADDER_ROWS = [  # label, model, step, colour, source table
    ("Softmax: $p_{\\max}$ | $H$", "softmax", "S2-S1", GREY, "lad"),
    ("Softmax: $p_{(2)}$ | $H, p_{\\max}$", "softmax", "S3-S2", GREY, "lad"),
    ("MC dropout: MI | PE", "softmax", "M2-M1", PURPLE, "lad"),
    ("MC dropout: $p_{\\max}$ | PE, MI", "softmax", "M3-M2", PURPLE, "lad"),
    ("Evidential: $u$ | $H$", "evidential", "R2-R1", BLUE, "lad"),
    ("Evidential: $b_{\\max}, b_{(2)}$ | $H, u$", "evidential", "R3plus-R2", BLUE, "lad"),
    ("Evidential: $(b_{\\max}, b_{(2)}, u)$ vs $(H, u)$", "evidential", "R3-R2", BLUE, "lad"),
    ("NS-U: $(T, F)$ vs $T - F$", "nsu", "N2-N1", ORANGE, "lad"),
    (
        "NS-U: $T, F$ | $H(\\tilde p), \\tilde p_{\\max}$ (pre-reg.)",
        "nsu",
        "N2plus-N2base",
        ORANGE,
        "lad",
    ),
    (
        "NS-U: $T$ vs $H(\\tilde p), \\tilde p_{\\max}$ (expl.)",
        "nsu",
        "T_only-N2base",
        ORANGE,
        "exp",
    ),
    ("NS-U: $F$ | $T$ (expl.)", "nsu", "N2-T_only", ORANGE, "exp"),
]
DECOMPOSITION = [
    ("T_only-N2base", "$T$ vs $H(\\tilde p), \\tilde p_{\\max}$"),
    ("T_only-N1", "$T$ vs $T - F$"),
    ("N2-T_only", "$F$ given $T$"),
    ("N2plus-N2", "$H(\\tilde p), \\tilde p_{\\max}$ given $T, F$"),
]
SCENES = [("pavia_u", "Pavia University"), ("salinas", "Salinas")]


def fig3(lad, exp, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.1), sharey=True)
    for ax, (sc, name), tag in zip(axes, SCENES, "ab"):
        ax.axvspan(-0.03, 0.03, color=GREY, alpha=0.12, lw=0)
        ax.axvline(0, color="#4a4a46", lw=0.6)
        for i, (_, model, step, colour, src) in enumerate(LADDER_ROWS):
            y = len(LADDER_ROWS) - 1 - i
            if src == "lad":
                v = lad[
                    (lad.scene == sc) & (lad.model == model) & (lad.step == step)
                ].delta_boot_mean.values
            else:
                v = exp[(exp.scene == sc) & (exp.step == step)].delta_boot_mean.values
            jitter = (np.random.default_rng(i).random(len(v)) - 0.5) * 0.35
            ax.scatter(np.clip(v, -0.1, 0.3), y + jitter, s=9, color=colour, alpha=0.55, lw=0)
            ax.plot([v.mean()], [y], "|", color="black", ms=11, mew=1.6)
        ax.set_title(f"({tag}) {name}", loc="left")
        ax.set_xlim(-0.1, 0.3)
        ax.set_xlabel("$\\Delta$AUC (per run; bar = mean)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(range(len(LADDER_ROWS)))
    axes[0].set_yticklabels([r[0] for r in LADDER_ROWS][::-1])
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig2(nsu, exp, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    for (sc, name), colour in zip(SCENES, (BLUE, ORANGE)):
        d = nsu[nsu.scene == sc]
        ax.scatter(d.share_TplusF_gt_1, d.mean_TplusF, color=colour, s=18, label=name, zorder=3)
    ax.axhline(1.0, color="#4a4a46", lw=0.6, ls="--")
    ax.set_ylim(0.985, 1.015)
    ax.set_xlim(0.2, 0.9)
    ax.set_xlabel("share of test pixels with $T + F > 1$")
    ax.set_ylabel("mean of $T + F$ over test pixels")
    ax.set_title("(a) Learned sum of the two heads", loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax = axes[1]
    ax.axvspan(-0.03, 0.03, color=GREY, alpha=0.12, lw=0)
    ax.axvline(0, color="#4a4a46", lw=0.6)
    for i, (step, _) in enumerate(DECOMPOSITION):
        y = len(DECOMPOSITION) - 1 - i
        for (sc, _), colour, off in zip(SCENES, (BLUE, ORANGE), (0.12, -0.12)):
            v = exp[(exp.scene == sc) & (exp.step == step)].delta_boot_mean.values
            ax.scatter(v, np.full(len(v), y + off), color=colour, s=12, alpha=0.7, lw=0)
            ax.plot([v.mean()], [y + off], "|", color="black", ms=8, mew=1.4)
    ax.set_yticks(range(len(DECOMPOSITION)))
    ax.set_yticklabels([s[1] for s in DECOMPOSITION][::-1])
    ax.set_xlabel("$\\Delta$AUC (per run; bar = mean)")
    ax.set_title("(b) Where the information sits", loc="left")
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q1", default="results/q1")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    q1, out = Path(a.q1), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lad = pd.read_csv(q1 / "ladders_runs.csv")
    lad = lad[lad.split_seed.astype(str) != "pooled"]
    exp = pd.read_csv(q1 / "nsu_explore.csv")
    exp = exp[exp.split.astype(str) != "pooled"]
    nsu = pd.read_csv(q1 / "nsu_sum_by_run.csv")
    fig3(lad, exp, out / "fig3_ladders.png")
    fig2(nsu, exp, out / "fig2_nsu.png")


if __name__ == "__main__":
    main()
