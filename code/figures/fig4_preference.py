"""Fig. 4: indeterminacy and evidence-magnitude steps on the preference data (Tables 5-6).

Input: preference_steps.csv with columns panel ("indeterminacy" or "evidence"), label, delta,
ci_lo, ci_hi,
in plotting order (top to bottom).
"""

import argparse
from pathlib import Path

import pandas as pd
from style import BLUE, GREY, ORANGE, plt


def forest(ax, rows, colour, title, xlim=(-0.06, 0.07)):
    ax.axvspan(-0.03, 0.03, color=GREY, alpha=0.10, lw=0)
    ax.axvline(0, color="#4a4a46", lw=0.7)
    for i, r in enumerate(rows.itertuples()):
        y = len(rows) - 1 - i
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=colour, lw=1.6, solid_capstyle="butt")
        ax.plot([r.delta], [y], "o", color=colour, ms=4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(list(rows.label)[::-1])
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_title(title, loc="left", color=colour, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="results/preference/preference_steps.csv")
    ap.add_argument("--out", default="figures/fig4_preference.png")
    a = ap.parse_args()
    d = pd.read_csv(a.steps)
    fig = plt.figure(figsize=(6.4, 4.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[10, 2.6], hspace=0.6)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    forest(
        ax1, d[d.panel == "indeterminacy"], ORANGE, "(a) Indeterminacy step: (T, F, I) over (T, F)"
    )
    forest(
        ax2,
        d[d.panel == "evidence"],
        BLUE,
        "(b) Evidence-magnitude step on SHP: (T, F) over the scalar",
    )
    ax2.set_xlabel(r"$\Delta$AUC (paired bootstrap 95% CI)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)


if __name__ == "__main__":
    main()
