"""Fig. 1: setting S3, image and its Guo-Cheng T, I, F maps (from simulation.py)."""

import argparse
from pathlib import Path

import numpy as np
from style import plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default="results/simulation/s3_maps.npz")
    ap.add_argument("--out", default="figures/fig1_simulation.png")
    a = ap.parse_args()
    d = np.load(a.maps)
    titles = [
        "(a) image $g$",
        "(b) $T$ (local mean)",
        "(c) $I$ (local deviation)",
        "(d) $F = 1 - T$",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 1.95))
    for ax, key, title in zip(axes, ["g", "T", "I", "F"], titles):
        ax.imshow(d[key], cmap="gray", vmin=0, vmax=0.6 if key == "I" else 1)
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(w_pad=0.4)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)


if __name__ == "__main__":
    main()
