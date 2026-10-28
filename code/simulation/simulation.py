"""Section 5: synthetic settings S1-S3 (Table 2) and the S3 maps used in Fig. 1."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 1337
B = 3000


def oof_scores(X, y, seed=SEED):
    """Out-of-fold probabilities of a standardized logistic regression (Section 4)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    out = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        model.fit(X[tr], y[tr])
        out[te] = model.predict_proba(X[te])[:, 1]
    return out


def ladder(rungs, y, seed=SEED):
    """AUC of each rung and paired bootstrap intervals for consecutive steps; rule (10)."""
    scores = [oof_scores(X, y, seed) for _, X in rungs]
    aucs = [roc_auc_score(y, s) for s in scores]
    rng = np.random.default_rng(seed)
    n = len(y)
    boot = np.zeros((B, len(rungs)))
    for b in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            boot[b] = np.nan
            continue
        for j, s in enumerate(scores):
            boot[b, j] = roc_auc_score(yb, s[idx])
    res = {"auc": {name: round(a, 4) for (name, _), a in zip(rungs, aucs)}, "steps": []}
    for j in range(1, len(rungs)):
        d = boot[:, j] - boot[:, j - 1]
        d = d[~np.isnan(d)]
        delta = aucs[j] - aucs[j - 1]
        lo, hi = np.percentile(d, [2.5, 97.5])
        res["steps"].append(
            {
                "from": rungs[j - 1][0],
                "to": rungs[j][0],
                "delta": round(float(delta), 4),
                "ci": [round(float(lo), 4), round(float(hi), 4)],
                "passes_rule": bool(delta > 0.03 and lo > 0),
            }
        )
    return res


def s1(n=20000, seed=SEED):
    """Normalized classifier triple: binary evidential classifier, W = 2, base rate 1/2."""
    rng = np.random.default_rng(seed)
    W = 2.0
    q = np.exp(rng.uniform(np.log(0.3), np.log(30.0), n))
    r_true = rng.gamma(shape=1.0 * q, scale=1.0)
    r_false = rng.gamma(shape=0.6 * q, scale=1.0)
    err = (r_false > r_true).astype(int)
    r_pred, r_other = np.maximum(r_true, r_false), np.minimum(r_true, r_false)
    S = r_pred + r_other + W
    T, F, I = r_pred / S, r_other / S, W / S
    P = T + 0.5 * I
    rungs = [("P", P), ("(T, F)", np.c_[T, F]), ("(T, F, I)", np.c_[T, F, I])]
    out = ladder(rungs, err, seed)
    out["n"] = n
    out["error_rate"] = round(float(err.mean()), 4)
    out["rank_1_T_F_I"] = int(np.linalg.matrix_rank(np.c_[np.ones(n), T, F, I]))  # Corollary 2
    return out


def s2(n=20000, seed=SEED):
    """Unnormalized triple from a generalized mass assignment, mapped by (3)."""
    rng = np.random.default_rng(seed + 1)
    m1, m2, m12, mU = rng.dirichlet([1.0, 1.0, 1.0, 1.0], size=n).T
    T, F, I = m1 + m12, m2 + m12, mU
    p = 1.0 / (1.0 + np.exp(-(-2.5 + 7.0 * m12)))
    y = (rng.uniform(size=n) < p).astype(int)
    rungs = [("T - F", T - F), ("(T, F)", np.c_[T, F]), ("(T, F, I)", np.c_[T, F, I])]
    out = ladder(rungs, y, seed)
    out["n"] = n
    out["positive_rate"] = round(float(y.mean()), 4)
    return out


def guo_cheng(g, w=5):
    """Neutrosophic image transform (5): T, I, F maps of a grayscale image."""
    gbar = uniform_filter(g, size=w, mode="reflect")
    delta = np.abs(g - gbar)
    T = (gbar - gbar.min()) / (gbar.max() - gbar.min())
    I = (delta - delta.min()) / (delta.max() - delta.min())
    return T, I, 1.0 - T


def s3(size=256, seed=SEED, n_sample=20000):
    """Guo-Cheng triple on a textured disk with the same mean as the background."""
    rng = np.random.default_rng(seed + 2)
    yy, xx = np.mgrid[0:size, 0:size]
    disk = ((xx - size / 2) ** 2 + (yy - size / 2) ** 2) < (size * 0.3) ** 2
    g = np.full((size, size), 0.5)
    g = np.clip(
        g + np.where(disk, rng.normal(0, 0.15, g.shape), rng.normal(0, 0.02, g.shape)), 0.0, 1.0
    )
    T, I, F = guo_cheng(g, w=5)
    lab = disk.astype(int).ravel()
    idx = rng.choice(lab.size, size=n_sample, replace=False)
    Tf, If, Ff, y = T.ravel()[idx], I.ravel()[idx], F.ravel()[idx], lab[idx]
    rungs = [("T", Tf), ("(T, F)", np.c_[Tf, Ff]), ("(T, F, I)", np.c_[Tf, Ff, If])]
    out = ladder(rungs, y, seed)
    out["n"] = n_sample
    return out, {"g": g, "T": T, "I": I, "F": F, "mask": disk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/simulation")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    r3, maps = s3()
    results = {"S1": s1(), "S2": s2(), "S3": r3}
    (out / "sim_results.json").write_text(json.dumps(results, indent=2))
    np.savez_compressed(out / "s3_maps.npz", **maps)


if __name__ == "__main__":
    main()
