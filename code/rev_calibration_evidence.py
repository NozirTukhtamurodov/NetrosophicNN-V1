"""Revision analyses (post hoc), comments 17 and 16 (B9, B10): NLL, Brier score (sum over classes)
and ECE
(15 equal-width bins, top label) of every probability output; evidential diagnostics (S - K =
total evidence,
shares with evidence on <= 1 class or ~0, per-class evidence and accuracy vs softmax). Tables S9,
S10."""

import os

import common as C
import numpy as np

EPS = 1e-12


def calib(p, y):
    n = len(y)
    pt = np.clip(p[np.arange(n), y], EPS, 1)
    nll = float(-np.log(pt).mean())
    oh = np.zeros_like(p)
    oh[np.arange(n), y] = 1
    brier = float(((p - oh) ** 2).sum(1).mean())
    conf, pred = p.max(1), p.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, 16)
    b = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, 14)
    ece = sum(
        abs(acc[b == i].mean() - conf[b == i].mean()) * (b == i).mean()
        for i in range(15)
        if (b == i).any()
    )
    return dict(acc=float(acc.mean()), nll=nll, brier=brier, ece=float(ece))


def probs(model, d):
    if model in ("softmax", "hybrid_softmax"):
        out = {"softmax p": C.softmax(d["logits"].astype(float))}
        if "mc_mean_p" in d:
            out["MC dropout mean p"] = d["mc_mean_p"].astype(float) / d["mc_mean_p"].astype(
                float
            ).sum(1, keepdims=True)
        return out
    if model in ("evidential", "hybrid_evidential"):
        al = np.logaddexp(0, d["logits"].astype(float)) + 1
        return {"alpha/S": al / al.sum(1, keepdims=True)}
    if model == "nsu":
        T = 1 / (1 + np.exp(-d["zT"].astype(float)))
        return {
            "softmax(zT-zF)": C.softmax(d["zT"].astype(float) - d["zF"].astype(float)),
            "T/sum(T)": T / T.sum(1, keepdims=True),
        }
    return {}


def main():
    rows = []
    for sc, m, s, t, p in C.runs():
        if m == "softmax" and t >= 3:
            continue
        d = dict(np.load(p))
        y = d["y"].astype(int)
        for out, pr in probs(m, d).items():
            r = calib(pr, y)
            rows.append(
                [sc, m, out, s, t, len(y), *[f"{r[k]:.6f}" for k in ("acc", "nll", "brier", "ece")]]
            )
    for sc in C.SCENE_LIST:
        P = np.stack(
            [
                C.softmax(
                    np.load(os.path.join(C.HSI_Q1, f"{sc}_softmax_s0_t{t}.npz"))["logits"].astype(
                        float
                    )
                )
                for t in range(5)
            ]
        )
        y = np.load(os.path.join(C.HSI_Q1, f"{sc}_softmax_s0_t0.npz"))["y"].astype(int)
        r = calib(P.mean(0), y)
        rows.append(
            [
                sc,
                "ensemble5",
                "mean p",
                0,
                "0-4",
                len(y),
                *[f"{r[k]:.6f}" for k in ("acc", "nll", "brier", "ece")],
            ]
        )
    C.write_csv(
        C.REV / "c17_calibration_by_run.csv",
        [
            "scene",
            "model",
            "output",
            "split_seed",
            "train_seed",
            "n_test",
            "accuracy_argmax",
            "nll",
            "brier",
            "ece15",
        ],
        rows,
    )
    agg = {}
    for r in rows:
        agg.setdefault((r[0], r[1], r[2]), []).append([float(x) for x in r[6:]])
    summ = []
    for (sc, m, o), v in agg.items():
        v = np.array(v)
        sd = v.std(0, ddof=1) if len(v) > 1 else np.full(4, np.nan)
        summ.append(
            [
                sc,
                m,
                o,
                len(v),
                *[
                    f"{a:.4f} ± {b:.4f}" if len(v) > 1 else f"{a:.4f} (single run)"
                    for a, b in zip(v.mean(0), sd, strict=True)
                ],
            ]
        )
    C.write_csv(
        C.REV / "c17_calibration_summary.csv",
        ["scene", "model", "output", "n_runs", "accuracy_argmax", "nll", "brier", "ece15"],
        summ,
    )
    for r in summ:
        print(r)

    # ---------------------------------------------------------------- B10
    ev, pcl, pacc = [], [], []
    for sc, m, s, t, p in C.runs(["evidential", "hybrid_evidential"]):
        d = dict(np.load(p))
        y = d["y"].astype(int)
        e = np.logaddexp(0, d["logits"].astype(float))
        K = e.shape[1]
        tot = e.sum(1)
        q = np.percentile(tot, [5, 25, 50, 75, 95])
        ev.append(
            [
                sc,
                m,
                s,
                t,
                len(y),
                f"{tot.mean():.4f}",
                *[f"{x:.4f}" for x in q],
                f"{tot.max():.4f}",
                f"{np.mean((e > 0.01).sum(1) <= 1):.4f}",
                f"{np.mean((e >= 1).sum(1) <= 1):.4f}",
                f"{np.mean(tot < 0.01):.4f}",
                f"{np.mean(tot < 0.1):.4f}",
                f"{np.mean(tot < 1):.4f}",
                f"{float((e.argmax(1) == y).mean()):.4f}",
            ]
        )
        soft = os.path.join(
            C.HSI_Q1,
            f"{sc}_{'hybrid_softmax' if m.startswith('hybrid') else 'softmax'}_s{s}_t{t}.npz",
        )
        ps = np.load(soft)["logits"].argmax(1) if os.path.exists(soft) else None
        for k in range(K):
            mk = y == k
            if not mk.any():
                continue
            pcl.append(
                [
                    sc,
                    m,
                    s,
                    t,
                    k + 1,
                    int(mk.sum()),
                    int(d["train_counts"][k]),
                    f"{e[mk, k].mean():.4f}",
                    f"{tot[mk].mean():.4f}",
                    f"{e[:, k].mean():.4f}",
                ]
            )
            pacc.append(
                [
                    sc,
                    m,
                    s,
                    t,
                    k + 1,
                    int(mk.sum()),
                    int(d["train_counts"][k]),
                    f"{(e[mk].argmax(1) == k).mean():.4f}",
                    "" if ps is None else f"{(ps[mk] == k).mean():.4f}",
                ]
            )
    C.write_csv(
        C.REV / "c16_evidential_evidence_by_run.csv",
        [
            "scene",
            "model",
            "split_seed",
            "train_seed",
            "n_test",
            "mean_S_minus_K",
            "p05_S_minus_K",
            "p25",
            "p50",
            "p75",
            "p95",
            "max_S_minus_K",
            "share_evidence_gt0.01_on_le1_class",
            "share_evidence_ge1_on_le1_class",
            "share_total_evidence_lt0.01",
            "share_total_evidence_lt0.1",
            "share_total_evidence_lt1",
            "accuracy",
        ],
        ev,
    )
    C.write_csv(
        C.REV / "c16_evidential_evidence_by_class.csv",
        [
            "scene",
            "model",
            "split_seed",
            "train_seed",
            "true_class",
            "n_test_class",
            "n_train_class",
            "mean_evidence_on_true_class",
            "mean_total_evidence",
            "mean_evidence_on_this_class_all_pixels",
        ],
        pcl,
    )
    C.write_csv(
        C.REV / "c16_perclass_accuracy_evidential_vs_softmax.csv",
        [
            "scene",
            "model",
            "split_seed",
            "train_seed",
            "class",
            "n_test_class",
            "n_train_class",
            "acc_evidential",
            "acc_softmax_same_split_seed",
        ],
        pacc,
    )
    for r in ev:
        print(r)


if __name__ == "__main__":
    main()
