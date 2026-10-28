"""Revision analysis (post hoc), comment 8 (D), Table S17: positive control on CIFAR-10H (plan:
plan/D_plan.md).
ResNet-18 with a classifier head (CIFAR-10 labels) and an indeterminacy head regressing the
annotator-label entropy;
ladder (H(p), p_max) -> (+ predicted disagreement) -> (sorted p, predicted disagreement),
detector and rule as Table 4.
Usage: python cifar10h_positive_control.py train   (GPU, resumable per fold), then  ... analyse"""

import os
import sys
import time

import cifar10h_common as CF
import common as C
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

OUTD = os.path.join(C.OUTPUTS_DIR, "cifar10h_outputs")
MS = ["all", 20, 10, 5, 3, 2]


def target_H(counts, m):
    """Entropy / ln 10 of the labels of m annotators per image drawn without replacement (rng 1337);
    m = "all": all."""
    if m == "all":
        c = counts
    else:
        rng = np.random.default_rng(1337)
        c = np.zeros_like(counts)
        for i, row in enumerate(counts.astype(int)):
            labels = np.repeat(np.arange(10), row)
            c[i] = np.bincount(rng.choice(labels, m, replace=False), minlength=10)
    p = c / c.sum(1, keepdims=True)
    return (-(p * np.log(np.where(p > 0, p, 1))).sum(1) / np.log(10)).astype(np.float32)


def train_fold(X, y, h, tr, seed, init, dev, epochs=30, bs=128):
    torch.manual_seed(seed)
    model = CF.resnet18(11)
    model.load_state_dict({k: v for k, v in init.items() if not k.startswith("fc.")}, strict=False)
    model = model.to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator().manual_seed(seed)
    Xg = torch.from_numpy(X[tr]).to(dev)
    yg, hg = torch.from_numpy(y[tr]).to(dev), torch.from_numpy(h[tr]).to(dev)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xg), generator=gen).to(dev)
        for s in range(0, len(Xg) - bs + 1, bs):
            b = perm[s : s + bs]
            x = CF.norm(CF.augment(Xg[b], gen, dev), dev).contiguous(
                memory_format=torch.channels_last
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(x)
            o = o.float()
            loss = F.cross_entropy(o[:, :10], yg[b]) + F.mse_loss(torch.sigmoid(o[:, 10]), hg[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sch.step()
    model.eval()
    return model


def train_all():
    dev = "cuda"
    raw = CF.load_raw()
    X, y = raw["Xte"], raw["yte"]
    os.makedirs(OUTD, exist_ok=True)
    runs = [("all", s) for s in (0, 1, 2)] + [(m, 0) for m in MS if m != "all"]
    for m, s in runs:
        h = target_H(raw["counts"], m)
        init = CF.pretrain(s, raw, dev)
        for f, (tr, _va, te) in enumerate(CF.folds()):
            out = os.path.join(OUTD, f"m{m}_s{s}_fold{f}.npz")
            if os.path.exists(out):
                continue
            t0 = time.time()
            model = train_fold(X, y, h, tr, 100 * s + f, init, dev)
            o = CF.outputs(model, X[te], dev)
            np.savez(out, o=o.astype(np.float32), te=te, wall_s=time.time() - t0)
            del model
            torch.cuda.empty_cache()
            print(f"m={m} seed={s} fold {f}: {time.time() - t0:.0f} s", flush=True)


def analyse():
    raw_c = np.load(os.path.join(C.DATA_DIR, "cifar10h", CF.COUNTS)).astype(float)
    yte = np.asarray(
        torchvision.datasets.CIFAR10(
            os.path.join(C.DATA_DIR, "torchvision"), train=False, download=False
        ).targets
    )
    rows = []
    for m in MS:
        for s in (0, 1, 2) if m == "all" else (0,):
            fs = [os.path.join(OUTD, f"m{m}_s{s}_fold{f}.npz") for f in range(5)]
            if not all(os.path.exists(f) for f in fs):
                rows.append([m, s, "MISSING"] + [""] * 9)
                continue
            O = np.zeros((10000, 11))
            for f in fs:
                z = np.load(f)
                O[z["te"]] = z["o"]
            p = C.softmax(O[:, :10])
            Ihat = 1 / (1 + np.exp(-O[:, 10]))
            y = (p.argmax(1) != yte).astype(int)
            H = C.ent(p)
            feats = {
                "D1": np.c_[H, p.max(1)],
                "D2": np.c_[H, p.max(1), Ihat],
                "D3": np.c_[-np.sort(-p, 1), Ihat],
            }
            sc = {k: C.oof_scores(X, y) for k, X in feats.items()}
            res, pt = C.boot_dist(y, sc, [("D2", "D1"), ("D3", "D2")])
            Hall = target_H(raw_c, "all")
            rho = np.corrcoef(np.argsort(np.argsort(Ihat)), np.argsort(np.argsort(Hall)))[0, 1]
            for st in (("D2", "D1"), ("D3", "D2")):
                sm = C.summ(*res[st])
                rows.append(
                    [
                        m,
                        s,
                        f"{st[0]}-{st[1]}",
                        f"{(y == 0).mean():.4f}",
                        int(y.sum()),
                        f"{pt[st[0]]:.4f}",
                        f"{pt[st[1]]:.4f}",
                        f"{sm['mean']:+.4f}",
                        f"{sm['lo']:+.4f}",
                        f"{sm['hi']:+.4f}",
                        C.rule(sm["mean"], sm["lo"]),
                        f"{rho:.4f}",
                    ]
                )
    C.write_csv(
        C.REV / "c08_cifar10h_positive_control.csv",
        [
            "annotators_per_image_m",
            "seed",
            "step",
            "accuracy",
            "errors",
            "auc_upper",
            "auc_lower",
            "delta_mean",
            "ci_lo",
            "ci_hi",
            "rule_pass",
            "spearman_Ihat_vs_H_all_annotators",
        ],
        rows,
    )


if __name__ == "__main__":
    {"train": train_all, "analyse": analyse}[sys.argv[1]]()
