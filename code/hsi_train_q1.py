"""One hyperspectral run (scene x model x split seed x training seed) of the prespecified protocol
(Table 3).
Writes outputs/hsi_outputs_q1/<scene>_<model>_s<split>_t<seed>.npz with the per-pixel test
outputs."""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import HSI_Q1, load_scene, make_patches, split_block_buffer
from hsi_models import HSINet, HybridSN, NSUNet, edl_loss
from sklearn.decomposition import PCA


def preprocess_train_only(cube, train_mask, n_comp, seed):
    """Per-band standardisation and PCA fitted on the training pixels only, applied to the whole
    scene."""
    h, w, b = cube.shape
    flat = cube.reshape(-1, b)
    tr = flat[train_mask.ravel()]
    mu, sd = tr.mean(0), tr.std(0) + 1e-8
    pca = PCA(n_components=n_comp, random_state=seed).fit((tr - mu) / sd)
    out = pca.transform((flat - mu) / sd).reshape(h, w, n_comp).astype(np.float32)
    return out, float(pca.explained_variance_ratio_.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=["pavia_u", "salinas"])
    ap.add_argument(
        "--model",
        required=True,
        choices=["softmax", "evidential", "nsu", "hybrid_softmax", "hybrid_evidential"],
    )
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--tseed", type=int, required=True)
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--patch", type=int, default=11)
    ap.add_argument("--buffer", type=int, default=6)
    ap.add_argument("--pca", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--mc", type=int, default=20)
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(HSI_Q1, exist_ok=True)
    tag = f"{args.scene}_{args.model}_s{args.split}_t{args.tseed}"
    out = os.path.join(HSI_Q1, tag + ".npz")
    if os.path.exists(out):
        print(f"[skip] {out} exists")
        return

    dev = "cuda"
    cube, gt = load_scene(args.scene)
    tr_m, te_m = split_block_buffer(
        gt, args.frac, np.random.default_rng(args.split), buffer=args.buffer
    )
    torch.manual_seed(args.tseed)  # training seed; also the PCA random_state
    cube, evr = preprocess_train_only(cube, tr_m, args.pca, args.tseed)
    K = int(gt.max())
    Xtr, _, _ = make_patches(cube, tr_m, args.patch)
    ytr = gt[tr_m] - 1
    Xte, ys, xs = make_patches(cube, te_m, args.patch)
    yte = gt[te_m] - 1
    train_counts = np.bincount(ytr, minlength=K)
    print(
        f"[{tag}] train {len(ytr)}  test {len(yte)}  K={K}  PCA var={evr:.4f}  "
        f"classes without training pixels: {int((train_counts == 0).sum())}",
        flush=True,
    )

    hyb = args.model.startswith("hybrid")
    Net = HybridSN if hyb else (NSUNet if args.model == "nsu" else HSINet)
    net = Net(args.pca, K, args.patch).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr).long()
    evid = args.model.endswith("evidential")

    def loss_fn(o, yb, ep):
        if args.model == "nsu":  # eq. (6)
            zT, zF = o
            yh = F.one_hot(yb, K).float()
            return (
                F.binary_cross_entropy_with_logits(zT, yh, reduction="none").sum(1)
                + F.binary_cross_entropy_with_logits(zF, 1 - yh, reduction="none").sum(1)
            ).mean()
        if evid:
            return edl_loss(o, yb, K, ep)
        return F.cross_entropy(o, yb)

    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Xtr_t))
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i : i + args.batch]
            xb, yb = Xtr_t[idx].to(dev), ytr_t[idx].to(dev)
            opt.zero_grad()
            loss = loss_fn(net(xb), yb, ep)
            loss.backward()
            opt.step()
            tot += loss.detach().item() * len(idx)
        sch.step()
        print(f"  epoch {ep + 1:>3d}  loss {tot / len(perm):.5f}", flush=True)

    net.eval()
    res = {}
    with torch.no_grad():
        outs = [net(torch.from_numpy(Xte[i : i + 512]).to(dev)) for i in range(0, len(Xte), 512)]
        if args.model == "nsu":
            res["zT"] = torch.cat([o[0] for o in outs]).float().cpu().numpy()
            res["zF"] = torch.cat([o[1] for o in outs]).float().cpu().numpy()
            sT, sF = 1 / (1 + np.exp(-res["zT"])), 1 / (1 + np.exp(-res["zF"]))
            pred = (sT - sF).argmax(1)
        else:
            res["logits"] = torch.cat(outs).float().cpu().numpy()
            pred = res["logits"].argmax(1)

        if args.model == "softmax" and args.mc > 0:
            # MC dropout: only the Dropout layer in train mode, BatchNorm stays in eval mode
            torch.manual_seed(1000 + args.tseed)
            for m in net.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
            psum = np.zeros((len(Xte), K), np.float64)
            hsum = np.zeros(len(Xte), np.float64)
            for _ in range(args.mc):
                pr = (
                    torch.cat(
                        [
                            F.softmax(net(torch.from_numpy(Xte[i : i + 512]).to(dev)), 1)
                            for i in range(0, len(Xte), 512)
                        ]
                    )
                    .double()
                    .cpu()
                    .numpy()
                )
                psum += pr
                hsum += -(pr * np.log(pr + 1e-12)).sum(1)
            pm = psum / args.mc
            res["mc_mean_p"] = pm.astype(np.float32)
            res["mc_pe"] = -(pm * np.log(pm + 1e-12)).sum(1)
            res["mc_mi"] = res["mc_pe"] - hsum / args.mc
            net.eval()

    oa = float((pred == yte).mean())
    wall = time.time() - t0
    np.savez_compressed(
        out,
        y=yte.astype(np.int16),
        pred=pred.astype(np.int16),
        ys=ys.astype(np.int16),
        xs=xs.astype(np.int16),
        train_counts=train_counts,
        n_train=len(ytr),
        K=K,
        pca_var=evr,
        n_params=n_par,
        wall_s=wall,
        **res,
    )
    print(
        f"[done] {tag}: OA={oa:.4f} errors {int((pred != yte).sum())}  "
        f"parameters {n_par:,}  {wall:.1f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
