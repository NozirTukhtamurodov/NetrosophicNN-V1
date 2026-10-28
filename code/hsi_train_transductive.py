"""Section 6.5: evidential HSINet with the earlier transductive preprocessing (standardisation and
PCA fitted on
all pixels of the scene, before the split). One run per scene x seed (split seed = training seed).
Writes outputs/hsi_outputs_transductive/hsi_<scene>_s<seed>.jsonl, one record per test pixel."""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from common import OUTPUTS_DIR, SCENES, load_scene, make_patches, split_block_buffer
from hsi_models import HSINet, edl_loss
from sklearn.decomposition import PCA

OUT_DIR = os.path.join(OUTPUTS_DIR, "hsi_outputs_transductive")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="pavia_u", choices=list(SCENES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--patch", type=int, default=11)
    ap.add_argument("--buffer", type=int, default=6)
    ap.add_argument("--pca", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cube, gt = load_scene(args.scene)
    h, w, b = cube.shape
    flat = cube.reshape(-1, b)
    flat = (flat - flat.mean(0)) / (flat.std(0) + 1e-8)  # all pixels: transductive
    cube = (
        PCA(n_components=args.pca, random_state=args.seed)
        .fit_transform(flat)
        .reshape(h, w, args.pca)
        .astype(np.float32)
    )

    tr_m, te_m = split_block_buffer(gt, args.frac, rng, buffer=args.buffer)
    n_cls = int(gt.max())
    print(
        f"[{args.scene} seed={args.seed}] train {int(tr_m.sum())}, "
        f"test {int(te_m.sum())}, classes {n_cls}, buffer {args.buffer}"
    )

    Xtr, _, _ = make_patches(cube, tr_m, args.patch)
    ytr = gt[tr_m] - 1
    Xte, ys, xs = make_patches(cube, te_m, args.patch)
    yte = gt[te_m] - 1

    net = HSINet(args.pca, n_cls, args.patch).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr).long()

    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Xtr_t))
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i : i + args.batch]
            xb, yb = Xtr_t[idx].to(dev), ytr_t[idx].to(dev)
            opt.zero_grad()
            loss = edl_loss(net(xb), yb, n_cls, ep)
            loss.backward()
            opt.step()
            tot += loss.detach().item() * len(idx)
        sch.step()
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep + 1:>3d}  loss {tot / len(perm):.4f}", flush=True)

    net.eval()
    recs = []
    with torch.no_grad():
        for i in range(0, len(Xte), 512):
            xb = torch.from_numpy(Xte[i : i + 512]).to(dev)
            ev = F.softplus(net(xb))
            alpha = ev + 1.0
            S = alpha.sum(1, keepdim=True)
            bel = (ev / S).cpu().numpy()
            u = (n_cls / S).cpu().numpy().ravel()
            p = (alpha / S).cpu().numpy()
            ent = -(p * np.log(p + 1e-12)).sum(1)
            pred = bel.argmax(1)
            srt = np.sort(bel, 1)[:, ::-1]
            for j in range(len(pred)):
                k = i + j
                recs.append(
                    {
                        "y": int(yte[k]),
                        "pred": int(pred[j]),
                        "err": int(pred[j] != yte[k]),
                        "u": float(u[j]),
                        "b_max": float(srt[j, 0]),
                        "b_2nd": float(srt[j, 1]),
                        "entropy": float(ent[j]),  # entropy of alpha/S
                        "S": float(S[j].item()),
                    }
                )

    acc = float(np.mean([r["pred"] == r["y"] for r in recs]))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"hsi_{args.scene}_s{args.seed}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(
        f"[done] {out}: {len(recs)} test pixels, OA={acc:.4f}, errors {sum(r['err'] for r in recs)}"
    )


if __name__ == "__main__":
    main()
