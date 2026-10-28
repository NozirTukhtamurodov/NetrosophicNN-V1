"""Revision analyses (post hoc): hyperspectral training with saved weights (the q1 runs did not save
them).
Same functions, call order and seeds as hsi_train_q1.py (cuDNN non-deterministic, as in q1);
variants:
  replica            prespecified protocol (used by C1(a) and C3)
  nsu_pen            C1(b): NS-U with explicit penalty lambda/2 |heads|^2 in the loss, heads with
  Adam without
                     decoupled weight decay, backbone unchanged (AdamW, wd 1e-4); 100 epochs
  evid_log           C2: Type-II maximum-likelihood evidential loss, same annealing min(1, t/10)
  evid_mse_anneal20  C2: MSE-form evidential loss, annealing min(1, t/20)
Writes outputs/hsi_outputs_revision/<variant>/<tag>.npz and
outputs/weights_revision/<variant>/<tag>.pt."""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import HSI_REV, WEIGHTS_REV, load_scene, make_patches, split_block_buffer
from hsi_models import HSINet, HybridSN, NSUNet, edl_loss, evid_log_loss, evid_mse_loss_a20
from sklearn.decomposition import PCA


def preprocess(cube, train_mask, n_comp, seed):
    """As hsi_train_q1.preprocess_train_only; also returns the fitted parameters (reused for the
    shift in C3)."""
    h, w, b = cube.shape
    flat = cube.reshape(-1, b)
    tr = flat[train_mask.ravel()]
    mu, sd = tr.mean(0), tr.std(0) + 1e-8
    pca = PCA(n_components=n_comp, random_state=seed).fit((tr - mu) / sd)
    out = pca.transform((flat - mu) / sd).reshape(h, w, n_comp).astype(np.float32)
    return (
        out,
        float(pca.explained_variance_ratio_.sum()),
        dict(mu=mu, sd=sd, pca_mean=pca.mean_, pca_comp=pca.components_),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument(
        "--model",
        required=True,
        choices=["softmax", "evidential", "nsu", "hybrid_softmax", "hybrid_evidential"],
    )
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--tseed", type=int, required=True)
    ap.add_argument(
        "--variant",
        default="replica",
        choices=["replica", "nsu_pen", "evid_log", "evid_mse_anneal20"],
    )
    ap.add_argument("--lam", type=float, default=0.0)
    a = ap.parse_args()
    epochs = 100 if a.variant == "nsu_pen" else 40
    vdir = a.variant + (f"_lam{a.lam:g}" if a.variant == "nsu_pen" else "")
    tag = f"{a.scene}_{a.model}_s{a.split}_t{a.tseed}"
    out = os.path.join(HSI_REV, vdir, tag + ".npz")
    if os.path.exists(out):
        print(f"[skip] {out}")
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.makedirs(os.path.join(WEIGHTS_REV, vdir), exist_ok=True)
    t0 = time.time()
    dev = "cuda"
    cube, gt = load_scene(a.scene)
    tr_m, te_m = split_block_buffer(gt, 0.15, np.random.default_rng(a.split), buffer=6)
    torch.manual_seed(a.tseed)
    cube, evr, prep = preprocess(cube, tr_m, 30, a.tseed)
    K = int(gt.max())
    Xtr, _, _ = make_patches(cube, tr_m, 11)
    ytr = gt[tr_m] - 1
    Xte, ys, xs = make_patches(cube, te_m, 11)
    yte = gt[te_m] - 1
    train_counts = np.bincount(ytr, minlength=K)
    hyb = a.model.startswith("hybrid")
    Net = HybridSN if hyb else (NSUNet if a.model == "nsu" else HSINet)
    net = Net(30, K, 11).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    if a.variant == "nsu_pen":
        head = list(net.hT.parameters()) + list(net.hF.parameters())
        hid = {id(p) for p in head}
        body = [p for p in net.parameters() if id(p) not in hid]
        opt = torch.optim.AdamW(
            [{"params": body, "weight_decay": 1e-4}, {"params": head, "weight_decay": 0.0}], lr=1e-3
        )
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr).long()
    evid = a.model.endswith("evidential")

    def head_pen():
        return sum((p**2).sum() for p in list(net.hT.parameters()) + list(net.hF.parameters()))

    def loss_fn(o, yb, ep):
        if a.model == "nsu":
            zT, zF = o
            yh = F.one_hot(yb, K).float()
            L = (
                F.binary_cross_entropy_with_logits(zT, yh, reduction="none").sum(1)
                + F.binary_cross_entropy_with_logits(zF, 1 - yh, reduction="none").sum(1)
            ).mean()
            return L + (a.lam / 2) * head_pen() if a.variant == "nsu_pen" else L
        if evid:
            if a.variant == "evid_log":
                return evid_log_loss(o, yb, K, ep)
            if a.variant == "evid_mse_anneal20":
                return evid_mse_loss_a20(o, yb, K, ep)
            return edl_loss(o, yb, K, ep)
        return F.cross_entropy(o, yb)

    curve = []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr_t))
        tot = 0.0
        for i in range(0, len(perm), 128):
            idx = perm[i : i + 128]
            xb, yb = Xtr_t[idx].to(dev), ytr_t[idx].to(dev)
            opt.zero_grad()
            loss = loss_fn(net(xb), yb, ep)
            loss.backward()
            opt.step()
            tot += loss.detach().item() * len(idx)
        sch.step()
        curve.append(tot / len(perm))
        print(f"  epoch {ep + 1:>3d}  loss {curve[-1]:.6f}", flush=True)

    net.eval()
    res = {}

    def feats(X):
        with torch.no_grad():
            return (
                torch.cat(
                    [net.f(torch.from_numpy(X[i : i + 512]).to(dev)) for i in range(0, len(X), 512)]
                )
                .cpu()
                .numpy()
            )

    with torch.no_grad():
        outs = [net(torch.from_numpy(Xte[i : i + 512]).to(dev)) for i in range(0, len(Xte), 512)]
        if a.model == "nsu":
            res["zT"] = torch.cat([o[0] for o in outs]).float().cpu().numpy()
            res["zF"] = torch.cat([o[1] for o in outs]).float().cpu().numpy()
            sT, sF = 1 / (1 + np.exp(-res["zT"])), 1 / (1 + np.exp(-res["zF"]))
            pred = (sT - sF).argmax(1)
            res["h_train"], res["h_test"], res["y_train"] = (
                feats(Xtr),
                feats(Xte),
                ytr.astype(np.int16),
            )
        else:
            res["logits"] = torch.cat(outs).float().cpu().numpy()
            pred = res["logits"].argmax(1)
        if a.model == "softmax":  # MC dropout, as in hsi_train_q1.py
            torch.manual_seed(1000 + a.tseed)
            for m in net.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
            psum = np.zeros((len(Xte), K), np.float64)
            hsum = np.zeros(len(Xte), np.float64)
            for _ in range(20):
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
            pm = psum / 20
            res["mc_mean_p"] = pm.astype(np.float32)
            res["mc_pe"] = -(pm * np.log(pm + 1e-12)).sum(1)
            res["mc_mi"] = res["mc_pe"] - hsum / 20
            net.eval()
    # C1(b): full-batch gradient of the NS-U heads at the end of training (eval mode)
    if a.model == "nsu":
        net.zero_grad()
        Ltot = 0.0
        for i in range(0, len(Xtr), 512):
            xb = torch.from_numpy(Xtr[i : i + 512]).to(dev)
            yb = torch.from_numpy(ytr[i : i + 512]).long().to(dev)
            zT, zF = net(xb)
            yh = F.one_hot(yb, K).float()
            li = (
                F.binary_cross_entropy_with_logits(zT, yh, reduction="none").sum(1)
                + F.binary_cross_entropy_with_logits(zF, 1 - yh, reduction="none").sum(1)
            ).sum() / len(Xtr)
            li.backward()
            Ltot += float(li)
        if a.variant == "nsu_pen":
            ((a.lam / 2) * head_pen()).backward()
        g = torch.cat(
            [p.grad.flatten() for p in list(net.hT.parameters()) + list(net.hF.parameters())]
        )
        res["head_grad_norm_end"] = float(g.norm())
        res["full_train_loss_end_eval_mode"] = Ltot
    torch.save(net.state_dict(), os.path.join(WEIGHTS_REV, vdir, tag + ".pt"))
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
        loss_curve=np.array(curve),
        epochs=epochs,
        lam=a.lam,
        prep_mu=prep["mu"],
        prep_sd=prep["sd"],
        prep_pca_mean=prep["pca_mean"],
        prep_pca_comp=prep["pca_comp"],
        **res,
    )
    with open(os.path.join(HSI_REV, "train_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                dict(
                    variant=vdir,
                    tag=tag,
                    wall_s=round(wall, 1),
                    epochs=epochs,
                    lam=a.lam,
                    oa=float((pred == yte).mean()),
                    finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
            + "\n"
        )
    print(f"[done] {vdir}/{tag}: OA={(pred == yte).mean():.4f}  {wall:.1f} s", flush=True)


if __name__ == "__main__":
    main()
