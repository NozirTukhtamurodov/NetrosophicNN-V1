"""CIFAR-10H positive control (Table S17): data, 5 folds, ResNet-18 for 32x32 inputs, pretraining on
CIFAR-10 train."""

import hashlib
import math
import os

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
)  # deterministic cuBLAS for the pretraining

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision  # noqa: E402
from common import DATA_DIR, OUTPUTS_DIR  # noqa: E402

COUNTS = "cifar10h-counts.npy"
COUNTS_SHA = "9b36b4550349af56c79083356312eb4af856cc5fd3caf1f669d40b9d0bbb69ff"
N_IMG, N_FOLD = 10000, 5
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
PRETRAIN_DIR = os.path.join(OUTPUTS_DIR, "weights_revision", "cifar10_pretrain")


def to_u8(X):
    """(n, H, W, 3) float in [0, 1] -> (n, 3, H, W) uint8."""
    return (np.clip(X, 0, 1) * 255).round().astype(np.uint8).transpose(0, 3, 1, 2).copy()


def load_raw(data_dir=DATA_DIR):
    """CIFAR-10H vote counts (10000, 10) and CIFAR-10 train/test images (uint8 NCHW) and labels."""
    p = os.path.join(data_dir, "cifar10h", COUNTS)
    with open(p, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    if h != COUNTS_SHA:
        raise RuntimeError(f"sha256 of {COUNTS} does not match: {h}")
    c = np.load(p).astype(np.float64)
    r = os.path.join(data_dir, "torchvision")
    tr = torchvision.datasets.CIFAR10(r, train=True, download=False)
    te = torchvision.datasets.CIFAR10(r, train=False, download=False)
    u8 = lambda a: to_u8(np.asarray(a, np.float32) / 255.0)  # noqa: E731
    return dict(
        counts=c,
        Xtr=u8(tr.data),
        ytr=np.asarray(tr.targets, np.int64),
        Xte=u8(te.data),
        yte=np.asarray(te.targets, np.int64),
    )


def folds():
    """5 folds of the 10,000 test images: (train 7000, validation 1000, test 2000) indices;
    permutation rng 777."""
    perm = np.random.default_rng(777).permutation(N_IMG)
    B = [perm[i * 2000 : (i + 1) * 2000] for i in range(N_FOLD)]
    out = []
    for f in range(N_FOLD):
        te, va = B[f], B[(f + 1) % N_FOLD][:1000]
        tr = np.setdiff1d(np.arange(N_IMG), np.concatenate([te, va]))
        out.append((tr, va, np.sort(te)))
    return out


def resnet18(out_dim):
    """torchvision ResNet-18 with a 3x3 stride-1 first convolution, no max-pool (32x32 inputs)."""
    m = torchvision.models.resnet18(num_classes=out_dim)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m


def norm(x, dev):
    return (x.float() / 255.0 - MEAN.to(dev)) / STD.to(dev)


def augment(xb, gen, dev):
    """Random shift (32-crop of the zero-padded 40x40 image) and horizontal flip; parameters from a
    CPU generator."""
    B = len(xb)
    i = torch.randint(0, 9, (B,), generator=gen).to(dev)
    j = torch.randint(0, 9, (B,), generator=gen).to(dev)
    fl = (torch.rand(B, generator=gen) < 0.5).to(dev)
    pad = F.pad(xb, (4, 4, 4, 4))
    ar = torch.arange(32, device=dev)
    rows, cols = (i[:, None] + ar)[:, None, :, None], (j[:, None] + ar)[:, None, None, :]
    out = pad[
        torch.arange(B, device=dev)[:, None, None, None],
        torch.arange(3, device=dev)[None, :, None, None],
        rows,
        cols,
    ]
    return torch.where(fl[:, None, None, None], out.flip(3), out)


@torch.no_grad()
def outputs(model, X, dev, bs=1000):
    """Logits (float64) of uint8 NCHW images, bf16 autocast."""
    model.eval()
    o = []
    for i in range(0, len(X), bs):
        x = norm(torch.from_numpy(X[i : i + bs]).to(dev), dev).contiguous(
            memory_format=torch.channels_last
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o.append(model(x).float().cpu())
    return torch.cat(o).numpy().astype(np.float64)


def train_softmax(X, y, seed, epochs, lr, dev, ckpt=None, bs=128):
    """Cross-entropy ResNet-18 (10 classes): SGD momentum 0.9, cosine lr, weight decay 5e-4, bf16
    autocast, crop + flip;
    the incomplete last batch is dropped. Per-epoch checkpoint for resuming."""
    torch.manual_seed(seed)
    model = resnet18(10).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator().manual_seed(seed)
    ep0 = 0
    if ckpt and os.path.exists(ckpt):
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        sch.load_state_dict(c["sch"])
        gen.set_state(c["gen"])
        torch.set_rng_state(c["cpu_rng"])
        torch.cuda.set_rng_state(c["cuda_rng"])
        ep0 = c["epoch"]
    Xg = torch.from_numpy(X).to(dev)
    yg = torch.from_numpy(y).to(dev)
    for ep in range(ep0, epochs):
        model.train()
        tot = 0.0
        perm = torch.randperm(len(Xg), generator=gen).to(dev)
        for s in range(0, len(Xg) - bs + 1, bs):
            b = perm[s : s + bs]
            x = norm(augment(Xg[b], gen, dev), dev).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
            loss = F.cross_entropy(out.float(), yg[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        sch.step()
        if not math.isfinite(tot):
            raise FloatingPointError(f"loss = {tot} at epoch {ep + 1}")
        print(f"  epoch {ep + 1}/{epochs} loss {tot / len(Xg):.4f}", flush=True)
        if ckpt:
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sch": sch.state_dict(),
                    "gen": gen.get_state(),
                    "epoch": ep + 1,
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                },
                ckpt + ".tmp",
            )
            os.replace(ckpt + ".tmp", ckpt)
    model.eval()
    return model


def pretrain(seed, raw, dev, epochs=100, lr=0.1):
    """ResNet-18 pretrained on CIFAR-10 train (official labels), one per seed; deterministic cuDNN
    as in the original run."""
    d = os.path.join(PRETRAIN_DIR, f"seed{seed}")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "model.pt")
    if os.path.exists(p):
        return torch.load(p, map_location="cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    m = train_softmax(
        raw["Xtr"], raw["ytr"], seed, epochs, lr, dev, ckpt=os.path.join(d, "ckpt_pre.pt")
    )
    sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    torch.save(sd, p + ".tmp")
    os.replace(p + ".tmp", p)
    os.remove(os.path.join(d, "ckpt_pre.pt"))
    return sd
