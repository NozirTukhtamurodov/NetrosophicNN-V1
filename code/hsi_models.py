"""HSINet, NS-U and HybridSN-style networks and the evidential losses (Section 6.1, Table 3)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HSINet(nn.Module):
    """2-D CNN on (bands, 11, 11) patches; head = dropout 0.3 + linear."""

    def __init__(self, bands, n_cls, patch):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(bands, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n_cls))

    def forward(self, x):
        return self.head(self.f(x))


class NSUNet(nn.Module):
    """HSINet body with two independent linear heads z_T, z_F (NS-U, eq. (6))."""

    def __init__(self, bands, n_cls, patch):
        super().__init__()
        base = HSINet(bands, n_cls, patch)
        self.f = base.f
        self.drop = nn.Dropout(0.3)
        self.hT = nn.Linear(128, n_cls)
        self.hF = nn.Linear(128, n_cls)

    def forward(self, x):
        h = self.drop(self.f(x))
        return self.hT(h), self.hF(h)


class HybridSN(nn.Module):
    """HybridSN-style 3D->2D network (after Roy et al., 2020): 3D 8@(7x3x3) -> 16@(5x3x3) ->
    32@(3x3x3)
    -> 2D 64@3x3 -> FC256 -> dropout 0.4 -> FC K. Written for this study; 11x11 patches instead
    of 25x25."""

    def __init__(self, bands, n_cls, patch):
        super().__init__()
        self.c3 = nn.Sequential(
            nn.Conv3d(1, 8, (7, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(8, 16, (5, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(16, 32, (3, 3, 3)),
            nn.ReLU(),
        )
        d, s = bands - 6 - 4 - 2, patch - 6
        self.c2 = nn.Sequential(nn.Conv2d(32 * d, 64, 3), nn.ReLU())
        s2 = s - 2
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * s2 * s2, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_cls),
        )

    def forward(self, x):
        z = self.c3(x.unsqueeze(1))  # (B, 32, d, s, s)
        z = z.reshape(z.shape[0], -1, z.shape[3], z.shape[4])
        return self.head(self.c2(z))


def edl_loss(logits, y, n_cls, epoch):
    """MSE-form evidential loss (Sensoy et al., 2018) + KL to the uniform Dirichlet on the wrong
    classes,
    annealed with lambda_t = min(1, t/10)."""
    ev = F.softplus(logits)
    alpha = ev + 1.0
    S = alpha.sum(1, keepdim=True)
    yh = F.one_hot(y, n_cls).float()
    err = ((yh - alpha / S) ** 2).sum(1)
    var = (alpha * (S - alpha) / (S * S * (S + 1))).sum(1)

    a_t = yh + (1 - yh) * alpha
    St = a_t.sum(1, keepdim=True)
    kl = (
        torch.lgamma(St.squeeze(1))
        - torch.lgamma(torch.tensor(float(n_cls), device=logits.device))
        - torch.lgamma(a_t).sum(1)
        + ((a_t - 1) * (torch.digamma(a_t) - torch.digamma(St))).sum(1)
    )
    lam = min(1.0, epoch / 10.0)
    return (err + var + lam * kl).mean()


def kl_term(alpha, yh, K):
    """KL(Dir(alpha_tilde) || Dir(1)) with the true-class evidence removed."""
    a_t = yh + (1 - yh) * alpha
    St = a_t.sum(1, keepdim=True)
    return (
        torch.lgamma(St.squeeze(1))
        - torch.lgamma(torch.tensor(float(K), device=alpha.device))
        - torch.lgamma(a_t).sum(1)
        + ((a_t - 1) * (torch.digamma(a_t) - torch.digamma(St))).sum(1)
    )


def evid_log_loss(logits, y, K, ep):
    """Type-II maximum-likelihood ("log") loss of Sensoy et al. (2018), eq. (3), + KL (C2)."""
    alpha = F.softplus(logits) + 1.0
    S = alpha.sum(1, keepdim=True)
    yh = F.one_hot(y, K).float()
    return (
        (yh * (torch.log(S) - torch.log(alpha))).sum(1)
        + min(1.0, ep / 10.0) * kl_term(alpha, yh, K)
    ).mean()


def evid_mse_loss_a20(logits, y, K, ep):
    """MSE-form evidential loss with annealing lambda_t = min(1, t/20) (C2)."""
    ev = F.softplus(logits)
    alpha = ev + 1.0
    S = alpha.sum(1, keepdim=True)
    yh = F.one_hot(y, K).float()
    err = ((yh - alpha / S) ** 2).sum(1)
    var = (alpha * (S - alpha) / (S * S * (S + 1))).sum(1)
    return (err + var + min(1.0, ep / 20.0) * kl_term(alpha, yh, K)).mean()
