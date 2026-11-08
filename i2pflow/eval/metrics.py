"""Accuracy and uncertainty metrics shared by all methods.

Accuracy (I2P convention): RRE = sum |Euler| of inv(pred) @ gt (also geodesic), RTE = ||t||;
Acc = RRE < 5 deg & RTE < 2 m, RR = RRE < 10 deg & RTE < 5 m.

Uncertainty, computed from N pose samples per frame so that every method is scored the same way:
  * KDE NLL of the ground truth with a product Gaussian kernel on the active tangent coordinates
    (bandwidth h_rot rad, h_trans m)  -- comparable across methods
  * exact NLL when a method provides it (flow: reverse ODE; Gaussian: closed form)
  * HPD coverage: the GT's HPD level is the fraction of samples whose KDE density exceeds the
    GT's; coverage(alpha) = mean(level <= alpha), ideal = alpha
  * failure-detection AUROC of a scalar uncertainty score against Acc failures
"""
import math

import numpy as np
import torch

from i2pflow.geometry import so3

ACC_THR = (5.0, 2.0)
RR_THR = (10.0, 5.0)


def accuracy_metrics(rre, rte, prefix=""):
    rre, rte = np.asarray(rre), np.asarray(rte)
    acc = (rre < ACC_THR[0]) & (rte < ACC_THR[1])
    rr = (rre < RR_THR[0]) & (rte < RR_THR[1])
    out = {
        "acc": 100.0 * acc.mean(), "rr": 100.0 * rr.mean(),
        "rre_mean": rre.mean(), "rre_std": rre.std(), "rte_mean": rte.mean(), "rte_std": rte.std(),
        "rre_median": np.median(rre), "rte_median": np.median(rte),
        "rre_mean_on_acc": rre[acc].mean() if acc.any() else float("nan"),
        "rte_mean_on_acc": rte[acc].mean() if acc.any() else float("nan"),
        "n": int(rre.size),
    }
    return {prefix + k: float(v) for k, v in out.items()}


def tangent_residuals(R_s, t_s, R_gt, t_gt, space):
    """Residuals of samples [B,N] w.r.t. the GT [B] in active tangent coords (rad, m) -> [B,N,D]."""
    rm, tm = space.masks(R_s.device, R_s.dtype)
    r_rot = so3.log(R_gt[:, None] @ R_s.transpose(-1, -2)) * rm
    r_t = (t_gt[:, None] - t_s) * tm
    return torch.cat([r_rot, r_t], -1)[..., space.active.to(R_s.device)]


def _kde_logpdf(x, centers, bw, leave_one_out=False):
    """log of mean_k N(x; c_k, diag(bw^2)); x [B,M,D], centers [B,N,D], bw [D] -> [B,M].

    leave_one_out (x is centers): exclude each point's own kernel.
    """
    z = (x[:, :, None] - centers[:, None]) / bw
    log_k = -0.5 * (z * z).sum(-1) - torch.log(bw).sum() - 0.5 * bw.numel() * math.log(2 * math.pi)
    n = centers.shape[1]
    if leave_one_out:
        eye = torch.eye(n, dtype=torch.bool, device=log_k.device)
        log_k = log_k.masked_fill(eye, -float("inf"))
        n -= 1
    return torch.logsumexp(log_k, dim=-1) - math.log(n)


def kde_scores(R_s, t_s, R_gt, t_gt, space, bw_rot_deg=2.0, bw_trans=0.5):
    """Returns (KDE log-density of GT [B], HPD level of GT [B])."""
    res = tangent_residuals(R_s, t_s, R_gt, t_gt, space)          # sample - centred at GT
    D = space.dim
    bw = torch.tensor([math.radians(bw_rot_deg)] * space.rot_dim + [bw_trans] * space.trans_dim,
                      device=R_s.device, dtype=res.dtype)
    zero = torch.zeros(res.shape[0], 1, D, device=res.device, dtype=res.dtype)
    log_gt = _kde_logpdf(zero, res, bw)[:, 0]
    log_s = _kde_logpdf(res, res, bw, leave_one_out=True)
    level = (log_s > log_gt[:, None]).float().mean(-1)
    return log_gt, level


def coverage_curve(levels, alphas=np.linspace(0.05, 0.95, 19)):
    levels = np.asarray(levels)
    cov = np.array([(levels <= a).mean() for a in alphas])
    return alphas, cov, float(np.abs(cov - alphas).mean())


def auroc(score, positive):
    """AUROC of `score` for detecting `positive` (e.g. failures); ties count half."""
    score, positive = np.asarray(score, dtype=np.float64), np.asarray(positive, dtype=bool)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def sample_spread(R_s, t_s, R_ref, t_ref):
    """Mean rotation (deg) and translation (m) deviation of samples [B,N] from a reference [B]."""
    rot = torch.rad2deg(so3.geodesic_distance(R_s, R_ref[:, None].expand_as(R_s))).mean(-1)
    tr = (t_s - t_ref[:, None]).norm(dim=-1).mean(-1)
    return rot, tr


def mode_recall(modes, gt_poses, eps_rot_deg=5.0, eps_trans=2.0):
    """Fraction of ground-truth modes (list of (R, t)) matched by some predicted mode."""
    hit = 0
    for R_g, t_g in gt_poses:
        for m in modes:
            if math.degrees(so3.geodesic_distance(m["R"], R_g).item()) < eps_rot_deg and \
                    (m["t"] - t_g).norm().item() < eps_trans:
                hit += 1
                break
    return hit / max(len(gt_poses), 1)
