"""Matched Gaussian head: same features, tokens and transformer as the flow head, single shot.

It evaluates the network once at the identity pose (tau = 0) and outputs a mean pose plus a full
covariance (Cholesky factor) on the active tangent coordinates
    r = [Log(R1 R_hat^T) (camera-frame rad), (t1 - t_hat) / s].
Trained by Gaussian NLL. This is the unimodal comparison for the flow head.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from i2pflow.geometry import so3
from i2pflow.models.pose_net import PoseNet


class GaussianPoseHead(nn.Module):
    def __init__(self, cfg, space):
        super().__init__()
        self.cfg = cfg
        self.space = space
        D = space.dim
        self.n_off = D * (D - 1) // 2
        self.net = PoseNet(cfg.model, space, out_dim=6 + D + self.n_off)
        self.min_log_std = math.log(cfg.gaussian.min_std)

    def _predict(self, feats):
        ctx = self.net.encode_context(feats)
        B, dev = feats["points"].shape[0], feats["points"].device
        R0 = torch.eye(3, device=dev).expand(B, 1, 3, 3)
        t0 = torch.zeros(B, 1, 3, device=dev)
        out = self.net(ctx, R0, t0, torch.zeros(B, 1, device=dev))[:, 0]
        R_hat, t_hat, _ = self.space.endpoint(out[:, None, :6], R0, t0)
        D = self.space.dim
        log_diag = self.min_log_std + F.softplus(out[:, 6:6 + D] - self.min_log_std)
        L = torch.diag_embed(log_diag.exp())
        if self.n_off:
            rows, cols = torch.tril_indices(D, D, offset=-1, device=dev)
            L = L.clone()
            L[:, rows, cols] = out[:, 6 + D:]
        return R_hat[:, 0], t_hat[:, 0], L, log_diag

    def residual(self, R_hat, t_hat, R1, t1):
        rm, tm = self.space.masks(R1.device)
        r_rot = so3.log(R1 @ R_hat.transpose(-1, -2)) * rm
        r_t = (t1 - t_hat) / self.space.s * tm
        return torch.cat([r_rot, r_t], -1)[..., self.space.active.to(R1.device)]

    def _nll(self, R_hat, t_hat, L, log_diag, R1, t1):
        r = self.residual(R_hat, t_hat, R1, t1)
        z = torch.linalg.solve_triangular(L, r[..., None], upper=False)[..., 0]
        D = self.space.dim
        return 0.5 * (z * z).sum(-1) + log_diag.sum(-1) + 0.5 * D * math.log(2 * math.pi)

    def loss(self, feats, R1, t1):
        R_hat, t_hat, L, log_diag = self._predict(feats)
        nll = self._nll(R_hat, t_hat, L, log_diag, R1, t1)
        loss = nll.mean()
        with torch.no_grad():
            logs = {
                "loss": loss.detach(),
                "x1_rot_err_deg": torch.rad2deg(so3.geodesic_distance(R_hat, R1)).mean(),
                "x1_trans_err_m": (t_hat - t1).norm(dim=-1).mean(),
                "mean_log_std": log_diag.mean(),
            }
        return loss, logs

    @torch.no_grad()
    def predict(self, feats):
        R_hat, t_hat, L, _ = self._predict(feats)
        return R_hat, t_hat, L @ L.transpose(-1, -2)

    @torch.no_grad()
    def sample(self, feats, n, **_):
        R_hat, t_hat, L, _ = self._predict(feats)
        B, dev = R_hat.shape[0], R_hat.device
        z = torch.randn(B, n, self.space.dim, device=dev)
        r = (L[:, None] @ z[..., None])[..., 0]
        full = torch.zeros(B, n, 6, device=dev)
        full[..., self.space.active.to(dev)] = r
        R = so3.exp(full[..., :3]) @ R_hat[:, None]
        t = t_hat[:, None] + self.space.s * full[..., 3:]
        return R, t

    @torch.no_grad()
    def log_likelihood(self, feats, R1, t1, **_):
        """Gaussian log-density in rad/m units (tangent-space approximation)."""
        R_hat, t_hat, L, log_diag = self._predict(feats)
        nll = self._nll(R_hat, t_hat, L, log_diag, R1, t1)
        return -nll - self.space.trans_dim * math.log(self.space.s)
