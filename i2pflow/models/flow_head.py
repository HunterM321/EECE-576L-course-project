"""Riemannian flow-matching pose head (I2P-Flow).

Training (FrameFlow-style x1-prediction): tau ~ p(tau), (R0, t0) ~ prior, (R_tau, t_tau) on the
geodesic/linear path to the ground truth, the network predicts the endpoint as a camera-frame
correction (a, b), and the loss compares it with the true correction, weighted like a velocity
loss: w(tau) = 1 / (1 - min(tau, clip))^2  (or 1 for a plain endpoint loss).

Sampling: Euler integration of the implied body-frame velocity from N prior samples, the last
step jumping to the predicted endpoint. `rot_anneal` > 0 replaces the rotation step 1/(1-tau)
with a constant rate (FrameFlow inference annealing): more accurate, but no longer a calibrated
density, so never use it for NLL/coverage.

Likelihood: exact log-density of a given pose (e.g. the ground truth) by integrating the ODE
backward with the exact divergence in tangent coordinates (3 or 6 vector-Jacobian products/step).
"""
import torch
import torch.nn as nn

from i2pflow.flow.time_sampling import sample_tau
from i2pflow.geometry import so3
from i2pflow.models.pose_net import PoseNet


class FlowPoseHead(nn.Module):
    def __init__(self, cfg, space):
        super().__init__()
        self.cfg = cfg
        self.space = space
        self.net = PoseNet(cfg.model, space, out_dim=6)

    # ---- training -----------------------------------------------------------

    def loss(self, feats, R1, t1):
        fc = self.cfg.flow
        B, dev = R1.shape[0], R1.device
        S = fc.samples_per_frame
        ctx = self.net.encode_context(feats)
        tau = sample_tau((B, S), dev, **fc.time_sampling)
        R0, t0 = self.space.sample_prior((B, S), dev)
        R1s, t1s = R1[:, None].expand(B, S, 3, 3), t1[:, None].expand(B, S, 3)
        R_tau, t_tau = self.space.interpolate(R0, t0, R1s, t1s, tau)

        sc = None
        if self.net.self_cond and torch.rand(()) < fc.self_cond_prob:
            with torch.no_grad():
                Rp, tp, _ = self.space.endpoint(self.net(ctx, R_tau, t_tau, tau), R_tau, t_tau)
            sc = (Rp.detach(), tp.detach())

        out = self.net(ctx, R_tau, t_tau, tau, sc)
        R1_hat, t1_hat, a = self.space.endpoint(out, R_tau, t_tau)
        a_star = self.space.target_correction(R_tau, R1s)
        rot_sq = ((a - a_star) ** 2).sum(-1)
        tr_sq = (((t1_hat - t1s) / self.space.s) ** 2).sum(-1)
        if fc.loss_weighting == "velocity":
            w = 1.0 / (1.0 - tau.clamp(max=fc.weight_clip)) ** 2
        elif fc.loss_weighting == "endpoint":
            w = torch.ones_like(tau)
        else:
            raise ValueError(fc.loss_weighting)
        loss = (w * (fc.lambda_rot * rot_sq + fc.lambda_trans * tr_sq)).mean()

        with torch.no_grad():
            rot_err = torch.rad2deg(so3.geodesic_distance(R1_hat, R1s))
            tr_err = (t1_hat - t1s).norm(dim=-1)
            late = tau > 0.8
            logs = {
                "loss": loss.detach(),
                "loss_rot": (w * rot_sq).mean(),
                "loss_trans": (w * tr_sq).mean(),
                "x1_rot_err_deg": rot_err.mean(),
                "x1_trans_err_m": tr_err.mean(),
                "x1_rot_err_deg_late": rot_err[late].mean() if late.any() else rot_err.new_tensor(float("nan")),
                "x1_trans_err_m_late": tr_err[late].mean() if late.any() else tr_err.new_tensor(float("nan")),
            }
        return loss, logs

    # ---- inference ----------------------------------------------------------

    @torch.no_grad()
    def sample(self, feats, n, steps=None, rot_anneal=None, chunk=None):
        ic = self.cfg.inference
        steps = steps or ic.steps
        rot_anneal = ic.rot_anneal if rot_anneal is None else rot_anneal
        chunk = chunk or ic.chunk
        ctx = self.net.encode_context(feats)
        Rs, ts = [], []
        for start in range(0, n, chunk):
            m = min(chunk, n - start)
            R, t = self._integrate(ctx, feats["points"].shape[0], m, steps, rot_anneal, ic.t_min)
            Rs.append(R)
            ts.append(t)
        return torch.cat(Rs, 1), torch.cat(ts, 1)

    def _integrate(self, ctx, B, m, steps, rot_anneal, t_min):
        dev = ctx["points"].device
        R, t = self.space.sample_prior((B, m), dev)
        grid = torch.linspace(t_min, 1.0, steps + 1, device=dev)
        sc = None
        for i in range(steps):
            tau = grid[i].expand(B, m)
            out = self.net(ctx, R, t, tau, sc if self.net.self_cond else None)
            R1_hat, t1_hat, a = self.space.endpoint(out, R, t)
            if i == steps - 1:
                R, t = R1_hat, t1_hat
                break
            h = grid[i + 1] - grid[i]
            f_t = h / (1.0 - grid[i])
            f_r = torch.clamp(rot_anneal * h, max=1.0) if rot_anneal and rot_anneal > 0 else f_t
            R = so3.exp(f_r * a) @ R
            t = t + f_t * (t1_hat - t)
            sc = (R1_hat, t1_hat)
        return R, t

    def log_likelihood(self, feats, R1, t1, steps=None, tau_end=None):
        """Exact log p(R1, t1) [B] (units: per rad^k m^j) via the reverse ODE and exact divergence."""
        ic = self.cfg.inference
        steps = steps or ic.nll_steps
        tau_end = tau_end or ic.nll_tau_end
        space = self.space
        was_training = self.training
        self.eval()
        params = [p for p in self.parameters() if p.requires_grad]
        for p in params:
            p.requires_grad_(False)
        amp = self.net.amp_dtype
        self.net.amp_dtype = None  # divergence via autograd: keep the network in fp32
        try:
            with torch.no_grad():
                ctx = self.net.encode_context(feats)
            B, dev = R1.shape[0], R1.device
            R, t = R1[:, None].clone(), t1[:, None].clone()
            grid = torch.linspace(0.0, tau_end, steps + 1, device=dev)
            div_int = torch.zeros(B, device=dev)
            for i in range(steps, 0, -1):
                tau = grid[i].expand(B, 1)
                h = grid[i] - grid[i - 1]
                with torch.enable_grad():
                    eps = torch.zeros(B, 1, space.dim, device=dev, requires_grad=True)
                    Rp, tp = space.perturb(R, t, eps)
                    out = self.net(ctx, Rp, tp, tau)
                    omega, v = space.velocity(out, Rp, tp, tau)
                    vel = space.active_coords(omega, v)  # [B, 1, dim]
                    div = torch.zeros(B, device=dev)
                    for d in range(space.dim):
                        g, = torch.autograd.grad(vel[..., d].sum(), eps, retain_graph=d < space.dim - 1)
                        div = div + g[:, 0, d]
                omega, v = omega.detach(), v.detach()
                R = R @ so3.exp(-h * omega)
                t = t - h * v
                div_int = div_int + h * div.detach()
            return space.prior_log_prob(R[:, 0], t[:, 0]) - div_int
        finally:
            self.net.amp_dtype = amp
            for p in params:
                p.requires_grad_(True)
            self.train(was_training)
