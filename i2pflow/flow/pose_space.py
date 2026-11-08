"""The pose manifold the flow lives on: S^1 x R^2 ("3dof": yaw + ground-plane x,z) or SO(3) x R^3 ("6dof").

Both are represented with full (R [...,3,3], t [...,3]) tensors; the 3dof space masks the
inactive rotation axes and translation components. Time convention: tau=0 prior, tau=1 data.

Network outputs are an endpoint correction expressed in the current camera frame:
    R1_hat = Exp(a) @ R_tau,    t1_hat = Exp(a) @ t_tau + s * b
so the implied body-frame velocity is  omega = R_tau^T a / (1 - tau),  v = (t1_hat - t_tau) / (1 - tau).
"""
import math

import torch

from i2pflow.geometry import so3


class PoseSpace:
    def __init__(self, dof="3dof", trans_scale=6.0):
        if dof not in ("3dof", "6dof"):
            raise ValueError(dof)
        self.dof = dof
        self.s = float(trans_scale)
        if dof == "3dof":
            self.rot_mask = torch.tensor([0.0, 1.0, 0.0])
            self.trans_mask = torch.tensor([1.0, 0.0, 1.0])
        else:
            self.rot_mask = torch.ones(3)
            self.trans_mask = torch.ones(3)
        # indices of active coordinates in the 6-vector [rot(3), trans(3)]
        self.active = torch.cat([self.rot_mask, self.trans_mask]).nonzero().flatten()
        self.dim = int(self.active.numel())
        self.rot_dim = int(self.rot_mask.sum())
        self.trans_dim = int(self.trans_mask.sum())

    def masks(self, device, dtype=torch.float32):
        return self.rot_mask.to(device, dtype), self.trans_mask.to(device, dtype)

    # ---- prior --------------------------------------------------------------

    def sample_prior(self, shape, device, dtype=torch.float32):
        shape = tuple(shape)
        if self.dof == "3dof":
            R = so3.yaw_matrix((torch.rand(shape, device=device, dtype=dtype) * 2 - 1) * math.pi)
        else:
            R = so3.random_rotation(shape, device=device, dtype=dtype)
        _, tm = self.masks(device, dtype)
        t = torch.randn(shape + (3,), device=device, dtype=dtype) * self.s * tm
        return R, t

    def prior_log_prob(self, R, t):
        """Log density in units of rad^-k m^-j (uniform rotation x isotropic Gaussian translation)."""
        log_rot = -math.log(2 * math.pi) if self.dof == "3dof" else -math.log(8 * math.pi ** 2)
        _, tm = self.masks(t.device, t.dtype)
        z = t / self.s * tm
        log_t = -0.5 * (z * z).sum(-1) - self.trans_dim * (0.5 * math.log(2 * math.pi) + math.log(self.s))
        return log_rot + log_t

    # ---- path ---------------------------------------------------------------

    def interpolate(self, R0, t0, R1, t1, tau):
        R = so3.geodesic_interp(R0, R1, tau)
        t = (1 - tau)[..., None] * t0 + tau[..., None] * t1
        return R, t

    def target_correction(self, R_tau, R1):
        """a* = Log(R1 R_tau^T): the camera-frame rotation still to go (masked)."""
        rm, _ = self.masks(R1.device, R1.dtype)
        return so3.log(R1 @ R_tau.transpose(-1, -2)) * rm

    # ---- network output <-> pose ----------------------------------------------

    def split_output(self, out):
        rm, tm = self.masks(out.device, out.dtype)
        return out[..., :3] * rm, out[..., 3:6] * tm

    def endpoint(self, out, R_tau, t_tau):
        a, b = self.split_output(out)
        dR = so3.exp(a)
        R1 = dR @ R_tau
        t1 = (dR @ t_tau[..., None])[..., 0] + self.s * b
        return R1, t1, a

    def velocity(self, out, R_tau, t_tau, tau):
        """Body-frame angular velocity and translational velocity (m / unit time)."""
        R1, t1, a = self.endpoint(out, R_tau, t_tau)
        inv = 1.0 / (1.0 - tau).clamp_min(1e-4)
        omega = (R_tau.transpose(-1, -2) @ a[..., None])[..., 0] * inv[..., None]
        v = (t1 - t_tau) * inv[..., None]
        return omega, v

    # ---- 6-vector helpers ---------------------------------------------------------

    def perturb(self, R, t, eps):
        """Apply a tangent perturbation eps [..., dim] (body rotation, metres) on active coords."""
        full = torch.zeros(eps.shape[:-1] + (6,), dtype=eps.dtype, device=eps.device)
        full[..., self.active.to(eps.device)] = eps
        return R @ so3.exp(full[..., :3]), t + full[..., 3:]

    def active_coords(self, omega, v):
        return torch.cat([omega, v], dim=-1)[..., self.active.to(omega.device)]
