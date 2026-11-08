"""SO(3) and pose utilities (batched over leading dims).

Pose convention used everywhere in i2pflow: T = (R, t) maps points of the (perturbed)
LiDAR cloud into the camera frame, p_cam = R @ p + t. This is the `P` key of CoFiI2P samples.
"""
import math

import torch

_EPS = 1e-6


def hat(w):
    """[..., 3] -> skew-symmetric [..., 3, 3]."""
    x, y, z = w.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([o, -z, y, z, o, -x, -y, x, o], dim=-1).reshape(w.shape[:-1] + (3, 3))


def vee(W):
    return torch.stack([W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]], dim=-1)


def exp(w):
    """Rotation vector [..., 3] -> rotation matrix [..., 3, 3] (Rodrigues, stable near 0)."""
    theta2 = (w * w).sum(-1, keepdim=True)
    theta = theta2.clamp_min(_EPS * _EPS).sqrt()
    small = theta2 < 1e-8
    a = torch.where(small, 1.0 - theta2 / 6.0, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24.0, (1.0 - torch.cos(theta)) / theta2.clamp_min(_EPS * _EPS))
    K = hat(w)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(K.shape)
    return eye + a[..., None] * K + b[..., None] * (K @ K)


def log(R):
    """Rotation matrix [..., 3, 3] -> rotation vector [..., 3], robust near 0 and pi."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
    theta = torch.acos(cos)
    v = vee(R - R.transpose(-1, -2)) / 2.0  # = sin(theta) * axis
    sin = torch.sin(theta)
    small = theta < 1e-4
    scale = torch.where(small, 1.0 + theta * theta / 6.0, theta / sin.clamp_min(_EPS))
    w = v * scale[..., None]

    # Near pi, sin -> 0 and v loses precision: recover the axis from the symmetric part,
    # (R + R^T) / 2 - cos(theta) I = (1 - cos(theta)) a a^T  (exact for any theta).
    near_pi = theta > math.pi - 1e-2
    if near_pi.any():
        Rn = R[near_pi]
        eye = torch.eye(3, dtype=R.dtype, device=R.device)
        B = (Rn + Rn.transpose(-1, -2)) / 2.0 - cos[near_pi][..., None, None] * eye
        diag = torch.diagonal(B, dim1=-2, dim2=-1)
        k = diag.argmax(-1)
        col = B[torch.arange(B.shape[0], device=R.device), :, k]
        axis = col / col.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        sign = torch.sign((axis * v[near_pi]).sum(-1, keepdim=True))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        w = w.clone()
        w[near_pi] = axis * sign * theta[near_pi][..., None]
    return w


def angle(R):
    """Rotation angle in radians of R [..., 3, 3]."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    return torch.acos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0))


def geodesic_distance(R1, R2):
    return angle(R1.transpose(-1, -2) @ R2)


def geodesic_interp(R0, R1, tau):
    """R0 Exp(tau Log(R0^T R1)); tau [...] broadcast against the batch dims."""
    w = log(R0.transpose(-1, -2) @ R1)
    return R0 @ exp(w * tau[..., None])


def yaw_matrix(theta):
    """Rotation about the camera y axis (the vertical axis in KITTI camera coordinates)."""
    c, s = torch.cos(theta), torch.sin(theta)
    o, i = torch.zeros_like(theta), torch.ones_like(theta)
    return torch.stack([c, o, s, o, i, o, -s, o, c], dim=-1).reshape(theta.shape + (3, 3))


def yaw_angle(R):
    """Inverse of yaw_matrix for pure y-rotations."""
    return torch.atan2(R[..., 0, 2], R[..., 0, 0])


def random_rotation(shape, device=None, dtype=None, generator=None):
    """Haar-uniform rotations via normalized Gaussian quaternions."""
    q = torch.randn(tuple(shape) + (4,), device=device, dtype=dtype or torch.get_default_dtype(), generator=generator)
    q = q / q.norm(dim=-1, keepdim=True)
    return quat_to_matrix(q)


def quat_to_matrix(q):
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return R.reshape(q.shape[:-1] + (3, 3))


def project_to_so3(M):
    """Closest rotation (Frobenius) to M [..., 3, 3]; used for chordal means."""
    U, _, Vh = torch.linalg.svd(M)
    d = torch.det(U @ Vh)
    D = torch.ones_like(M[..., 0])
    D[..., 2] = d
    return U @ torch.diag_embed(D) @ Vh


def left_jacobian(w):
    """SO(3) left Jacobian J_l(w): Exp(w + d) ~= Exp(J_l(w) d) Exp(w)."""
    theta2 = (w * w).sum(-1, keepdim=True)
    theta = theta2.clamp_min(_EPS * _EPS).sqrt()
    small = theta2 < 1e-8
    a = torch.where(small, 0.5 - theta2 / 24.0, (1 - torch.cos(theta)) / theta2.clamp_min(_EPS * _EPS))
    b = torch.where(small, 1.0 / 6.0 - theta2 / 120.0, (theta - torch.sin(theta)) / (theta2 * theta).clamp_min(_EPS ** 3))
    K = hat(w)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(K.shape)
    return eye + a[..., None] * K + b[..., None] * (K @ K)


# ---- poses as 4x4 matrices -------------------------------------------------

def to_matrix(R, t):
    T = torch.zeros(R.shape[:-2] + (4, 4), dtype=R.dtype, device=R.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def from_matrix(T):
    return T[..., :3, :3], T[..., :3, 3]


def pose_errors(T_pred, T_gt):
    """Return (rre_euler_deg, rre_geodesic_deg, rte_m) using inv(pred) @ gt, as in CoFiI2P/VP2P.

    rre_euler is the sum of |Euler angles| ('xzy'), the convention of the I2P literature.
    Accepts numpy arrays or tensors [..., 4, 4]; returns numpy arrays.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    T_pred = np.asarray(T_pred.detach().cpu() if torch.is_tensor(T_pred) else T_pred, dtype=np.float64)
    T_gt = np.asarray(T_gt.detach().cpu() if torch.is_tensor(T_gt) else T_gt, dtype=np.float64)
    shape = T_pred.shape[:-2]
    D = np.linalg.inv(T_pred.reshape(-1, 4, 4)) @ T_gt.reshape(-1, 4, 4)
    rot = Rotation.from_matrix(D[:, :3, :3])
    rre_euler = np.abs(rot.as_euler("xzy", degrees=True)).sum(-1)
    rre_geo = np.degrees(np.linalg.norm(rot.as_rotvec(), axis=-1))
    rte = np.linalg.norm(D[:, :3, 3], axis=-1)
    return rre_euler.reshape(shape), rre_geo.reshape(shape), rte.reshape(shape)
