"""Pinhole projection and feature sampling at projected points."""
import torch
import torch.nn.functional as F


def transform_points(R, t, p):
    """R [..., 3, 3], t [..., 3], p [..., N, 3] -> [..., N, 3]."""
    return p @ R.transpose(-1, -2) + t[..., None, :]


def project(q, K, min_depth=0.1):
    """Camera-frame points q [..., N, 3] -> pixel coords uv [..., N, 2], depth [..., N], in_front mask.

    K follows CoFiI2P: integer pixel coordinates are pixel centres.
    """
    z = q[..., 2]
    in_front = z > min_depth
    zc = z.clamp_min(min_depth)
    uvw = q @ K.transpose(-1, -2)
    uv = uvw[..., :2] / zc[..., None]
    return uv, z, in_front


def sample_map(fmap, uv, valid):
    """Bilinearly sample fmap [B, C, H, W] at uv [B, S, N, 2] (pixel coords); zero where invalid.

    Returns features [B, S, N, C] and the in-image mask [B, S, N].
    """
    B, C, H, W = fmap.shape
    gx = 2.0 * uv[..., 0] / (W - 1) - 1.0
    gy = 2.0 * uv[..., 1] / (H - 1) - 1.0
    inside = valid & (gx.abs() <= 1.0) & (gy.abs() <= 1.0)
    grid = torch.stack([gx, gy], dim=-1).clamp(-2.0, 2.0)
    S, N = uv.shape[1], uv.shape[2]
    out = F.grid_sample(fmap, grid.reshape(B, S * N, 1, 2).to(fmap.dtype), mode="bilinear",
                        padding_mode="zeros", align_corners=True)  # [B, C, S*N, 1]
    out = out[..., 0].permute(0, 2, 1).reshape(B, S, N, C)
    return out * inside[..., None].to(out.dtype), inside
