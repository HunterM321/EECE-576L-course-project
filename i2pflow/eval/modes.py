"""Mode extraction from pose samples (density-based greedy clustering, DBSCAN-like).

Distance between poses: d = sqrt((rot_angle / eps_rot)^2 + (|dt| / eps_trans)^2).
Repeatedly take the sample with the most neighbours (d < 1), make its neighbourhood a mode,
remove it, and continue. Mode weight = members / N; representative = chordal mean rotation
and mean translation of the members.
"""
import math

import torch

from i2pflow.geometry import so3


def pairwise_distance(R, t, eps_rot_deg, eps_trans):
    """R [N,3,3], t [N,3] -> [N,N] normalised pose distance."""
    rel = R[:, None].transpose(-1, -2) @ R[None]
    rot = so3.angle(rel) / math.radians(eps_rot_deg)
    tr = torch.cdist(t[None], t[None])[0] / eps_trans
    return torch.sqrt(rot ** 2 + tr ** 2)


def extract_modes(R, t, eps_rot_deg=5.0, eps_trans=1.5, max_modes=5, min_weight=0.0):
    """Single frame. Returns list of dicts {R, t, weight, size} sorted by weight."""
    N = R.shape[0]
    D = pairwise_distance(R, t, eps_rot_deg, eps_trans)
    adj = D < 1.0
    remaining = torch.ones(N, dtype=torch.bool, device=R.device)
    modes = []
    while remaining.any() and len(modes) < max_modes:
        counts = (adj & remaining[None] & remaining[:, None]).sum(-1).masked_fill(~remaining, -1)
        c = int(counts.argmax())
        members = adj[c] & remaining
        w = members.sum().item() / N
        if w < min_weight:
            break
        Rm = so3.project_to_so3(R[members].sum(0))
        modes.append({"R": Rm, "t": t[members].mean(0), "weight": w, "size": int(members.sum())})
        remaining &= ~members
    return modes
