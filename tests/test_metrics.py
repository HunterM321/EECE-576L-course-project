import math

import numpy as np
import torch

from i2pflow.eval import metrics as M
from i2pflow.eval.modes import extract_modes
from i2pflow.flow.pose_space import PoseSpace
from i2pflow.geometry import so3


def test_auroc_perfect_random_and_ties():
    y = np.array([0, 0, 1, 1], bool)
    assert M.auroc([0.1, 0.2, 0.8, 0.9], y) == 1.0
    assert M.auroc([0.9, 0.8, 0.2, 0.1], y) == 0.0
    assert M.auroc([0.5, 0.5, 0.5, 0.5], y) == 0.5


def test_accuracy_thresholds():
    m = M.accuracy_metrics([1.0, 4.9, 6.0, 1.0], [0.1, 1.9, 0.1, 3.0])
    assert m["acc"] == 50.0 and m["rr"] == 100.0


def test_two_modes_found_with_weights():
    th = torch.cat([torch.full((70,), 0.0), torch.full((30,), math.pi / 2)]) + torch.randn(100) * 0.005
    R = so3.yaw_matrix(th)
    t = torch.zeros(100, 3)
    t[70:, 0] = 10.0
    modes = extract_modes(R, t, 5.0, 1.5)
    assert len(modes) == 2
    assert abs(modes[0]["weight"] - 0.7) < 1e-6 and abs(modes[1]["weight"] - 0.3) < 1e-6
    assert abs(modes[1]["t"][0].item() - 10.0) < 1e-4


def test_kde_level_calibrated_for_matching_gaussian():
    """GT and samples drawn from the same distribution -> HPD levels ~ uniform."""
    torch.manual_seed(0)
    sp = PoseSpace("3dof", 6.0)
    B, N = 400, 256
    std = torch.tensor([math.radians(3.0), 0.5, 0.5])

    def draw(shape):
        z = torch.randn(*shape, 3) * std
        return so3.yaw_matrix(z[..., 0]), torch.stack([z[..., 1], torch.zeros_like(z[..., 1]), z[..., 2]], -1)

    R_s, t_s = draw((B, N))
    R_g, t_g = draw((B,))
    _, level = M.kde_scores(R_s, t_s, R_g, t_g, sp, bw_rot_deg=1.0, bw_trans=0.15)
    _, _, err = M.coverage_curve(level.numpy())
    assert err < 0.06
