import math

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from i2pflow.geometry import so3


@pytest.fixture(autouse=True)
def float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def test_exp_matches_scipy():
    w = torch.randn(100, 3) * 2
    R = so3.exp(w)
    ref = Rotation.from_rotvec(w.numpy()).as_matrix()
    assert np.allclose(R.numpy(), ref, atol=1e-10)


def test_log_exp_roundtrip_including_near_pi_and_zero():
    axis = torch.nn.functional.normalize(torch.randn(300, 3), dim=-1)
    ang = torch.cat([torch.rand(100) * 1e-6, torch.rand(100) * math.pi, math.pi - torch.rand(100) * 1e-3])
    w = axis * ang[:, None]
    w_back = so3.log(so3.exp(w))
    assert torch.allclose(so3.exp(w_back), so3.exp(w), atol=1e-6)
    ok = ang < math.pi - 1e-6
    assert torch.allclose(w_back[ok], w[ok], atol=1e-5)


def test_geodesic_interp_endpoints_and_midpoint():
    R0, R1 = so3.random_rotation((50,)), so3.random_rotation((50,))
    assert torch.allclose(so3.geodesic_interp(R0, R1, torch.zeros(50)), R0, atol=1e-8)
    assert torch.allclose(so3.geodesic_interp(R0, R1, torch.ones(50)), R1, atol=1e-6)
    Rm = so3.geodesic_interp(R0, R1, torch.full((50,), 0.5))
    d0, d1, d = so3.geodesic_distance(R0, Rm), so3.geodesic_distance(Rm, R1), so3.geodesic_distance(R0, R1)
    assert torch.allclose(d0, d / 2, atol=1e-6) and torch.allclose(d1, d / 2, atol=1e-6)


def test_yaw_roundtrip_and_axis():
    th = (torch.rand(100) * 2 - 1) * math.pi
    R = so3.yaw_matrix(th)
    assert torch.allclose(so3.yaw_angle(R), th, atol=1e-10)
    w = so3.log(R)
    assert torch.allclose(w[:, [0, 2]], torch.zeros(100, 2), atol=1e-8)


def test_left_jacobian_numerically():
    w, d = torch.randn(3), torch.randn(3) * 1e-6
    lhs = so3.exp(w + d)
    rhs = so3.exp(so3.left_jacobian(w) @ d) @ so3.exp(w)
    assert torch.allclose(lhs, rhs, atol=1e-11)


def test_pose_errors_identity_and_known():
    T = so3.to_matrix(so3.yaw_matrix(torch.tensor([0.3])), torch.tensor([[1.0, 0.0, 2.0]]))
    rre_e, rre_g, rte = so3.pose_errors(T, T)
    assert rre_e[0] < 1e-8 and rte[0] < 1e-8
    T2 = so3.to_matrix(so3.yaw_matrix(torch.tensor([0.3 + math.radians(4)])), torch.tensor([[1.0, 0.0, 2.0]]))
    _, rre_g, _ = so3.pose_errors(T2, T)
    assert abs(rre_g[0] - 4.0) < 1e-6
