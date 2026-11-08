import math

import pytest
import torch

from i2pflow.builders import build_head
from i2pflow.config import load_config
from i2pflow.flow.pose_space import PoseSpace
from i2pflow.flow.time_sampling import sample_tau
from i2pflow.geometry import so3


@pytest.mark.parametrize("dof", ["3dof", "6dof"])
def test_prior_respects_dof(dof):
    sp = PoseSpace(dof, 6.0)
    R, t = sp.sample_prior((2000,), "cpu")
    if dof == "3dof":
        assert torch.allclose(so3.log(R)[:, [0, 2]], torch.zeros(2000, 2), atol=1e-5)
        assert torch.all(t[:, 1] == 0)
    assert abs(t[:, 0].std().item() - 6.0) < 0.5


@pytest.mark.parametrize("dof", ["3dof", "6dof"])
def test_oracle_correction_reaches_target_and_velocity_matches(dof):
    sp = PoseSpace(dof, 6.0)
    R0, t0 = sp.sample_prior((64,), "cpu")
    R1, t1 = sp.sample_prior((64,), "cpu")
    tau = torch.rand(64) * 0.9
    R, t = sp.interpolate(R0, t0, R1, t1, tau)
    a = sp.target_correction(R, R1)
    b = (t1 - (so3.exp(a) @ t[..., None])[..., 0]) / sp.s
    out = torch.cat([a, b], -1)
    R1h, t1h, _ = sp.endpoint(out, R, t)
    assert torch.allclose(R1h, R1, atol=1e-4) and torch.allclose(t1h, t1, atol=1e-4)
    # implied velocity equals the conditional target velocity of the geodesic/linear path
    omega, v = sp.velocity(out, R, t, tau)
    assert torch.allclose(omega, so3.log(R0.transpose(-1, -2) @ R1), atol=1e-3)
    assert torch.allclose(v, t1 - t0, atol=1e-4)


def test_prior_log_prob_translation_normalised():
    sp = PoseSpace("3dof", 6.0)
    xs = torch.linspace(-60, 60, 601)
    X, Z = torch.meshgrid(xs, xs, indexing="ij")
    t = torch.stack([X, torch.zeros_like(X), Z], -1).reshape(-1, 3)
    R = torch.eye(3).expand(t.shape[0], 3, 3)
    p = (sp.prior_log_prob(R, t) + math.log(2 * math.pi)).exp()  # remove the yaw factor
    mass = p.sum() * (xs[1] - xs[0]) ** 2
    assert abs(mass.item() - 1.0) < 1e-3


def test_time_sampling_range():
    tau = sample_tau((10000,), "cpu")
    assert tau.min() >= 0.01 and tau.max() <= 0.99 and tau.mean() > 0.5


def fake_feats(B=2, N=1280):
    f = lambda *s: torch.nn.functional.normalize(torch.randn(*s), dim=1)
    K_c = torch.tensor([[44.2, 0, 31.4], [0, 44.2, 8.3], [0, 0, 1.0]])
    return {
        "points": torch.randn(B, N, 3) * 10, "pc_feat": torch.nn.functional.normalize(torch.randn(B, N, 128), dim=-1),
        "pc_score": torch.rand(B, N), "pc_fine_feat": torch.nn.functional.normalize(torch.randn(B, N, 64), dim=-1),
        "img_coarse": f(B, 128, 20, 64), "img_score": torch.rand(B, 1, 20, 64), "img_fine": f(B, 64, 80, 256),
        "K_coarse": K_c.expand(B, 3, 3), "K_fine": (K_c * torch.tensor([[4.0], [4.0], [1.0]])).expand(B, 3, 3),
    }


@pytest.mark.parametrize("method,dof", [("flow", "3dof"), ("flow", "6dof"), ("gaussian", "3dof")])
def test_heads_cpu_forward_backward_sample(method, dof):
    cfg = load_config("configs/debug.yaml", [f"method={method}", f"pose.dof={dof}"])
    sp = PoseSpace(dof, cfg.pose.trans_scale)
    head = build_head(cfg, sp)
    feats = fake_feats()
    R1, t1 = sp.sample_prior((2,), "cpu")
    loss, logs = head.loss(feats, R1, t1)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    R, t = head.sample(feats, 5)
    assert R.shape == (2, 5, 3, 3) and t.shape == (2, 5, 3)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3).expand_as(R), atol=1e-4)
    ll = head.log_likelihood(feats, R1, t1)
    assert ll.shape == (2,) and torch.isfinite(ll).all()


def test_untrained_flow_likelihood_equals_prior():
    """A fresh flow head outputs zero correction -> zero velocity -> density = prior density."""
    cfg = load_config("configs/debug.yaml")
    sp = PoseSpace("3dof", cfg.pose.trans_scale)
    head = build_head(cfg, sp)
    R1, t1 = sp.sample_prior((2,), "cpu")
    ll = head.log_likelihood(fake_feats(), R1, t1)
    assert torch.allclose(ll, sp.prior_log_prob(R1, t1), atol=1e-4)
