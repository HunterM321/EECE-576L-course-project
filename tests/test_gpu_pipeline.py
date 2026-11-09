"""End-to-end checks on real KITTI frames (requires a GPU, the data and checkpoints). No training."""
import numpy as np
import pytest
import torch

from i2pflow.builders import build_backbone, build_head, build_space
from i2pflow.config import load_config
from i2pflow.data.datasets import build_dataset, build_options, collate_list, gt_pose, to_device
from i2pflow.engine.evaluator import evaluate_head, evaluate_pnp, summarize
from i2pflow.geometry import so3
from i2pflow.paths import PROJECT_ROOT

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
# Per-frame dumps of upstream eval_all.py (gitignored; regenerate with scripts/slurm/upstream_eval.sbatch kitti).
_UPSTREAM_DIRS = sorted(p for p in (PROJECT_ROOT / "results/baselines/cofii2p_pnp/kitti").glob("upstream_eval_*")
                        if p.is_dir())
UPSTREAM = _UPSTREAM_DIRS[-1] if _UPSTREAM_DIRS else None


@pytest.fixture(scope="module")
def setup():
    cfg = load_config("configs/kitti_flow.yaml", ["data.num_workers=0"])
    dev = torch.device("cuda")
    opt = build_options(cfg)
    test = build_dataset(cfg, "test")
    samples = [to_device(test[i], dev) for i in range(2)]
    backbone = build_backbone(cfg, opt, dev)
    return cfg, dev, opt, test, samples, backbone


def test_test_split_matches_upstream_order(setup):
    cfg, _, opt, test, _, _ = setup
    from data.kitti import kitti_pc_img_dataset  # CoFiI2P (on sys.path via i2pflow)
    upstream = kitti_pc_img_dataset(opt, "val", is_front=False)
    assert test.dataset == upstream.dataset  # same order -> same per-index seeded perturbations


def test_train_val_split_disjoint(setup):
    cfg = setup[0]
    tr, va = build_dataset(cfg, "train"), build_dataset(cfg, "val")
    assert not set(tr.dataset) & set(va.dataset)
    assert len(va.dataset) > 0 and len(tr.dataset) > 30000


def test_backbone_feature_shapes(setup):
    _, _, _, _, samples, backbone = setup
    f = backbone(samples)
    assert f["points"].shape == (2, 1280, 3) and f["pc_feat"].shape == (2, 1280, 128)
    assert f["pc_fine_feat"].shape == (2, 1280, 64) and f["img_coarse"].shape == (2, 128, 20, 64)
    assert f["img_fine"].shape == (2, 64, 80, 256) and f["K_coarse"].shape == (2, 3, 3)


def test_gt_projects_superpoints_into_image(setup):
    """Sanity of the pose convention: under the GT, a good share of high-score points lands in the image."""
    _, dev, _, _, samples, backbone = setup
    f = backbone(samples)
    R, t = gt_pose(samples, dev)
    q = f["points"] @ R.transpose(-1, -2) + t[:, None]
    uv = q @ f["K_coarse"].transpose(-1, -2)
    uv = uv[..., :2] / q[..., 2:].clamp_min(0.1)
    inside = (q[..., 2] > 0) & (uv[..., 0] >= 0) & (uv[..., 0] <= 63) & (uv[..., 1] >= 0) & (uv[..., 1] <= 19)
    hi = f["pc_score"] > 0.9
    assert (inside & hi).sum() / hi.sum().clamp_min(1) > 0.8


@pytest.mark.parametrize("method", ["flow", "gaussian"])
def test_head_loss_backward_sample_likelihood(setup, method):
    cfg, dev, _, _, samples, backbone = setup
    cfg = load_config(f"configs/kitti_{method}.yaml", ["data.num_workers=0"])
    space = build_space(cfg)
    head = build_head(cfg, space).to(dev)
    feats = backbone(samples)
    R1, t1 = gt_pose(samples, dev)
    torch.cuda.reset_peak_memory_stats()
    loss, logs = head.loss(feats, R1, t1)
    loss.backward()  # gradients only; no optimizer step
    assert torch.isfinite(loss)
    print(f"\n{method}: loss {loss.item():.3f}, peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB "
          f"for {len(samples)} frames x {cfg.flow.samples_per_frame} hypotheses")
    R, t = head.sample(feats, 16)
    assert torch.isfinite(R).all() and torch.isfinite(t).all()
    ll = head.log_likelihood(feats, R1, t1)
    assert torch.isfinite(ll).all()


def test_pnp_baseline_reproduces_upstream_frame(setup):
    if UPSTREAM is None:
        pytest.skip("no upstream eval dumps; run sbatch scripts/slurm/upstream_eval.sbatch kitti")
    cfg, dev, opt, _, samples, _ = setup
    from i2pflow.baselines.pnp_ransac import CoFiI2PPnP
    pnp = CoFiI2PPnP(opt, cfg.backbone.ckpt, dev)
    out = pnp(samples[0])
    up = np.load(UPSTREAM / "000000.npy", allow_pickle=True).item()
    assert np.allclose(np.asarray(up["GT_P"]), samples[0]["P"].cpu().numpy(), atol=1e-5)
    _, rre_ours, rte_ours = so3.pose_errors(out["T"], samples[0]["P"].cpu().numpy())
    _, rre_up, rte_up = so3.pose_errors(np.asarray(up["pred_P"]), np.asarray(up["GT_P"]))
    # RANSAC is randomised: same frame, errors of the same magnitude
    assert out["success"] and abs(rre_ours - rre_up) < 1.0 and abs(rte_ours - rte_up) < 0.3
    assert np.all(np.linalg.eigvalsh(out["cov"]) > 0)


def test_evaluators_run(setup):
    cfg, dev, opt, test, _, backbone = setup
    cfg = load_config("configs/debug.yaml", ["data.num_workers=0"])
    space = build_space(cfg)
    head = build_head(cfg, space).to(dev)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(test, range(4)), batch_size=2,
                                         collate_fn=collate_list)
    res = evaluate_head(head, backbone, loader, space, cfg.eval, dev, log_every=0)
    assert res["T_est"].shape == (4, 4, 4)
    s = summarize(res, space)
    assert "acc" in s and "kde_nll" in s
    from i2pflow.baselines.pnp_ransac import CoFiI2PPnP
    res = evaluate_pnp(CoFiI2PPnP(opt, cfg.backbone.ckpt, dev), loader, space, cfg.eval, dev, log_every=0)
    s = summarize(res, space)
    assert s["acc"] >= 50.0  # PnP is ~99.5% on the full split
