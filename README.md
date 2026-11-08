# I2P-Flow (EECE 576L project)

Flow-matching pose head for image-to-point-cloud registration on a frozen CoFiI2P backbone,
with the CoFiI2P + PnP-RANSAC baseline kept in the same evaluation pipeline.

## Layout

```
i2pflow/                 our package
  geometry/  so3.py (Exp/Log robust at 0 and pi, geodesics, pose errors), camera.py (projection, map sampling)
  flow/      pose_space.py (3dof S^1xR^2 / 6dof SO(3)xR^3: priors, paths, endpoint<->velocity), time_sampling.py
  models/    backbone.py (frozen CoFiI2P features), pose_net.py (render-and-compare transformer),
             flow_head.py (RFM head: loss / sampler / exact NLL), gaussian_head.py (matched unimodal baseline)
  baselines/ pnp_ransac.py (CoFiI2P + PnP-RANSAC, inlier ratio + Jacobian covariance)
  data/      datasets.py (CoFiI2P loaders, configurable sequences, held-out val split)
  eval/      metrics.py (Acc/RR/RRE/RTE, KDE NLL, HPD coverage, AUROC), modes.py (mode extraction)
  engine/    trainer.py, evaluator.py
configs/                 default.yaml + experiments (kitti_flow, kitti_gaussian, kitti_flow_6dof, kitti_flow_highway, nuscenes_flow, debug)
scripts/                 train.py, evaluate.py, cofii2p_kitti_metrics.py
tests/                   CPU unit tests + GPU pipeline tests (no training)
results/baselines/       CoFiI2P+PnP results (upstream KITTI eval: Acc 99.53%, RRE 1.24°, RTE 0.31 m)
third_party/CoFiI2P      upstream baseline (only change: eval_all.py respects CUDA_VISIBLE_DEVICES)
third_party/GeoTransformer  provides the compiled `geotransformer.ext` CoFiI2P needs
data/, checkpoints/      symlinks into /data/masibo/cofii2p
runs/                    training outputs
```

Environment: `conda activate /data/conda_envs/cofii2p` (Python 3.8, PyTorch 1.13.1+cu116).

## Setup from a fresh clone

`third_party/`, `data/` and `checkpoints/` are not tracked.

1. **Third-party code**, pinned to the commits used here:
   ```bash
   git clone https://github.com/WHU-USI3DV/CoFiI2P third_party/CoFiI2P
   git -C third_party/CoFiI2P checkout ed90edf51cca540ee76152d91133c7e79b63256c
   git -C third_party/CoFiI2P apply ../../patches/cofii2p_eval_all_gpu.patch
   git clone https://github.com/qinzheng93/GeoTransformer third_party/GeoTransformer
   git -C third_party/GeoTransformer checkout e7a135af4c318ff3b8d7f6c963df094d7e4ea540
   ```
2. **Environment**:
   - Python 3.8, with `torch==1.13.1+cu116` and `torchvision==0.14.1+cu116`.
   - Packages: `open3d==0.17.0`, `opencv-python<5`, `fvcore`, `einops`, `scipy`, `pyyaml`, `matplotlib`, `pyquaternion`, `nuscenes-devkit`, `tensorboard<2.15`, `pytest`.
   - A CUDA 11.x `nvcc` to build the extension: `cd third_party/GeoTransformer && CUDA_HOME=<cuda11.x> pip install --no-build-isolation -e .`
3. **Data and checkpoints**:
   - `data/kitti` must point to CoFiI2P's preprocessed KITTI (layout `calib/`, `sequences/NN/{img_P2,img_P3,K_P2,K_P3,pc_npy_with_normal}`).
   - Put `cofii2p_kitti.t7` (and `cofii2p_nuscenes.t7`) under `checkpoints/cofii2p/`. Download links are in the CoFiI2P README.

## Method (defaults in `configs/default.yaml`)

- **Space**: `pose.dof=3dof`, yaw + ground-plane (x, z), which matches the benchmark perturbation. `6dof` is the ablation.
  Pose convention: `p_cam = R p + t` (CoFiI2P's `sample['P']`). Time: τ=0 prior, τ=1 data.
- **Path / prior**: geodesic rotation × linear translation. Yaw ~ U(−π, π) (Haar on SO(3) for 6dof). Translation ~ N(0, s²), s = 6 m.
- **Prediction**: the network outputs an endpoint correction in the current camera frame,
  `R̂1 = Exp(a) R_τ`, `t̂1 = Exp(a) t_τ + s·b`. This gives the velocity ω = R_τᵀa/(1−τ), v = (t̂1−t_τ)/(1−τ).
- **Loss**: FrameFlow-style velocity weighting 1/(1−min(τ, 0.9))² on ‖a−a*‖² + ‖(t̂1−t1)/s‖².
  τ ~ 0.2·U + 0.8·Beta(1.9, 1). Self-conditioning is on with p = 0.5.
- **Network** (render-and-compare):
  - Inputs:
    - the 512 CoFiI2P superpoints, each moved by the current pose and projected into the stride-8 fused and stride-2 fine feature maps;
    - the sampled features, the point/image cosine similarities, depth and visibility;
    - a [POSE] token.
  - Architecture: 6 DiT blocks with adaLN time conditioning. The first 2 blocks also cross-attend to the pooled image tokens.
- **Inference**:
  - Sampling: 64 starts, 10 Euler steps, last step jumps to x̂1, then mode extraction; the top mode is the estimate.
  - `inference.rot_anneal=10` gives the annealed (accuracy) sampler. Don't use it for NLL or coverage.
  - Exact GT log-likelihood: reverse ODE with exact divergence (`eval.exact_nll=true`).
- **Baselines** (all scored by the same evaluator):
  - CoFiI2P+PnP-RANSAC (`--method pnp`);
  - the matched Gaussian head (`configs/kitti_gaussian.yaml`).

## Usage

```bash
# tests (no training)
python -m pytest -q tests/                                  # GPU tests need a visible GPU
# train (one GPU; ~0.58 s/step, 18.9 GB peak at defaults)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/kitti_flow.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/kitti_gaussian.yaml
python scripts/train.py --config configs/kitti_flow.yaml --resume runs/<run>/last.pt
# evaluate
python scripts/evaluate.py --ckpt runs/<run>/best.pt eval.exact_nll=true
python scripts/evaluate.py --ckpt runs/<run>/best.pt inference.rot_anneal=10 --tag best_anneal
python scripts/evaluate.py --method pnp --config configs/kitti_flow.yaml
```

Any config key can be overridden as `key.sub=value`. Runs write `config.yaml`, `tb/` (TensorBoard),
`val_log.jsonl`, `best.pt` (by validation Acc) and `last.pt`.

## Notes / known limitations

- **Validation split**: the last 5% of each training sequence. The test split (09–10) is never used for model selection.
- **Fixed training perturbations**: CoFiI2P seeds each sample's perturbation and crop by its index, so the training perturbations are the same every epoch. Diversity comes from the head's own prior and τ draws.
- **Throughput**: the frozen backbone is ~60% of step time. Caching its features (~1 MB/frame without the fine map, ~4 MB with it) is the next speed-up if needed.
- **Not implemented yet**:
  - the "twin places" synthetic benchmark (`eval.metrics.mode_recall` is ready for it);
  - nuScenes data (not downloaded).
