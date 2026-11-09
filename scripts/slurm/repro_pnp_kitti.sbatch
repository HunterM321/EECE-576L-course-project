#!/bin/bash -l
#SBATCH --account=aip-rjliao
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --job-name=repro_pnp_kitti
#SBATCH --output=logs/%x-%j.out
# GPU pipeline tests + full CoFiI2P+PnP-RANSAC evaluation on KITTI test (seqs 09-10, 5584 cases).
# Submit from the repo root: sbatch scripts/slurm/repro_pnp_kitti.sbatch
set -uo pipefail
source scripts/slurm/env.sh
nvidia-smi --query-gpu=name --format=csv,noheader
python -c "
import torch
from i2pflow.config import load_config
from i2pflow.data.datasets import build_options
from i2pflow.models.backbone import load_cofii2p
for ds in ('kitti', 'nuscenes'):
    cfg = load_config('configs/kitti_flow.yaml' if ds == 'kitti' else 'configs/nuscenes_flow.yaml')
    net = load_cofii2p(build_options(cfg), cfg.backbone.ckpt, 'cuda')   # strict=True
    print(ds, 'checkpoint loads strictly:', sum(p.numel() for p in net.parameters()) / 1e6, 'M params')
"
python -m pytest -q -p no:cacheprovider tests/test_gpu_pipeline.py
python scripts/evaluate.py --method pnp --config configs/kitti_flow.yaml --split test \
  --tag pnp_test_alliance data.num_workers=$SLURM_CPUS_PER_TASK
