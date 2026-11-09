#!/bin/bash -l
#SBATCH --account=aip-rjliao
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --job-name=eval_pnp
#SBATCH --output=logs/%x-%j.out
# CoFiI2P + PnP-RANSAC through the I2P-Flow evaluator on the test split. Usage (from the repo root):
#   sbatch scripts/slurm/eval_pnp.sbatch configs/kitti_flow.yaml|configs/nuscenes_flow.yaml [tag]
set -euo pipefail
source scripts/slurm/env.sh
python scripts/evaluate.py --method pnp --config "$1" --split test --tag "${2:-pnp_test_alliance}" \
  data.num_workers=$SLURM_CPUS_PER_TASK
