#!/bin/bash -l
#SBATCH --account=aip-rjliao
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=train
#SBATCH --output=logs/%x-%j.out
# Train a pose head, logging to wandb (API key from ~/.netrc; compute nodes have internet).
# Submit from the repo root:
#   sbatch scripts/slurm/train.sbatch configs/kitti_flow.yaml [--name NAME] [--resume runs/X/last.pt] [key=value ...]
#   sbatch --time=0:20:00 scripts/slurm/train.sbatch configs/debug.yaml     # smoke test
set -euo pipefail
source scripts/slurm/env.sh
CONFIG=${1:?usage: sbatch scripts/slurm/train.sbatch <config> [args...]}; shift
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/train.py --config "$CONFIG" "$@" wandb.tags="[slurm-$SLURM_JOB_ID]"
