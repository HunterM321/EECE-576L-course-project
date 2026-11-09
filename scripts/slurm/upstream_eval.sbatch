#!/bin/bash -l
#SBATCH --account=aip-rjliao
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --job-name=upstream_eval
#SBATCH --output=logs/%x-%j.out
# Runs CoFiI2P's own evaluation/eval_all.py on the test split and keeps its per-frame dumps as the reference
# for tests/test_gpu_pipeline.py and scripts/cofii2p_kitti_metrics.py. Usage (from the repo root):
#   sbatch scripts/slurm/upstream_eval.sbatch kitti|nuscenes
# Upstream reads data from ../corri2p_data (KITTI) or ../nuscenes_i2p (nuScenes) relative to
# third_party/CoFiI2P; both are symlinks in third_party/ to /project/6101778/masibo/cofii2p/.
set -euo pipefail
DS=${1:-kitti}
source scripts/slurm/env.sh
R=$PWD
OUT=$R/results/baselines/cofii2p_pnp/$DS; mkdir -p "$OUT"
STAMP=$(date +%Y%m%d)
tmp=$OUT/_upstream_tmp; rm -rf "$tmp"
cd third_party/CoFiI2P
PYTHONPATH=$PWD python evaluation/eval_all.py $R/checkpoints/cofii2p/cofii2p_$DS.t7 $DS --eval_path "$tmp" \
  | tee "$OUT/upstream_eval_$STAMP.log"
mv "$tmp"/$DS/* "$OUT/upstream_eval_$STAMP"; rm -rf "$tmp"
cd $R && python scripts/cofii2p_kitti_metrics.py "$OUT/upstream_eval_$STAMP"
