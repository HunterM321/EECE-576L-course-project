#!/bin/bash -l
#SBATCH --account=aip-rjliao
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --job-name=extract_data
#SBATCH --output=logs/%x-%j.out
# Extracts the CoFiI2P preprocessed KITTI / nuScenes zips into /project. Usage (from the repo root):
#   sbatch scripts/slurm/extract_data.sbatch kitti|nuscenes
set -euo pipefail
B=/project/6101778/masibo/cofii2p
case "$1" in
  kitti)    zip=$B/downloads/I2P_KITTI.zip;    out=$B/kitti ;;
  nuscenes) zip=$B/downloads/nuscenes_i2p.zip; out=$B/nuscenes ;;
  *) echo "usage: $0 kitti|nuscenes"; exit 1 ;;
esac
unzip -tq "$zip" | tail -1            # integrity check before extracting
tmp=$out/_extract; rm -rf "$tmp"; mkdir -p "$tmp"
unzip -q "$zip" -d "$tmp"
# Flatten a single top-level folder (the KITTI zip wraps everything in corri2p_data/).
top=("$tmp"/*)
if [ ${#top[@]} -eq 1 ] && [ -d "${top[0]}" ]; then src=${top[0]}; else src=$tmp; fi
find "$src" -mindepth 1 -maxdepth 1 -exec mv -t "$out" {} +
rm -rf "$tmp"
ls "$out"; du -sh "$out"
echo EXTRACT_DONE
