#!/bin/bash
# I2P-Flow / CoFiI2P environment on the Alliance cluster (uv venv; mirrors the moss conda env that
# reproduced CoFiI2P exactly). Downloads wheels only, so it is safe to run on a login node; GeoTransformer's
# CUDA extension is compiled afterwards on a compute node: sbatch scripts/slurm/build_geotransformer.sbatch
#
# The cluster's module Pythons refuse manylinux wheels (and its wheelhouse has no open3d 0.17), so we use a
# uv-managed standalone CPython 3.8 instead.
set -euo pipefail
UV=${UV:-/project/6101778/masibo/tools/uv}
E=${I2P_ENV:-/project/6101778/masibo/uv_envs/cofii2p}
export UV_PYTHON_INSTALL_DIR=/project/6101778/masibo/uv_envs/python
export UV_CACHE_DIR=/project/6101778/masibo/uv_cache        # keep the ~3 GB cache out of $HOME
export PIP_CONFIG_FILE=/dev/null UV_NO_CONFIG=1

$UV python install 3.8
$UV venv --python 3.8 --seed "$E"
pipi() { $UV pip install --python "$E/bin/python" "$@"; }
# torch 1.13.1+cu116 is required: open3d 0.17's ML ops are built for CUDA 11.6 (cu117 silently falls back to CPU)
pipi --no-build torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116 --index-strategy unsafe-best-match
pipi --no-build open3d==0.17.0 "opencv-python<5" "numpy<1.25" scipy tqdm easydict einops tensorboardX \
  "tensorboard<2.15" matplotlib pyyaml pytest ninja wandb
# pure-Python sdists (no compilation)
pipi fvcore pyquaternion nuscenes-devkit
$UV pip freeze --python "$E/bin/python" > "$(dirname "$0")/../results/env_freeze.txt"
echo INSTALL_DONE
