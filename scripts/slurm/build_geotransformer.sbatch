#!/bin/bash -l
#SBATCH --account=aip-rjliao
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --job-name=build_geotr
#SBATCH --output=logs/%x-%j.out
# Builds GeoTransformer's CUDA extension (needed by CoFiI2P's KPConv ops) and smoke-tests the env.
# Submit from the repo root: sbatch scripts/slurm/build_geotransformer.sbatch
set -euo pipefail
source scripts/slurm/env.sh
nvidia-smi --query-gpu=name,driver_version --format=csv
# nvcc 11.7 has no sm_89/sm_90 targets; sm_86 SASS runs on L40S (sm_89), +PTX JITs elsewhere.
export TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" MAX_JOBS=$SLURM_CPUS_PER_TASK
(cd third_party/GeoTransformer && pip install --no-deps --no-build-isolation -e .)
python -c "import torch, geotransformer.ext; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import torch, open3d as o3d, open3d.ml.torch as ml3d; print('open3d', o3d.__version__, 'ml ops cuda:', o3d.core.cuda.is_available())"
python -m pytest -q -p no:cacheprovider tests --ignore=tests/test_gpu_pipeline.py  # GPU tests need data + checkpoints
# Smoke-test the data loaders on real KITTI (frame counts + one sample per split).
python - <<'PY'
from i2pflow.config import load_config
from i2pflow.data.datasets import build_dataset, to_device, gt_pose
cfg = load_config("configs/kitti_flow.yaml", ["data.num_workers=0"])
for split in ("train", "val", "test"):
    ds = build_dataset(cfg, split)
    s = ds[0]
    R, t = gt_pose([to_device(s, "cuda")], "cuda")
    print(split, len(ds), "frames | sample0 keys:", len(s), "| img", tuple(s["img"].shape),
          "| t", [round(float(x), 2) for x in t[0]], "| index", s["index"])
PY
