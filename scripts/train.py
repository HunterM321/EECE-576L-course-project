"""Train a pose head (flow or Gaussian) on the frozen CoFiI2P backbone.

    CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/kitti_flow.yaml [key=value ...]
    python scripts/train.py --config configs/kitti_flow.yaml --resume runs/<name>/last.pt
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from i2pflow.config import load_config  # noqa: E402
from i2pflow.engine.trainer import Trainer  # noqa: E402
from i2pflow.paths import resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default=None, help="run name (default: <config>_<timestamp>)")
    ap.add_argument("--resume", default=None, help="checkpoint to resume (its run directory is reused)")
    ap.add_argument("overrides", nargs="*", help="config overrides, e.g. train.lr=2e-4")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.resume:
        out_dir = resolve(args.resume).parent
    else:
        name = args.name or f"{Path(args.config).stem}_{datetime.now():%Y%m%d_%H%M%S}"
        out_dir = resolve(cfg.output_root) / name
    Trainer(cfg, out_dir, resume=args.resume).fit()


if __name__ == "__main__":
    main()
