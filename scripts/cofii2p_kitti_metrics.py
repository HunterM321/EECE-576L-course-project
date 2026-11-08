"""Re-score per-frame dumps written by the upstream third_party/CoFiI2P/evaluation/eval_all.py.

    python scripts/cofii2p_kitti_metrics.py results/baselines/cofii2p_pnp/kitti/upstream_eval_20261020

The dumps pickle CUDA tensors, so run with a GPU visible. For new baseline runs prefer
`scripts/evaluate.py --method pnp`, which writes the same format as the heads.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402,F401  (needed to unpickle the dumps)

from i2pflow.eval.metrics import accuracy_metrics  # noqa: E402
from i2pflow.geometry.so3 import pose_errors  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", type=Path)
    args = ap.parse_args()
    files = sorted(args.eval_dir.glob("*.npy"))
    dumps = [np.load(f, allow_pickle=True).item() for f in files]
    pred = np.stack([d["pred_P"] for d in dumps])
    gt = np.stack([d["GT_P"] for d in dumps])
    rre_e, rre_g, rte = pose_errors(pred, gt)
    print(f"{len(files)} frames from {args.eval_dir}")
    for name, rre in (("RRE = sum |Euler| (I2P convention)", rre_e), ("RRE = geodesic", rre_g)):
        m = accuracy_metrics(rre, rte)
        print(f"[{name}] RRE {m['rre_mean']:.2f} ± {m['rre_std']:.2f} deg, RTE {m['rte_mean']:.2f} ± {m['rte_std']:.2f} m, "
              f"median {m['rre_median']:.2f} deg / {m['rte_median']:.2f} m, Acc {m['acc']:.2f}%, RR {m['rr']:.2f}%")


if __name__ == "__main__":
    main()
