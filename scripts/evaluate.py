"""Evaluate a trained head, or the CoFiI2P + PnP-RANSAC baseline, on the test split.

    # trained head (config is read from the run directory)
    python scripts/evaluate.py --ckpt runs/<name>/best.pt [eval.exact_nll=true inference.rot_anneal=10]
    # baseline (same metrics pipeline)
    python scripts/evaluate.py --method pnp --config configs/kitti_flow.yaml

Writes <out>/<tag>.npz (per-frame outputs) and <tag>_metrics.json.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from i2pflow.builders import build_backbone, build_space  # noqa: E402
from i2pflow.config import Config, apply_overrides, load_config  # noqa: E402
from i2pflow.data.datasets import build_loader, build_options  # noqa: E402
from i2pflow.engine.evaluator import evaluate_head, evaluate_pnp, format_summary, save_results, summarize  # noqa: E402
from i2pflow.engine.trainer import load_head_for_eval  # noqa: E402
from i2pflow.paths import resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default=None, help="required for --method pnp; optional with --ckpt")
    ap.add_argument("--method", default=None, help="pnp to evaluate the baseline")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--save-samples", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    if args.method == "pnp":
        cfg = load_config(args.config, args.overrides)
        out = resolve(args.out or f"results/baselines/cofii2p_pnp/{cfg.data.dataset}")
        tag = args.tag or f"pnp_{args.split}"
    else:
        ck = torch.load(resolve(args.ckpt), map_location="cpu")
        cfg = load_config(args.config, args.overrides) if args.config else \
            Config.wrap(apply_overrides(ck["config"], args.overrides))
        out = resolve(args.out or resolve(args.ckpt).parent / "eval")
        tag = args.tag or f"{Path(args.ckpt).stem}_{args.split}" + ("_anneal" if cfg.inference.rot_anneal else "")

    device = torch.device("cuda")
    space = build_space(cfg)
    opt = build_options(cfg)
    cfg.data.eval_num_workers = cfg.data.num_workers  # no train workers alongside: use them all
    loader = build_loader(cfg, args.split, shuffle=False, batch_size=cfg.eval.frames_per_batch)
    limit = cfg.data.get(f"max_{args.split}_frames")

    if args.method == "pnp":
        from i2pflow.baselines.pnp_ransac import CoFiI2PPnP
        pnp = CoFiI2PPnP(opt, cfg.backbone.ckpt, device)
        res = evaluate_pnp(pnp, loader, space, cfg.eval, device, max_frames=limit, save_samples=args.save_samples)
    else:
        if cfg.inference.rot_anneal and cfg.eval.exact_nll:
            print("note: rot_anneal > 0 changes the sampler only; exact NLL still uses the probability-flow ODE")
        backbone = build_backbone(cfg, opt, device)
        head, _ = load_head_for_eval(args.ckpt, cfg, device, use_ema=not args.no_ema)
        res = evaluate_head(head, backbone, loader, space, cfg.eval, device, max_frames=limit,
                            save_samples=args.save_samples)
    summ = summarize(res, space)
    save_results(res, summ, out, tag)
    print(format_summary(summ))
    print(f"saved -> {out}/{tag}.npz, {tag}_metrics.json")


if __name__ == "__main__":
    main()
