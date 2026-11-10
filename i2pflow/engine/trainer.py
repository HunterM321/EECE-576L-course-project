"""Training loop for the pose heads on a frozen CoFiI2P backbone."""
import copy
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from i2pflow.builders import build_backbone, build_head, build_space
from i2pflow.config import save_config
from i2pflow.data.datasets import build_loader, build_options, gt_pose, to_device
from i2pflow.engine.evaluator import evaluate_head, format_summary, save_results, summarize
from i2pflow.paths import resolve


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


def lr_at(step, tc):
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    p = (step - tc.warmup_steps) / max(tc.total_steps - tc.warmup_steps, 1)
    return tc.lr * (tc.min_lr_ratio + (1 - tc.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


def _amp_dtype(name):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[name]


class WandbLogger:
    """Thin wrapper so the trainer runs unchanged with wandb off. Resuming a run directory continues its wandb run."""

    def __init__(self, cfg, out_dir):
        wc = cfg.get("wandb") or {}
        self.run = None
        if not wc.get("enabled", False) or wc.get("mode") == "disabled":
            return
        import wandb
        id_file = out_dir / "wandb_id.txt"
        run_id = id_file.read_text().strip() if id_file.exists() else wandb.util.generate_id()
        id_file.write_text(run_id)
        self.run = wandb.init(project=wc.get("project", "i2pflow"), entity=wc.get("entity"), name=out_dir.name,
                              id=run_id, resume="allow", dir=str(out_dir), mode=wc.get("mode", "online"),
                              tags=list(wc.get("tags") or []) + [cfg.method, cfg.data.dataset, cfg.pose.dof],
                              config=cfg.to_dict())
        self.run.define_metric("train/step")
        self.run.define_metric("val/*", step_metric="train/step")
        self.run.define_metric("train/*", step_metric="train/step")

    def log(self, metrics, step):
        if self.run is not None:
            self.run.log({**metrics, "train/step": step})

    def summary(self, **kv):
        if self.run is not None:
            self.run.summary.update(kv)

    def finish(self):
        if self.run is not None:
            self.run.finish()


class Trainer:
    def __init__(self, cfg, out_dir, resume=None):
        self.cfg = cfg
        self.tc = cfg.train
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, self.out_dir / "config.yaml")
        torch.manual_seed(cfg.seed)
        self.device = torch.device("cuda")

        self.space = build_space(cfg)
        self.opt = build_options(cfg)
        self.backbone = build_backbone(cfg, self.opt, self.device)
        self.head = build_head(cfg, self.space).to(self.device)
        self.ema = EMA(self.head, self.tc.ema_decay) if self.tc.ema_decay > 0 else None
        self.optim = torch.optim.AdamW(self.head.parameters(), lr=self.tc.lr, weight_decay=self.tc.weight_decay,
                                       betas=tuple(self.tc.betas))
        self.scaler = torch.cuda.amp.GradScaler(enabled=_amp_dtype(cfg.model.amp) == torch.float16)
        self.step = 0
        self.best = -float("inf")
        if resume:
            self.load(resume)

        self.train_loader = build_loader(cfg, "train")
        self.val_loader = build_loader(cfg, "val", shuffle=False, batch_size=cfg.eval.frames_per_batch)
        self.writer = SummaryWriter(str(self.out_dir / "tb"))
        self.wb = WandbLogger(cfg, self.out_dir)
        n_params = sum(p.numel() for p in self.head.parameters())
        self.wb.summary(head_params=n_params, train_frames=len(self.train_loader.dataset),
                        val_frames=len(self.val_loader.dataset))
        print(f"head: {cfg.method}, {n_params / 1e6:.2f}M params | train frames {len(self.train_loader.dataset)}"
              f" | val frames {len(self.val_loader.dataset)} | out {self.out_dir}", flush=True)

    # ---- checkpointing ------------------------------------------------------

    def state(self):
        return {"step": self.step, "best": self.best, "head": self.head.state_dict(),
                "ema": self.ema.model.state_dict() if self.ema else None, "optim": self.optim.state_dict(),
                "scaler": self.scaler.state_dict(), "config": self.cfg.to_dict()}

    def save(self, name):
        torch.save(self.state(), self.out_dir / name)

    def load(self, path):
        ck = torch.load(resolve(path), map_location=self.device)
        self.head.load_state_dict(ck["head"])
        if self.ema and ck.get("ema"):
            self.ema.model.load_state_dict(ck["ema"])
        self.optim.load_state_dict(ck["optim"])
        self.scaler.load_state_dict(ck["scaler"])
        self.step, self.best = ck["step"], ck["best"]
        print(f"resumed from {path} at step {self.step}", flush=True)

    # ---- loop ----------------------------------------------------------------

    def train_step(self, samples):
        samples = [to_device(s, self.device) for s in samples]
        R1, t1 = gt_pose(samples, self.device)
        feats = self.backbone(samples)
        self.head.train()
        loss, logs = self.head.loss(feats, R1, t1)  # PoseNet applies autocast internally (cfg.model.amp)
        for g in self.optim.param_groups:
            g["lr"] = lr_at(self.step, self.tc)
        self.optim.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optim)
        gnorm = torch.nn.utils.clip_grad_norm_(self.head.parameters(), self.tc.grad_clip)
        self.scaler.step(self.optim)
        self.scaler.update()
        if self.ema:
            self.ema.update(self.head)
        logs["grad_norm"] = gnorm
        logs["lr"] = torch.tensor(self.optim.param_groups[0]["lr"])
        return logs

    def validate(self):
        model = self.ema.model if self.ema else self.head
        ec = self.cfg.eval
        res = evaluate_head(model, self.backbone, self.val_loader, self.space, ec, self.device,
                            max_frames=ec.val_max_frames, log_every=0)
        summ = summarize(res, self.space)
        scalars = {f"val/{k}": v for k, v in summ.items() if isinstance(v, (int, float)) and math.isfinite(v)}
        for k, v in scalars.items():
            self.writer.add_scalar(k, v, self.step)
        self.wb.log(scalars, self.step)
        print(f"[val step {self.step}] {format_summary(summ)}", flush=True)
        with open(self.out_dir / "val_log.jsonl", "a") as f:
            f.write(json.dumps({"step": self.step, **summ}) + "\n")
        return summ

    def fit(self):
        tc = self.tc
        t_last, acc = time.time(), {}
        while self.step < tc.total_steps:
            for samples in self.train_loader:
                logs = self.train_step(samples)
                for k, v in logs.items():
                    v = float(v)
                    if math.isfinite(v):
                        acc.setdefault(k, []).append(v)
                self.step += 1
                if self.step % tc.log_every == 0:
                    means = {k: sum(v) / len(v) for k, v in acc.items()}
                    for k, v in means.items():
                        self.writer.add_scalar(f"train/{k}", v, self.step)
                    dt = (time.time() - t_last) / tc.log_every
                    self.wb.log({**{f"train/{k}": v for k, v in means.items()}, "train/sec_per_it": dt,
                                 "train/frames_per_sec": tc.frames_per_step / max(dt, 1e-9)}, self.step)
                    print(f"step {self.step}/{tc.total_steps}  {dt:.2f}s/it  " +
                          "  ".join(f"{k}={v:.4g}" for k, v in means.items()), flush=True)
                    acc, t_last = {}, time.time()
                if self.step % tc.val_every == 0 or self.step == tc.total_steps:
                    summ = self.validate()
                    metric = summ[tc.select_metric] * (1 if tc.select_mode == "max" else -1)
                    if metric > self.best:
                        self.best = metric
                        self.save("best.pt")
                        self.wb.summary(**{f"best/{tc.select_metric}": summ[tc.select_metric],
                                           "best/step": self.step})
                if self.step % tc.ckpt_every == 0 or self.step == tc.total_steps:
                    self.save("last.pt")
                if self.step >= tc.total_steps:
                    break
        self.writer.close()
        self.wb.finish()


def load_head_for_eval(ckpt_path, cfg, device, use_ema=True):
    space = build_space(cfg)
    head = build_head(cfg, space).to(device)
    ck = torch.load(resolve(ckpt_path), map_location=device)
    head.load_state_dict(ck["ema"] if use_ema and ck.get("ema") else ck["head"])
    return head.eval(), space


__all__ = ["Trainer", "load_head_for_eval", "save_results"]
