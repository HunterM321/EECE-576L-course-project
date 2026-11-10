"""CoFiI2P data loaders with configurable sequences and a held-out validation split.

Samples are produced by CoFiI2P's own dataset classes, so preprocessing (voxel/random downsampling to
20480 points, per-index seeded perturbation, crops, intrinsics) is exactly upstream's. Only the list of
frames is changed:

    train  cfg.data.train_seqs minus the last `val_fraction` of each sequence   (CoFiI2P mode 'train':
           random crop + colour jitter)
    val    the last `val_fraction` of each training sequence, interleaved round-robin across sequences so
           that a prefix (val_max_frames) covers every sequence                 (mode 'val': centre crop)
    test   cfg.data.test_seqs                                                    (mode 'val')

CoFiI2P seeds each sample's perturbation by its position in the frame list. The KITTI test list for
seqs [9, 10] is built in upstream order, so test indices (and perturbations) match upstream eval_all.py.
`max_<split>_frames` keeps a prefix, which preserves that correspondence.

Pose convention: sample['P'] maps the perturbed cloud to the camera frame, p_cam = R p + t.
"""
import math
import os
from pathlib import Path

import torch

from i2pflow.paths import ensure_cofii2p_on_path, resolve

SPLITS = ("train", "val", "test")


# ---- options -------------------------------------------------------------------

def build_options(cfg):
    """CoFiI2P's Options object for cfg.data.dataset, pointed at cfg.data.root."""
    ensure_cofii2p_on_path()
    from data.options import Options_KITTI, Options_Nuscenes  # CoFiI2P package

    name = cfg.data.dataset
    if name == "kitti":
        opt = Options_KITTI()
    elif name == "nuscenes":
        opt = Options_Nuscenes()
    else:
        raise ValueError(f"unknown dataset {name!r} (kitti | nuscenes)")
    opt.data_path = str(resolve(cfg.data.root))
    opt.num_workers = cfg.data.num_workers
    return opt


# ---- frame lists ---------------------------------------------------------------

def _n_val(n, frac):
    return int(math.ceil(n * frac)) if frac > 0 else 0


def _interleave(groups):
    out, k = [], 0
    while any(k < len(g) for g in groups):
        out.extend(g[k] for g in groups if k < len(g))
        k += 1
    return out


def _kitti_entries(root, seqs, split, val_fraction):
    """Upstream entry tuples (img_folder, pc_folder, K_folder, seq, i, cam, n), both cameras per frame."""
    groups = []
    for seq in seqs:
        d = os.path.join(root, "sequences", "%02d" % seq)
        folders = {cam: (os.path.join(d, f"img_{cam}"), os.path.join(d, "pc_npy_with_normal"),
                         os.path.join(d, f"K_{cam}")) for cam in ("P2", "P3")}
        n = len(os.listdir(folders["P2"][0]))
        if split == "test":
            frames = range(n)
        else:
            cut = n - _n_val(n, val_fraction)
            frames = range(cut) if split == "train" else range(cut, n)
        groups.append([(img, pc, K, seq, i, cam, n) for i in frames for cam in ("P2", "P3")
                       for img, pc, K in [folders[cam]]])
    if split == "val":
        # round-robin over sequences, keeping each frame's two cameras together
        pairs = [[g[j:j + 2] for j in range(0, len(g), 2)] for g in groups]
        return [e for pair in _interleave(pairs) for e in pair]
    return [e for g in groups for e in g]


def _nuscenes_entries(root, split, val_fraction):
    folder = "test" if split == "test" else "train"
    files = sorted(os.listdir(os.path.join(root, folder, "pc")))
    if split == "test":
        return files
    cut = len(files) - _n_val(len(files), val_fraction)
    return files[:cut] if split == "train" else files[cut:]


def _nuscenes_val_view(root):
    """A directory whose `test/` is the real `train/`, so CoFiI2P's nuScenes loader reads held-out training
    frames with its deterministic (test-mode) crop and no augmentation."""
    view = Path(root) / "_val_view"
    view.mkdir(exist_ok=True)
    link = view / "test"
    if not link.exists():
        try:
            link.symlink_to(Path("..") / "train", target_is_directory=True)
        except FileExistsError:  # created concurrently by another process
            pass
    return str(view)


# ---- datasets ------------------------------------------------------------------

def _subclass(base, entries_method):
    class Split(base):
        """`base` with a fixed frame list (the upstream class builds its own from hard-coded sequences)."""

        def __init__(self, opt, mode, entries):
            self._entries = list(entries)
            super().__init__(opt, mode, is_front=False)

        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample.setdefault("index", index)  # KITTI's (patched) loader sets it; nuScenes' does not
            return sample

    setattr(Split, entries_method, lambda self, root_path, mode: self._entries)
    return Split


def build_dataset(cfg, split):
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    opt = build_options(cfg)
    dc = cfg.data
    root = opt.data_path
    if not os.path.isdir(root):
        raise FileNotFoundError(f"{cfg.data.dataset} data not found at {root} (cfg.data.root={dc.root})")
    mode = "train" if split == "train" else "val"

    if dc.dataset == "kitti":
        from data.kitti import kitti_pc_img_dataset as base
        entries = _kitti_entries(root, dc.train_seqs if split != "test" else dc.test_seqs, split, dc.val_fraction)
        cls = _subclass(base, "make_kitti_dataset")
    else:
        from data.nuscenes import nuscenes_pc_img_dataset as base
        entries = _nuscenes_entries(root, split, dc.val_fraction)
        if split == "val":
            opt.data_path = _nuscenes_val_view(root)
        cls = _subclass(base, "make_nuscenes_dataset")

    limit = dc.get(f"max_{split}_frames")
    if limit:
        entries = entries[:limit]
    return cls(opt, mode, entries)


def collate_list(samples):
    """Frames have different shapes (KPConv pyramids), so batches are plain lists of samples."""
    return list(samples)


def build_loader(cfg, split, shuffle=None, batch_size=None):
    ds = build_dataset(cfg, split)
    train = split == "train"
    # eval loaders run alongside the (persistent) train workers, at ~1.5 GB RSS each: keep them few
    nw = cfg.data.num_workers if train else cfg.data.get("eval_num_workers", cfg.data.num_workers)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size or (cfg.train.frames_per_step if train else cfg.eval.frames_per_batch),
        shuffle=train if shuffle is None else shuffle,
        drop_last=train,
        num_workers=nw,
        collate_fn=collate_list,
        persistent_workers=nw > 0,
        prefetch_factor=cfg.data.get("prefetch_factor", 2) if nw > 0 else None,  # batches buffered per worker
    )


# ---- device / pose helpers -----------------------------------------------------

def _move(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _move(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_move(v, device) for v in x)
    return x


def _squeezed(x, device):
    t = x if torch.is_tensor(x) else torch.as_tensor(x)
    return torch.squeeze(t).to(device, non_blocking=True)


def to_device(sample, device):
    """Move one sample to `device`. KPConv inputs are converted as upstream eval_all.py's batch-size-1
    collation + squeeze does (e.g. the per-stage int `lengths` become 0-d tensors)."""
    out = {}
    for k, v in sample.items():
        if k == "pc_data_dict":
            out[k] = {kk: [_squeezed(t, device) for t in vv] if isinstance(vv, list) else _squeezed(vv, device)
                      for kk, vv in v.items()}
        else:
            out[k] = _move(v, device)
    return out


def gt_pose(samples, device):
    """Ground-truth (R [B,3,3], t [B,3]) with p_cam = R p + t, from sample['P'] (= inv(perturbation))."""
    P = torch.stack([s["P"] for s in samples]).to(device=device, dtype=torch.float32)
    return P[:, :3, :3], P[:, :3, 3]
