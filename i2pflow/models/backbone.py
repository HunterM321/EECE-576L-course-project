"""Frozen CoFiI2P feature extractor.

Runs CoFiI2P up to (and including) its coarse image<->point transformer and the fine image
decoder, i.e. everything before the hard match selection and PnP-RANSAC. Outputs, per frame:

    points        [N=1280, 3]   coarse superpoints of the perturbed cloud
    pc_feat       [N, 128]      fused coarse point features (L2-normalised)
    pc_score      [N]           CoFiI2P's in-frustum score for each superpoint
    pc_fine_feat  [N, 64]       fine (stride-2 matching) feature of the nearest level-1 point
    img_coarse    [128, 20, 64] fused coarse image features, stride 8 (L2-normalised)
    img_score     [1, 20, 64]   CoFiI2P's image-side score
    img_fine      [64, 80, 256] fine image features, stride 2 (L2-normalised)
    K_coarse      [3, 3]        intrinsics of the stride-8 map  (sample['K_4'])
    K_fine        [3, 3]        intrinsics of the stride-2 map  (sample['K'])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from i2pflow.paths import ensure_cofii2p_on_path, resolve


def load_cofii2p(opt, ckpt_path, device):
    ensure_cofii2p_on_path()
    from model.network import CoFiI2P  # noqa: E402  (CoFiI2P package)

    net = CoFiI2P(opt)
    state = torch.load(resolve(ckpt_path), map_location="cpu")
    net.load_state_dict(state, strict=True)
    return net.to(device).eval()


def nearest_index(nodes, points):
    """For each of points [M, 3], index of the nearest of nodes [N, 3]."""
    d = torch.cdist(points[None], nodes[None])[0]
    return d.argmin(-1)


class FrozenCoFiI2P(nn.Module):
    def __init__(self, opt, ckpt_path, device):
        super().__init__()
        self.net = load_cofii2p(opt, ckpt_path, device)
        for p in self.net.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):  # always stays in eval mode
        return super().train(False)

    @torch.no_grad()
    def forward_single(self, sample):
        net = self.net
        d = sample["pc_data_dict"]
        img = sample["img"][None]
        pcs = net.pc_encoder(d)
        ims = net.img_encoder(img)

        pc_mid = F.normalize(net.pc_feature_layer(pcs[-1]), dim=1, p=2)   # [1280, 128]
        pc_fine_all = F.normalize(pcs[-4], dim=1, p=2)                    # [10240, 64]
        img_s8 = F.normalize(ims[-4], dim=1, p=2)                         # [1, 128, 20, 64]
        img_s4, img_s2 = ims[-5], ims[-6]

        dev = img.device
        ix, iy = torch.meshgrid(torch.arange(net.pe_H, device=dev), torch.arange(net.pe_W, device=dev), indexing="ij")
        img_xy = rearrange(torch.stack([ix, iy], dim=-1)[None], "b h w d -> b (h w) d")
        img_pos = net.img_pos_encoding(img_xy)
        pc_pos = net.pc_pos_encoding(d["points"][-1][None])
        img_tok = rearrange(img_s8, "b c h w -> b (h w) c") + img_pos
        pc_tok = pc_mid[None] + pc_pos
        img_mid, pc_fus = net.transformer(img_tok, pc_tok)
        img_mid = rearrange(img_mid, "b (h w) c -> b c h w", h=img_s8.shape[2])
        pc_fus = rearrange(pc_fus, "b n c -> b c n")

        pc_score = net.pc_score_layer(pc_fus)[0, 0]                        # [1280]
        img_score = net.img_score_layer(img_mid)[0]                        # [1, 20, 64]
        img_fine = F.normalize(net.img_upsample_2(net.img_upsample_1(img_s8, img_s4), img_s2), dim=1, p=2)

        points = d["points"][-1]
        fine_idx = nearest_index(d["points"][1], points)
        return {
            "points": points.float(),
            "pc_feat": F.normalize(pc_fus[0], dim=0, p=2).t().contiguous(),
            "pc_score": pc_score,
            "pc_fine_feat": pc_fine_all[fine_idx],
            "img_coarse": F.normalize(img_mid, dim=1, p=2)[0],
            "img_score": img_score,
            "img_fine": img_fine[0],
            "K_coarse": sample["K_4"].float(),
            "K_fine": sample["K"].float(),
        }

    @torch.no_grad()
    def forward(self, samples):
        """samples: list of per-frame dicts already on device -> dict of batched tensors [B, ...]."""
        feats = [self.forward_single(s) for s in samples]
        return {k: torch.stack([f[k] for f in feats]) for k in feats[0]}
