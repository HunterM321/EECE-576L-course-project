"""Pose-conditioned transformer shared by the flow head and the matched Gaussian head.

For a batch of B frames and S pose hypotheses (R, t) per frame it builds one token per selected
superpoint ("render-and-compare"): the point is moved into the camera frame with the hypothesis,
projected into CoFiI2P's coarse (stride 8) and fine (stride 2) image feature maps, and the sampled
image features, point/image feature similarities, depth and visibility are concatenated with the
pose-independent point features. A [POSE] token carries the current pose (and, with
self-conditioning, the previous endpoint estimate). Blocks are DiT-style with adaLN-Zero time
conditioning; optional cross-attention to pooled image tokens gives global context when few
points are visible. The output is read from the [POSE] token.
"""
import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from i2pflow.geometry.camera import project, sample_map, transform_points


def timestep_embedding(tau, dim, max_period=10000.0):
    """Sinusoidal embedding of tau in [0, 1] (scaled by 1000 as in DiT)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=tau.device, dtype=torch.float32) / half)
    args = tau.float()[..., None] * 1000.0 * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def sine_pos_2d(h, w, dim, device):
    """Fixed 2D sine positional encoding [h*w, dim]."""
    y, x = torch.meshgrid(torch.linspace(0, 1, h, device=device), torch.linspace(0, 1, w, device=device), indexing="ij")
    quarter = dim // 4
    freqs = torch.exp(torch.arange(quarter, device=device, dtype=torch.float32) * (math.log(100.0) / max(quarter - 1, 1)))
    ax = x.reshape(-1, 1) * freqs * math.pi
    ay = y.reshape(-1, 1) * freqs * math.pi
    pe = torch.cat([ax.sin(), ax.cos(), ay.sin(), ay.cos()], dim=-1)
    return F.pad(pe, (0, dim - pe.shape[-1]))


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class Mlp(nn.Sequential):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))


class Block(nn.Module):
    def __init__(self, d, heads, mlp_ratio=4.0, cross=False):
        super().__init__()
        self.cross = cross
        self.n_mod = 9 if cross else 6
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        if cross:
            self.norm_c = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
            self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(d, int(d * mlp_ratio), d)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, self.n_mod * d))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c, ctx=None, key_padding_mask=None):
        m = self.ada(c).chunk(self.n_mod, dim=-1)
        h = modulate(self.norm1(x), m[0], m[1])
        x = x + m[2][:, None] * self.attn(h, h, h, need_weights=False, key_padding_mask=key_padding_mask)[0]
        if self.cross:
            h = modulate(self.norm_c(x), m[6], m[7])
            x = x + m[8][:, None] * self.cross_attn(h, ctx, ctx, need_weights=False)[0]
        h = modulate(self.norm2(x), m[3], m[4])
        return x + m[5][:, None] * self.mlp(h)


class PoseNet(nn.Module):
    PC_DIM, PC_FINE_DIM, IMG_DIM, IMG_FINE_DIM = 128, 64, 128, 64

    def __init__(self, cfg, space, out_dim):
        super().__init__()
        self.space = space
        self.d = d = cfg.d_model
        self.n_points = cfg.n_points
        self.point_select = cfg.point_select
        self.self_cond = cfg.self_cond
        self.img_pool = cfg.img_pool
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[cfg.amp]
        self.max_coord = 5.0  # clamp for normalised camera-frame coordinates

        n_geo = 3 + 1 + 1 + 2 + 2  # q/s, log z, in front, in coarse/fine image, coarse uv
        n_dyn = self.IMG_DIM + self.IMG_FINE_DIM + 3 + n_geo + 2  # + cos_c, cos_f, img score, + self-cond
        self.static_proj = nn.Sequential(nn.Linear(self.PC_DIM + self.PC_FINE_DIM + 1, d), nn.LayerNorm(d))
        self.dyn_proj = Mlp(n_dyn, d, d)
        self.pose_proj = Mlp(9 + 9 + 1, d, d)  # current pose, self-cond endpoint, self-cond flag
        self.pose_token = nn.Parameter(torch.zeros(1, 1, d))
        self.time_mlp = Mlp(d, d, d)
        self.img_proj = nn.Linear(self.IMG_DIM + 1, d)

        n_cross = cfg.cross_attn_layers
        self.blocks = nn.ModuleList([Block(d, cfg.heads, cfg.mlp_ratio, cross=i < n_cross) for i in range(cfg.depth)])
        self.uses_ctx = n_cross > 0
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out = nn.Linear(d, out_dim)
        for m in (self.final_ada[-1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    # ---- pose-independent context ------------------------------------------------

    def select_points(self, score):
        B, N = score.shape
        k = min(self.n_points, N)
        if self.point_select == "topk" or not self.training:
            return score.topk(k, dim=-1).indices
        if self.point_select == "sample":  # score-weighted sampling without replacement
            return torch.multinomial(score.clamp_min(1e-3), k, replacement=False)
        if self.point_select == "random":
            return torch.rand(B, N, device=score.device).argsort(-1)[:, :k]
        raise ValueError(self.point_select)

    def encode_context(self, feats):
        idx = self.select_points(feats["pc_score"])
        gather = lambda x: torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))
        ctx = {
            "points": gather(feats["points"]),
            "pc_feat": gather(feats["pc_feat"]),
            "pc_fine_feat": gather(feats["pc_fine_feat"]),
            "img_coarse": feats["img_coarse"],
            "img_score": feats["img_score"],
            "img_fine": feats["img_fine"],
            "K_coarse": feats["K_coarse"],
            "K_fine": feats["K_fine"],
        }
        score = torch.gather(feats["pc_score"], 1, idx)
        ctx["static"] = self.static_proj(torch.cat([ctx["pc_feat"], ctx["pc_fine_feat"], score[..., None]], -1))
        if self.uses_ctx:
            img = torch.cat([feats["img_coarse"], feats["img_score"]], 1)
            if self.img_pool > 1:
                img = F.avg_pool2d(img, self.img_pool)
            B, C, H, W = img.shape
            tok = self.img_proj(img.flatten(2).transpose(1, 2))
            ctx["img_tokens"] = tok + sine_pos_2d(H, W, self.d, img.device)[None]
        return ctx

    # ---- pose-dependent tokens ----------------------------------------------------------

    def _pose_vec(self, R, t):
        return torch.cat([R[..., :, 0], R[..., :, 1], (t / self.space.s).clamp(-self.max_coord, self.max_coord)], -1)

    def point_features(self, ctx, R, t, sc=None):
        """Pose-dependent per-point features [B, S, N, n_dyn] (fp32 geometry)."""
        B, S = R.shape[:2]
        s = self.space.s
        q = transform_points(R, t, ctx["points"][:, None])                     # [B, S, N, 3]
        uv_c, z, front = project(q, ctx["K_coarse"][:, None])
        f_c, in_c = sample_map(ctx["img_coarse"], uv_c, front)
        sc_img, _ = sample_map(ctx["img_score"], uv_c, front)
        uv_f, _, _ = project(q, ctx["K_fine"][:, None])
        f_f, in_f = sample_map(ctx["img_fine"], uv_f, front)
        cos_c = (f_c * ctx["pc_feat"][:, None]).sum(-1, keepdim=True)
        cos_f = (f_f * ctx["pc_fine_feat"][:, None]).sum(-1, keepdim=True)

        H, W = ctx["img_coarse"].shape[-2:]
        uv_n = torch.stack([2 * uv_c[..., 0] / (W - 1) - 1, 2 * uv_c[..., 1] / (H - 1) - 1], -1).clamp(-2, 2)
        geo = torch.cat([
            (q / s).clamp(-self.max_coord, self.max_coord),
            torch.log(z.clamp(0.1, 200.0))[..., None] / 3.0,
            front[..., None].float(), in_c[..., None].float(), in_f[..., None].float(),
            uv_n * front[..., None],
        ], -1)

        if sc is not None:  # self-conditioning: coarse similarity under the previous endpoint estimate
            q_sc = transform_points(sc[0], sc[1], ctx["points"][:, None])
            uv_sc, _, front_sc = project(q_sc, ctx["K_coarse"][:, None])
            f_sc, in_sc = sample_map(ctx["img_coarse"], uv_sc, front_sc)
            sc_feat = torch.cat([(f_sc * ctx["pc_feat"][:, None]).sum(-1, keepdim=True), in_sc[..., None].float()], -1)
        else:
            sc_feat = torch.zeros(B, S, q.shape[2], 2, device=q.device, dtype=q.dtype)

        return torch.cat([f_c, f_f, cos_c, cos_f, sc_img, geo, sc_feat], -1)

    def forward(self, ctx, R, t, tau, sc=None):
        """R [B,S,3,3], t [B,S,3], tau [B,S]; sc = optional (R1_prev, t1_prev). Returns [B, S, out_dim].

        Geometry (projection, sampling) runs in fp32; the transformer runs under autocast(self.amp_dtype).
        """
        B, S = R.shape[:2]
        R, t = R.float(), t.float()
        if sc is not None:
            sc = (sc[0].float(), sc[1].float())
        dyn = self.point_features(ctx, R, t, sc)
        use_amp = self.amp_dtype is not None and R.is_cuda
        with torch.autocast("cuda", dtype=self.amp_dtype) if use_amp else contextlib.nullcontext():
            tokens = ctx["static"][:, None] + self.dyn_proj(dyn)                                # [B, S, N, d]
            if sc is not None:
                sc_vec = torch.cat([self._pose_vec(*sc), torch.ones(B, S, 1, device=R.device)], -1)
            else:
                sc_vec = torch.zeros(B, S, 10, device=R.device)
            pose = self.pose_token + self.pose_proj(torch.cat([self._pose_vec(R, t), sc_vec], -1))  # [B, S, d]
            x = torch.cat([pose[:, :, None].to(tokens.dtype), tokens], 2).flatten(0, 1)            # [B*S, 1+N, d]
            c = self.time_mlp(timestep_embedding(tau.flatten(), self.d))
            img_ctx = None
            if self.uses_ctx:
                img_ctx = ctx["img_tokens"][:, None].expand(-1, S, -1, -1).flatten(0, 1)
            for blk in self.blocks:
                x = blk(x, c, img_ctx)
            shift, scale = self.final_ada(c).chunk(2, dim=-1)
            out = self.out(modulate(self.final_norm(x[:, :1]), shift, scale)[:, 0])
        return out.float().view(B, S, -1)
