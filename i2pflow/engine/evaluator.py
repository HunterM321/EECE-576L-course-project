"""Shared evaluation loop: every method yields per-frame pose samples + a point estimate, which are
scored with the same accuracy and uncertainty metrics (see i2pflow.eval.metrics)."""
import json
import time

import numpy as np
import torch

from i2pflow.data.datasets import gt_pose, to_device
from i2pflow.eval import metrics as M
from i2pflow.eval.modes import extract_modes
from i2pflow.geometry import so3


def _frame_outputs(R_s, t_s, R_gt, t_gt, space, ec, R_est=None, t_est=None):
    """Point estimate (top mode unless given), modes, and uncertainty scores for a batch."""
    B = R_s.shape[0]
    n_modes = ec.max_modes
    est_R, est_t = [], []
    modes_T = torch.zeros(B, n_modes, 4, 4, device=R_s.device)
    modes_w = torch.zeros(B, n_modes, device=R_s.device)
    n_found = torch.zeros(B, device=R_s.device)
    for b in range(B):
        modes = extract_modes(R_s[b], t_s[b], ec.mode_eps_rot_deg, ec.mode_eps_trans, max_modes=n_modes)
        n_found[b] = len(modes)
        for k, m in enumerate(modes):
            modes_T[b, k] = so3.to_matrix(m["R"], m["t"])
            modes_w[b, k] = m["weight"]
        est_R.append(modes[0]["R"])
        est_t.append(modes[0]["t"])
    if R_est is None:
        R_est, t_est = torch.stack(est_R), torch.stack(est_t)
    rot_sp, tr_sp = M.sample_spread(R_s, t_s, R_est, t_est)
    log_kde, level = M.kde_scores(R_s, t_s, R_gt, t_gt, space, ec.kde_bw_rot_deg, ec.kde_bw_trans)
    # min-over-samples error (oracle): does any sample land within the Acc thresholds?
    rel = R_s.transpose(-1, -2) @ R_gt[:, None]
    s_rot = torch.rad2deg(so3.angle(rel))
    s_tr = (t_s - t_gt[:, None]).norm(dim=-1)
    oracle = ((s_rot < M.ACC_THR[0]) & (s_tr < M.ACC_THR[1])).any(-1)
    return {
        "T_est": so3.to_matrix(R_est, t_est), "modes_T": modes_T, "modes_w": modes_w, "n_modes": n_found,
        "spread_rot_deg": rot_sp, "spread_trans_m": tr_sp,
        "score": rot_sp / M.ACC_THR[0] + tr_sp / M.ACC_THR[1],
        "score_mode_weight": 1.0 - modes_w[:, 0],
        "kde_log_gt": log_kde, "hpd_level": level, "oracle_hit": oracle.float(),
    }


@torch.no_grad()
def evaluate_head(head, backbone, loader, space, ec, device, max_frames=None, save_samples=False, log_every=100):
    head.eval()
    out, n_done, t0 = {}, 0, time.time()
    for samples in loader:
        samples = [to_device(s, device) for s in samples]
        R_gt, t_gt = gt_pose(samples, device)
        feats = backbone(samples)
        R_s, t_s = head.sample(feats, ec.n_samples)
        R_est = t_est = None
        if hasattr(head, "predict"):  # Gaussian head: use its mean as the point estimate
            R_est, t_est, _ = head.predict(feats)
        fo = _frame_outputs(R_s, t_s, R_gt, t_gt, space, ec, R_est, t_est)
        fo["T_gt"] = so3.to_matrix(R_gt, t_gt)
        if ec.exact_nll and hasattr(head, "log_likelihood"):
            fo["exact_log_gt"] = head.log_likelihood(feats, R_gt, t_gt)
        if save_samples:
            fo["samples_T"] = so3.to_matrix(R_s, t_s).half()
        fo["index"] = torch.tensor([s["index"] for s in samples], device=device)
        for k, v in fo.items():
            out.setdefault(k, []).append(v.detach().cpu())
        n_done += len(samples)
        if log_every and n_done % log_every < len(samples):
            print(f"  eval {n_done}/{len(loader.dataset)} frames  {time.time() - t0:.0f}s", flush=True)
        if max_frames and n_done >= max_frames:
            break
    return {k: torch.cat(v).numpy() for k, v in out.items()}


@torch.no_grad()
def evaluate_pnp(pnp, loader, space, ec, device, max_frames=None, save_samples=False, log_every=100):
    """CoFiI2P + PnP-RANSAC, with samples drawn from its Jacobian covariance for the uncertainty metrics."""
    out, n_done, t0 = {}, 0, time.time()
    for samples in loader:
        samples = [to_device(s, device) for s in samples]
        R_gt, t_gt = gt_pose(samples, device)
        res = [pnp(s) for s in samples]
        T_est = torch.tensor(np.stack([r["T"] for r in res]), device=device, dtype=torch.float32)
        cov = torch.tensor(np.stack([r["cov"] for r in res]), device=device, dtype=torch.float32)
        R_est, t_est = so3.from_matrix(T_est)
        L = torch.linalg.cholesky(cov.double()).float()  # cov is PD by construction; float64 for wide spectra
        z = (L[:, None] @ torch.randn(len(res), ec.n_samples, 6, 1, device=device))[..., 0]
        R_s = so3.exp(z[..., :3]) @ R_est[:, None]
        t_s = t_est[:, None] + z[..., 3:]
        fo = _frame_outputs(R_s, t_s, R_gt, t_gt, space, ec, R_est, t_est)
        fo["T_gt"] = so3.to_matrix(R_gt, t_gt)
        fo["inlier_ratio"] = torch.tensor([r["inlier_ratio"] for r in res])
        fo["score_inlier"] = 1.0 - fo["inlier_ratio"]
        fo["pnp_success"] = torch.tensor([float(r["success"]) for r in res])
        fo["cov"] = cov
        if save_samples:
            fo["samples_T"] = so3.to_matrix(R_s, t_s).half()
        fo["index"] = torch.tensor([s["index"] for s in samples])
        for k, v in fo.items():
            out.setdefault(k, []).append(v.detach().cpu() if torch.is_tensor(v) else torch.as_tensor(v))
        n_done += len(samples)
        if log_every and n_done % log_every < len(samples):
            print(f"  eval {n_done}/{len(loader.dataset)} frames  {time.time() - t0:.0f}s", flush=True)
        if max_frames and n_done >= max_frames:
            break
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def summarize(res, space):
    rre_e, rre_g, rte = so3.pose_errors(res["T_est"], res["T_gt"])
    summ = {}
    summ.update(M.accuracy_metrics(rre_e, rte))
    summ.update(M.accuracy_metrics(rre_g, rte, prefix="geo_"))
    fail = ~((rre_e < M.ACC_THR[0]) & (rte < M.ACC_THR[1]))
    summ["kde_nll"] = float(-res["kde_log_gt"].mean())
    if "exact_log_gt" in res:
        summ["exact_nll"] = float(-res["exact_log_gt"].mean())
    _, _, cov_err = M.coverage_curve(res["hpd_level"])
    summ["coverage_mae"] = cov_err
    for k in [k for k in res if k.startswith("score")]:
        summ[f"auroc_{k}"] = M.auroc(res[k], fail)
    summ["oracle_acc"] = float(100.0 * res["oracle_hit"].mean())
    summ["mean_modes"] = float(res["n_modes"].mean())
    summ["top_mode_weight"] = float(res["modes_w"][:, 0].mean())
    summ["spread_rot_deg"] = float(res["spread_rot_deg"].mean())
    summ["spread_trans_m"] = float(res["spread_trans_m"].mean())
    summ["n_failures"] = int(fail.sum())
    return summ


def save_results(res, summ, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{name}.npz", **res)
    with open(out_dir / f"{name}_metrics.json", "w") as f:
        json.dump(summ, f, indent=2)


def format_summary(summ):
    keys = ["acc", "rr", "rre_mean", "rte_mean", "rre_median", "rte_median", "geo_acc", "oracle_acc",
            "kde_nll", "exact_nll", "coverage_mae", "auroc_score", "auroc_score_inlier", "mean_modes", "n"]
    return "  ".join(f"{k}={summ[k]:.3f}" if isinstance(summ[k], float) else f"{k}={summ[k]}"
                     for k in keys if k in summ)
