"""CoFiI2P + PnP-RANSAC baseline, reproducing third_party/CoFiI2P/evaluation/eval_all.py.

Besides the pose it returns the two uncertainty proxies used in the proposal:
  * inlier ratio of RANSAC (higher = more confident),
  * a first-order covariance sigma^2 (J^T J)^-1 from the PnP reprojection Jacobian on inliers,
    expressed in the same tangent coordinates as the heads (camera-frame rotation rad, metres).
"""
import cv2
import numpy as np
import torch

from i2pflow.geometry import so3
from i2pflow.models.backbone import load_cofii2p

_FAIL_STD = (np.pi, 10.0)  # rad, m used when PnP fails
_FAIL_COV = np.diag([_FAIL_STD[0] ** 2] * 3 + [_FAIL_STD[1] ** 2] * 3)
_MIN_VAR = 1e-8  # floor on covariance eigenvalues (rad^2 / m^2); keeps float32 Cholesky stable


def _well_conditioned(cov):
    """Symmetric PD version of `cov`: eigenvalues clipped to [_MIN_VAR, failure variance].

    Degenerate inlier sets (few or near-collinear points, zero residual) give singular or indefinite J^T J
    inverses, which crash the Cholesky used for sampling."""
    if not np.all(np.isfinite(cov)):
        return _FAIL_COV.copy()
    w, V = np.linalg.eigh(0.5 * (cov + cov.T))
    w = np.clip(w, _MIN_VAR, _FAIL_STD[1] ** 2)
    return (V * w) @ V.T


class CoFiI2PPnP:
    def __init__(self, opt, ckpt_path, device, iterations=10000, reproj_error=8.0):
        self.net = load_cofii2p(opt, ckpt_path, device)
        self.iterations = iterations
        self.reproj_error = reproj_error

    @torch.no_grad()
    def correspondences(self, s):
        out = self.net(s["pc_data_dict"], s["img"][None], s["fine_center_kpt_coors"], s["fine_xy_coors"],
                       s["fine_pc_inline_index"], "test")
        fine_img_feature_patch, fine_pc_inline_feature, fine_center_xy, coarse_pc_points = out[4:8]
        fine_pc_inline_feature = fine_pc_inline_feature.unsqueeze(-1)
        dist = torch.cosine_similarity(fine_img_feature_patch.unsqueeze(-1), fine_pc_inline_feature.unsqueeze(-2))
        predict_index = torch.argmax(torch.squeeze(dist), dim=1)
        fine_xy = fine_center_xy - 2
        fine_xy[0] = fine_xy[0] + predict_index // 4
        fine_xy[1] = fine_xy[1] + predict_index % 4
        return fine_xy.T.cpu().numpy().astype(np.float64), coarse_pc_points.cpu().numpy().astype(np.float64)

    def __call__(self, s):
        K = s["K"].cpu().numpy().astype(np.float64)
        img_pts, obj_pts = self.correspondences(s)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=obj_pts, imagePoints=img_pts, cameraMatrix=K,
                                                     distCoeffs=None, iterationsCount=self.iterations,
                                                     reprojectionError=self.reproj_error)
        n = len(obj_pts)
        if not ok or inliers is None or len(inliers) < 4:
            cov = _FAIL_COV.copy()
            return {"T": np.eye(4), "success": False, "inlier_ratio": 0.0, "n_corr": n, "cov": cov}
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec[:, 0]
        inl = inliers[:, 0]
        return {"T": T, "success": True, "inlier_ratio": len(inl) / n, "n_corr": n,
                "cov": self._covariance(obj_pts[inl], img_pts[inl], rvec, tvec, K)}

    @staticmethod
    def _covariance(obj, img, rvec, tvec, K):
        proj, J = cv2.projectPoints(obj, rvec, tvec, K, None)
        J = J[:, :6]  # d(uv)/d(rvec, tvec)
        res = (proj[:, 0] - img).reshape(-1)
        dof = max(res.size - 6, 1)
        sigma2 = float(res @ res) / dof
        try:
            cov_param = sigma2 * np.linalg.inv(J.T @ J)
        except np.linalg.LinAlgError:
            return _FAIL_COV.copy()
        # rvec perturbation -> camera-frame (left) rotation perturbation: a = J_l(rvec) d_rvec
        Jl = so3.left_jacobian(torch.from_numpy(rvec[:, 0])).numpy()
        A = np.zeros((6, 6))
        A[:3, :3] = Jl
        A[3:, 3:] = np.eye(3)
        return _well_conditioned(A @ cov_param @ A.T)
