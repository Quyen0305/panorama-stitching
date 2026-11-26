"""Module ước lượng ma trận Homography (3×3).

TỰ CÀI ĐẶT HOÀN TOÀN BẰNG NUMPY THUẦN:
1. Chuẩn hóa đẳng hướng Hartley (Isotropic Point Normalization).
2. Thuật toán DLT (Direct Linear Transform) giải hệ Ah = 0 bằng SVD.
3. Kiểm tra suy biến hình học (3 điểm thẳng hàng trong tập mẫu 4 điểm).
4. Adaptive RANSAC kết hợp hàm mục tiêu MSAC (M-estimator SAmple Consensus),
   tự động cập nhật số vòng lặp tối đa theo tỷ lệ inlier tốt nhất.
5. Tinh chỉnh phi tuyến Levenberg-Marquardt (tự viết ma trận Jacobian giải tích)
   trên toàn bộ tập inliers để cực tiểu hóa sai số chiếu lại (reprojection error).
6. Kiểm định hình học đầu ra (bảo toàn định hướng lồi của tứ giác ảnh sau warp).

Song song cung cấp chế độ `opencv` (cv2.findHomography) để đánh giá và so sánh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import ImplMode


@dataclass
class HomographyResult:
    """Kết quả ước lượng Homography giữa hai ảnh."""

    H: Optional[np.ndarray]  # Ma trận 3×3 ánh xạ src -> dst
    inlier_mask: np.ndarray  # Mảng bool (N,)
    num_inliers: int
    inlier_ratio: float
    reprojection_rmse: float
    iterations_run: int
    is_valid: bool
    reason: str = ""


def normalize_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Chuẩn hóa tọa độ theo Hartley (1997): tịnh tiến tâm về (0,0), chuẩn hóa
    khoảng cách trung bình tới gốc tọa độ bằng sqrt(2) giúp SVD ổn định số học.
    """
    pts = np.asarray(pts, dtype=np.float64)
    mean = np.mean(pts, axis=0)
    shifted = pts - mean
    mean_dist = float(np.mean(np.hypot(shifted[:, 0], shifted[:, 1])))
    scale = np.sqrt(2.0) / max(mean_dist, 1e-12)

    T = np.array(
        [
            [scale, 0.0, -scale * mean[0]],
            [0.0, scale, -scale * mean[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    norm_pts = shifted * scale
    return norm_pts, T


def _has_collinear_triple(pts4: np.ndarray, eps: float = 1e-3) -> bool:
    """Kiểm tra xem có 3 điểm nào trong 4 điểm mẫu gần như thẳng hàng hay không."""
    # Kiểm tra diện tích 2× của tất cả C(4,3) = 4 tam giác
    triples = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    for i, j, k in triples:
        v1 = pts4[j] - pts4[i]
        v2 = pts4[k] - pts4[i]
        area2 = abs(v1[0] * v2[1] - v1[1] * v2[0])
        if area2 < eps:
            return True
    return False


def compute_homography_dlt(
    src_pts: np.ndarray, dst_pts: np.ndarray
) -> Optional[np.ndarray]:
    """TỰ CÀI ĐẶT: Normalized Direct Linear Transform (DLT) giải ma trận H (3×3) bằng SVD."""
    n = len(src_pts)
    if n < 4:
        return None

    src_norm, T_src = normalize_points(src_pts)
    dst_norm, T_dst = normalize_points(dst_pts)

    x = src_norm[:, 0]
    y = src_norm[:, 1]
    u = dst_norm[:, 0]
    v = dst_norm[:, 1]

    A = np.zeros((2 * n, 9), dtype=np.float64)
    A[0::2, 0] = -x
    A[0::2, 1] = -y
    A[0::2, 2] = -1.0
    A[0::2, 6] = u * x
    A[0::2, 7] = u * y
    A[0::2, 8] = u

    A[1::2, 3] = -x
    A[1::2, 4] = -y
    A[1::2, 5] = -1.0
    A[1::2, 6] = v * x
    A[1::2, 7] = v * y
    A[1::2, 8] = v

    # Khi n=4, A có kích thước (8, 9) nên cần full_matrices=True để lấy hàng thứ 9 (null space);
    # khi n >= 5 (2n >= 10), full_matrices=False vẫn trả về Vt đủ (9, 9) và tránh cấp phát U lớn.
    try:
        _, _, Vt = np.linalg.svd(A, full_matrices=(A.shape[0] < A.shape[1]))
    except np.linalg.LinAlgError:
        return None

    H_norm = Vt[-1].reshape(3, 3)

    # Giải chuẩn hóa về hệ tọa độ pixel gốc: H = T_dst^{-1} @ H_norm @ T_src
    try:
        H = np.linalg.solve(T_dst, H_norm @ T_src)
    except np.linalg.LinAlgError:
        return None

    if abs(H[2, 2]) < 1e-12:
        return None
    H = H / H[2, 2]
    if not np.all(np.isfinite(H)):
        return None
    return H


def compute_reprojection_errors(
    H: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray
) -> np.ndarray:
    """Tính sai số chiếu lại Euclid (pixel) cho từng cặp điểm: ||proj(H p_i) - p'_i||_2."""
    x = src_pts[:, 0]
    y = src_pts[:, 1]
    u = H[0, 0] * x + H[0, 1] * y + H[0, 2]
    v = H[1, 0] * x + H[1, 1] * y + H[1, 2]
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]

    errors = np.full(len(src_pts), 1e9, dtype=np.float64)
    valid = w > 1e-6
    if np.any(valid):
        dx = (u[valid] / w[valid]) - dst_pts[valid, 0]
        dy = (v[valid] / w[valid]) - dst_pts[valid, 1]
        errors[valid] = np.hypot(dx, dy)
    return errors


def refine_homography_lm(
    H_init: np.ndarray,
    src_inliers: np.ndarray,
    dst_inliers: np.ndarray,
    max_iters: int = 12,
) -> np.ndarray:
    """TỰ CÀI ĐẶT: Tinh chỉnh phi tuyến Levenberg-Marquardt với Jacobian giải tích.

    Cực tiểu hóa tổng bình phương sai số chiếu lại hình học:
        min_h  sum_i || (u_i/w_i - x'_i, v_i/w_i - y'_i) ||^2
    với 8 bậc tự do h = [h00, h01, h02, h10, h11, h12, h20, h21] và h22 = 1.
    """
    if len(src_inliers) < 4:
        return H_init

    # Chuẩn hóa điểm trước khi tối ưu giúp hệ phương trình Jacobian có điều kiện số đẹp
    src_n, T_src = normalize_points(src_inliers)
    dst_n, T_dst = normalize_points(dst_inliers)
    try:
        H_n = T_dst @ H_init @ np.linalg.inv(T_src)
    except np.linalg.LinAlgError:
        return H_init
    if abs(H_n[2, 2]) < 1e-12:
        return H_init
    H_n = H_n / H_n[2, 2]

    x = src_n[:, 0]
    y = src_n[:, 1]
    xp = dst_n[:, 0]
    yp = dst_n[:, 1]
    n = len(src_n)

    def _residuals(h_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u = h_mat[0, 0] * x + h_mat[0, 1] * y + h_mat[0, 2]
        v = h_mat[1, 0] * x + h_mat[1, 1] * y + h_mat[1, 2]
        w = h_mat[2, 0] * x + h_mat[2, 1] * y + 1.0
        inv_w = 1.0 / np.where(np.abs(w) < 1e-9, 1e-9, w)
        px = u * inv_w
        py = v * inv_w
        res = np.empty(2 * n, dtype=np.float64)
        res[0::2] = px - xp
        res[1::2] = py - yp
        return res, px, py, inv_w

    lam = 1e-3
    res, px, py, inv_w = _residuals(H_n)
    cost = float(np.dot(res, res))

    J = np.zeros((2 * n, 8), dtype=np.float64)
    for _ in range(max_iters):
        # Ma trận Jacobian giải tích của hàm chiếu xạ ảnh
        J[0::2, 0] = x * inv_w
        J[0::2, 1] = y * inv_w
        J[0::2, 2] = inv_w
        J[0::2, 6] = -px * x * inv_w
        J[0::2, 7] = -px * y * inv_w

        J[1::2, 3] = x * inv_w
        J[1::2, 4] = y * inv_w
        J[1::2, 5] = inv_w
        J[1::2, 6] = -py * x * inv_w
        J[1::2, 7] = -py * y * inv_w

        JTJ = J.T @ J
        g = J.T @ res
        diag = np.maximum(np.diag(JTJ), 1e-6)
        A_lm = JTJ + lam * np.diag(diag)

        try:
            delta = -np.linalg.solve(A_lm, g)
        except np.linalg.LinAlgError:
            break

        H_cand = H_n.copy()
        H_cand[0, 0] += delta[0]
        H_cand[0, 1] += delta[1]
        H_cand[0, 2] += delta[2]
        H_cand[1, 0] += delta[3]
        H_cand[1, 1] += delta[4]
        H_cand[1, 2] += delta[5]
        H_cand[2, 0] += delta[6]
        H_cand[2, 1] += delta[7]

        res_new, px_new, py_new, inv_w_new = _residuals(H_cand)
        cost_new = float(np.dot(res_new, res_new))

        if cost_new < cost:
            improvement = (cost - cost_new) / max(cost, 1e-12)
            H_n = H_cand
            res, px, py, inv_w = res_new, px_new, py_new, inv_w_new
            cost = cost_new
            lam = max(lam * 0.3, 1e-9)
            if improvement < 1e-7:
                break
        else:
            lam = min(lam * 5.0, 1e6)

    try:
        H_refined = np.linalg.solve(T_dst, H_n @ T_src)
        if abs(H_refined[2, 2]) > 1e-12 and np.all(np.isfinite(H_refined)):
            return H_refined / H_refined[2, 2]
    except np.linalg.LinAlgError:
        pass
    return H_init


def validate_homography_geometry(
    H: np.ndarray,
    src_shape: Tuple[int, int],
    max_area_ratio: float = 25.0,
    min_area_ratio: float = 0.04,
) -> Tuple[bool, str]:
    """Kiểm định tính hợp lệ hình học của H:
    1. Không suy biến (determinant không quá nhỏ, không có điểm bị gập qua mặt phẳng vô cực w <= 0).
    2. 4 góc ảnh nguồn sau chiếu tạo thành tứ giác lồi cùng chiều (không tự cắt hình cánh bướm).
    3. Tỷ lệ diện tích biến đổi nằm trong giới hạn vật lý hợp lý.
    """
    if H is None or not np.all(np.isfinite(H)):
        return False, "Ma trận Homography chứa giá trị không hữu hạn."

    det = float(np.linalg.det(H))
    if abs(det) < 1e-7:
        return False, "Ma trận Homography suy biến (det ≈ 0)."

    h, w = src_shape[:2]
    corners = np.array(
        [[0.0, 0.0, 1.0], [w, 0.0, 1.0], [w, h, 1.0], [0.0, h, 1.0]], dtype=np.float64
    )
    proj = (H @ corners.T).T
    if np.any(proj[:, 2] <= 1e-4):
        return False, "Phép chiếu xạ ảnh đẩy góc ảnh ra sau mặt phẳng camera (w <= 0)."

    pts = proj[:, :2] / proj[:, 2:3]

    # Kiểm tra tính lồi & định hướng dương (theo chiều kim đồng hồ trong tọa độ ảnh y xuống)
    crosses = []
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        p2 = pts[(i + 2) % 4]
        v1 = p1 - p0
        v2 = p2 - p1
        crosses.append(v1[0] * v2[1] - v1[1] * v2[0])

    if not all(c > 0 for c in crosses):
        return False, "Tứ giác sau biến đổi bị lật ngược hoặc tự cắt (không lồi)."

    # Diện tích tứ giác theo công thức Shoelace
    area_dst = 0.5 * abs(
        np.dot(pts[:, 0], np.roll(pts[:, 1], -1))
        - np.dot(pts[:, 1], np.roll(pts[:, 0], -1))
    )
    area_src = float(w * h)
    ratio = area_dst / max(area_src, 1.0)
    if ratio > max_area_ratio or ratio < min_area_ratio:
        return False, f"Biến dạng diện tích quá lớn ({ratio:.2f}x)."

    return True, "OK"


def estimate_homography_ransac_custom(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    ransac_threshold: float = 4.0,
    max_iters: int = 2000,
    confidence: float = 0.995,
    min_inliers: int = 15,
    src_shape: Optional[Tuple[int, int]] = None,
    seed: int = 42,
) -> HomographyResult:
    """TỰ CÀI ĐẶT: Ước lượng Homography bền vững bằng Adaptive RANSAC + Normalized DLT + LM."""
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)
    n = len(src_pts)

    if n < 4:
        return HomographyResult(
            H=None,
            inlier_mask=np.zeros(n, dtype=bool),
            num_inliers=0,
            inlier_ratio=0.0,
            reprojection_rmse=float("inf"),
            iterations_run=0,
            is_valid=False,
            reason=f"Không đủ cặp điểm khớp ({n} < 4).",
        )

    rng = np.random.default_rng(seed)
    thresh_sq = ransac_threshold * ransac_threshold

    best_H: Optional[np.ndarray] = None
    best_mask = np.zeros(n, dtype=bool)
    best_inliers = 0
    best_msac_cost = float("inf")

    adaptive_limit = max_iters
    iters_done = 0

    for it in range(max_iters):
        if it >= adaptive_limit:
            break
        iters_done = it + 1

        idx = rng.choice(n, size=4, replace=False)
        s4 = src_pts[idx]
        d4 = dst_pts[idx]

        if _has_collinear_triple(s4, eps=1.0) or _has_collinear_triple(d4, eps=1.0):
            continue

        H_cand = compute_homography_dlt(s4, d4)
        if H_cand is None:
            continue

        errors = compute_reprojection_errors(H_cand, src_pts, dst_pts)
        mask = errors < ransac_threshold
        count = int(np.sum(mask))

        # Hàm mục tiêu MSAC (trọng số mượt hơn đếm inlier thuần túy)
        err_sq = np.minimum(errors * errors, thresh_sq)
        msac_cost = float(np.sum(err_sq))

        if (count > best_inliers) or (count == best_inliers and msac_cost < best_msac_cost):
            best_inliers = count
            best_msac_cost = msac_cost
            best_mask = mask
            best_H = H_cand

            # Cập nhật số vòng lặp thích nghi: N = log(1 - p) / log(1 - w^4)
            w_ratio = best_inliers / float(n)
            if w_ratio > 0.05:
                prob_all_inliers = min(w_ratio ** 4, 1.0 - 1e-12)
                denom = np.log(max(1.0 - prob_all_inliers, 1e-12))
                if denom < 0:
                    est_iters = int(np.ceil(np.log(1.0 - confidence) / denom))
                    # Luôn chạy ít nhất 50 vòng để tránh dừng quá sớm
                    adaptive_limit = min(max_iters, max(50, est_iters))

    if best_H is None or best_inliers < 4:
        return HomographyResult(
            H=None,
            inlier_mask=np.zeros(n, dtype=bool),
            num_inliers=0,
            inlier_ratio=0.0,
            reprojection_rmse=float("inf"),
            iterations_run=iters_done,
            is_valid=False,
            reason="RANSAC không tìm được mô hình Homography phù hợp.",
        )

    # Bước tinh chỉnh 1: Tính lại Normalized DLT trên toàn bộ tập Inliers
    H_dlt = compute_homography_dlt(src_pts[best_mask], dst_pts[best_mask])
    if H_dlt is not None:
        # Bước tinh chỉnh 2: Tối ưu phi tuyến Levenberg-Marquardt
        H_ref = refine_homography_lm(H_dlt, src_pts[best_mask], dst_pts[best_mask])
        errs_ref = compute_reprojection_errors(H_ref, src_pts, dst_pts)
        mask_ref = errs_ref < ransac_threshold
        if int(np.sum(mask_ref)) >= best_inliers:
            best_H = H_ref
            best_mask = mask_ref
            best_inliers = int(np.sum(mask_ref))
        else:
            errs_dlt = compute_reprojection_errors(H_dlt, src_pts, dst_pts)
            mask_dlt = errs_dlt < ransac_threshold
            if int(np.sum(mask_dlt)) >= best_inliers:
                best_H = H_dlt
                best_mask = mask_dlt
                best_inliers = int(np.sum(mask_dlt))

    final_errs = compute_reprojection_errors(best_H, src_pts, dst_pts)
    inlier_errs = final_errs[best_mask]
    rmse = float(np.sqrt(np.mean(inlier_errs ** 2))) if len(inlier_errs) > 0 else float("inf")
    inlier_ratio = best_inliers / float(n)

    is_valid = best_inliers >= min_inliers
    reason = "OK" if is_valid else f"Số inliers chưa đạt ngưỡng ({best_inliers} < {min_inliers})."

    if is_valid and src_shape is not None:
        geom_ok, geom_reason = validate_homography_geometry(best_H, src_shape)
        if not geom_ok:
            is_valid = False
            reason = geom_reason

    return HomographyResult(
        H=best_H,
        inlier_mask=best_mask,
        num_inliers=best_inliers,
        inlier_ratio=inlier_ratio,
        reprojection_rmse=rmse,
        iterations_run=iters_done,
        is_valid=is_valid,
        reason=reason,
    )


def estimate_homography_opencv(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    ransac_threshold: float = 4.0,
    max_iters: int = 2000,
    confidence: float = 0.995,
    min_inliers: int = 15,
    src_shape: Optional[Tuple[int, int]] = None,
) -> HomographyResult:
    """Chế độ đối chiếu dùng `cv2.findHomography` (RANSAC của OpenCV)."""
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)
    n = len(src_pts)

    if n < 4:
        return HomographyResult(
            H=None,
            inlier_mask=np.zeros(n, dtype=bool),
            num_inliers=0,
            inlier_ratio=0.0,
            reprojection_rmse=float("inf"),
            iterations_run=0,
            is_valid=False,
            reason=f"Không đủ cặp điểm khớp ({n} < 4).",
        )

    H, status = cv2.findHomography(
        src_pts.reshape(-1, 1, 2),
        dst_pts.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=max_iters,
        confidence=confidence,
    )

    if H is None or status is None:
        return HomographyResult(
            H=None,
            inlier_mask=np.zeros(n, dtype=bool),
            num_inliers=0,
            inlier_ratio=0.0,
            reprojection_rmse=float("inf"),
            iterations_run=max_iters,
            is_valid=False,
            reason="OpenCV findHomography không hội tụ.",
        )

    mask = status.ravel().astype(bool)
    num_inliers = int(np.sum(mask))
    errs = compute_reprojection_errors(H, src_pts, dst_pts)
    inlier_errs = errs[mask]
    rmse = float(np.sqrt(np.mean(inlier_errs ** 2))) if len(inlier_errs) > 0 else float("inf")
    inlier_ratio = num_inliers / float(n)

    is_valid = num_inliers >= min_inliers
    reason = "OK" if is_valid else f"Số inliers chưa đạt ngưỡng ({num_inliers} < {min_inliers})."

    if is_valid and src_shape is not None:
        geom_ok, geom_reason = validate_homography_geometry(H, src_shape)
        if not geom_ok:
            is_valid = False
            reason = geom_reason

    return HomographyResult(
        H=H,
        inlier_mask=mask,
        num_inliers=num_inliers,
        inlier_ratio=inlier_ratio,
        reprojection_rmse=rmse,
        iterations_run=max_iters,
        is_valid=is_valid,
        reason=reason,
    )


def estimate_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    mode: ImplMode = "custom",
    ransac_threshold: float = 4.0,
    max_iters: int = 2000,
    confidence: float = 0.995,
    min_inliers: int = 15,
    src_shape: Optional[Tuple[int, int]] = None,
    seed: int = 42,
) -> HomographyResult:
    """Hàm điều phối gọi chế độ tự cài đặt hoặc OpenCV."""
    if mode == "custom":
        return estimate_homography_ransac_custom(
            src_pts,
            dst_pts,
            ransac_threshold=ransac_threshold,
            max_iters=max_iters,
            confidence=confidence,
            min_inliers=min_inliers,
            src_shape=src_shape,
            seed=seed,
        )
    elif mode == "opencv":
        return estimate_homography_opencv(
            src_pts,
            dst_pts,
            ransac_threshold=ransac_threshold,
            max_iters=max_iters,
            confidence=confidence,
            min_inliers=min_inliers,
            src_shape=src_shape,
        )
    raise ValueError(f"Chế độ homography không hợp lệ: {mode}")
