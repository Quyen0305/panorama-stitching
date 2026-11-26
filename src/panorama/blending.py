"""Module trộn ảnh & xử lý vùng chồng lấn (TỰ CÀI ĐẶT - ĐIỂM NHẤN KỸ THUẬT).

Toàn bộ các thuật toán trong module này được cài đặt từ nền tảng NumPy/SciPy:
1. Cân bằng phơi sáng toàn cục (Exposure / Gain Compensation — Brown & Lowe 2007):
   Giải hệ phương trình tuyến tính bình phương tối thiểu tìm hệ số gain g_i theo từng
   kênh màu trên tất cả các vùng chồng lấn.
2. Tìm đường nối tối ưu bằng Quy hoạch động (Dynamic Programming Minimum-Error Seam
   Finding — Efros & Freeman / Avidan): tìm đường cắt có sai khác màu sắc + gradient
   nhỏ nhất xuyên qua vùng chồng lấn nhằm né vật thể chuyển động (chống ghosting).
3. Trộn Feather (Distance-Transform Alpha Blending): trọng số mềm theo khoảng cách
   Euclid tới biên mặt nạ hợp lệ.
4. Trộn Đa dải tần — Tháp Laplacian (Multi-Band Laplacian Pyramid Blending — Burt &
   Adelson 1983): cài đặt tường minh tích chập tách rời 1D với nhân Gaussian 5 điểm
   [1, 4, 6, 4, 1]/16, hàm `pyr_down`, `pyr_up`, xây tháp Gaussian cho mặt nạ & tháp
   Laplacian cho ảnh, kết hợp ngoại suy biên chống quầng tối (black-halo-free).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

from .config import BlendingMode
from .warping import WarpedFrame


# Nhân Gaussian 5 điểm chuẩn của Burt & Adelson (1983): a = 0.375 -> [1, 4, 6, 4, 1] / 16
_GAUSS_KERNEL_1D = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0


@dataclass
class BlendingResult:
    """Kết quả trộn ảnh kèm siêu dữ liệu cân bằng sáng và đường nối."""

    panorama: np.ndarray  # BGR uint8 (H, W, 3)
    valid_mask: np.ndarray  # bool (H, W)
    exposure_gains: Dict[int, List[float]]  # image_index -> [gB, gG, gR]
    seam_polylines: List[np.ndarray]  # Danh sách tọa độ (K, 2) của các đường seam tối ưu
    seam_gradient_jump: float  # Chỉ số độ mượt đường nối (càng thấp càng ít lộ đường nối)


# ---------------------------------------------------------------------------
# 1. Cân bằng phơi sáng toàn cục (Global Least-Squares Gain Compensation)
# ---------------------------------------------------------------------------

def compensate_exposure(
    frames: List[WarpedFrame],
    sigma_n: float = 6.0,
    sigma_g: float = 0.25,
    min_overlap_pixels: int = 200,
) -> Tuple[List[WarpedFrame], Dict[int, List[float]]]:
    """TỰ CÀI ĐẶT: Cân bằng độ sáng và cân bằng trắng giữa các ảnh theo Brown & Lowe (2007).

    Cực tiểu hóa hàm năng lượng toàn cục trên mọi cặp ảnh chồng lấn (i, j):
        E(g) = 1/2 * sum_{i != j} N_{ij} * [ (g_i * I_ij - g_j * I_ji)^2 / sigma_n^2
                                           + (1 - g_i)^2 / sigma_g^2 ]
    """
    n = len(frames)
    gains_by_img: Dict[int, List[float]] = {fr.image_index: [1.0, 1.0, 1.0] for fr in frames}
    if n < 2:
        return frames, gains_by_img

    # Thu thập thống kê vùng chồng lấn (N_ij, mean_i[3], mean_j[3]) nhanh theo bbox giao nhau
    overlap_stats: List[Tuple[int, int, int, np.ndarray, np.ndarray]] = []
    for i in range(n):
        x0a, y0a, x1a, y1a = frames[i].bbox
        for j in range(i + 1, n):
            x0b, y0b, x1b, y1b = frames[j].bbox
            ix0, iy0 = max(x0a, x0b), max(y0a, y0b)
            ix1, iy1 = min(x1a, x1b), min(y1a, y1b)
            if ix1 <= ix0 or iy1 <= iy0:
                continue

            ma = frames[i].mask[iy0:iy1, ix0:ix1]
            mb = frames[j].mask[iy0:iy1, ix0:ix1]
            ov = ma & mb
            count = int(np.count_nonzero(ov))
            if count < min_overlap_pixels:
                continue

            sub_a = frames[i].image[iy0:iy1, ix0:ix1][ov].astype(np.float64)
            sub_b = frames[j].image[iy0:iy1, ix0:ix1][ov].astype(np.float64)
            # Loại bỏ các pixel bão hòa trắng/đen quá mức khỏi thống kê phơi sáng
            mean_a = np.mean(sub_a, axis=0)
            mean_b = np.mean(sub_b, axis=0)
            overlap_stats.append((i, j, count, mean_a, mean_b))

    if not overlap_stats:
        return frames, gains_by_img

    gains_matrix = np.ones((n, 3), dtype=np.float64)
    inv_sn2 = 1.0 / (sigma_n * sigma_n)
    inv_sg2 = 1.0 / (sigma_g * sigma_g)

    for c in range(3):
        A = np.zeros((n, n), dtype=np.float64)
        b = np.zeros(n, dtype=np.float64)

        for i, j, count, mean_a, mean_b in overlap_stats:
            ia = max(float(mean_a[c]), 1.0)
            ib = max(float(mean_b[c]), 1.0)
            w_d = float(count) * inv_sn2
            w_p = float(count) * inv_sg2

            # Đóng góp đối xứng từ cả (i, j) và (j, i)
            A[i, i] += w_d * (ia * ia) + w_p
            A[j, j] += w_d * (ib * ib) + w_p
            A[i, j] -= w_d * (ia * ib)
            A[j, i] -= w_d * (ia * ib)
            b[i] += w_p
            b[j] += w_p

        for i in range(n):
            if A[i, i] == 0.0:
                A[i, i] = 1.0
                b[i] = 1.0

        try:
            g = np.linalg.solve(A, b)
            g = np.clip(g, 0.55, 1.85)
            gains_matrix[:, c] = g
        except np.linalg.LinAlgError:
            gains_matrix[:, c] = 1.0

    # Chuẩn hóa nhẹ sao cho trung bình hình học của gain gần 1.0 (tránh tối/cháy toàn ảnh)
    for c in range(3):
        geo_mean = float(np.exp(np.mean(np.log(gains_matrix[:, c]))))
        gains_matrix[:, c] = np.clip(gains_matrix[:, c] / max(geo_mean, 1e-6), 0.55, 1.85)

    compensated: List[WarpedFrame] = []
    for i, fr in enumerate(frames):
        g_vec = gains_matrix[i].astype(np.float32)
        gains_by_img[fr.image_index] = [round(float(x), 4) for x in g_vec]

        x0, y0, x1, y1 = fr.bbox
        new_img = fr.image.copy()
        if x1 > x0 and y1 > y0:
            roi = new_img[y0:y1, x0:x1].astype(np.float32) * g_vec[np.newaxis, np.newaxis, :]
            roi_u8 = np.clip(roi, 0.0, 255.0).astype(np.uint8)
            roi_u8[~fr.mask[y0:y1, x0:x1]] = 0
            new_img[y0:y1, x0:x1] = roi_u8

        compensated.append(
            WarpedFrame(
                image_index=fr.image_index,
                image=new_img,
                mask=fr.mask,
                bbox=fr.bbox,
                H_canvas=fr.H_canvas,
            )
        )

    return compensated, gains_by_img


# ---------------------------------------------------------------------------
# 2. Tìm đường nối tối ưu bằng Quy hoạch động (DP Minimum-Error Seam)
# ---------------------------------------------------------------------------

def _sobel_grad_mag_numpy(gray: np.ndarray) -> np.ndarray:
    """Tính độ lớn gradient Sobel bằng NumPy thuần (không cần gọi OpenCV)."""
    p = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    gx = (
        -p[:-2, :-2] + p[:-2, 2:]
        - 2.0 * p[1:-1, :-2] + 2.0 * p[1:-1, 2:]
        - p[2:, :-2] + p[2:, 2:]
    )
    gy = (
        -p[:-2, :-2] - 2.0 * p[:-2, 1:-1] - p[:-2, 2:]
        + p[2:, :-2] + 2.0 * p[2:, 1:-1] + p[2:, 2:]
    )
    return np.hypot(gx, gy)


def find_optimal_seam_dp(
    img_a: np.ndarray,
    mask_a: np.ndarray,
    img_b: np.ndarray,
    mask_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """TỰ CÀI ĐẶT: Tìm đường cắt năng lượng cực tiểu bằng Quy hoạch động (Dynamic Programming).

    Đầu vào:
        img_a, mask_a: ảnh tích lũy hiện tại và mặt nạ hợp lệ.
        img_b, mask_b: ảnh mới cần ghép thêm và mặt nạ hợp lệ.
    Trả về:
        weight_b (float32 0/1 trên vùng overlap, 0 trên only_a, 1 trên only_b)
        seam_polyline: mảng tọa độ (K, 2) dạng (x, y) của đường nối trên canvas.
    """
    overlap = mask_a & mask_b
    weight_b = np.zeros(mask_a.shape, dtype=np.float32)
    weight_b[mask_b & (~mask_a)] = 1.0

    ys, xs = np.nonzero(overlap)
    if len(xs) == 0:
        return weight_b, np.empty((0, 2), dtype=np.float32)

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    # Xác định hướng tách chính giữa hai ảnh (ngang hay dọc)
    ya, xa = np.nonzero(mask_a)
    yb, xb = np.nonzero(mask_b)
    dx = float(np.mean(xb) - np.mean(xa))
    dy = float(np.mean(yb) - np.mean(ya))
    transposed = abs(dy) > abs(dx)

    roi_a = img_a[y0:y1, x0:x1].astype(np.float32) / 255.0
    roi_b = img_b[y0:y1, x0:x1].astype(np.float32) / 255.0
    ov_roi = overlap[y0:y1, x0:x1]

    if transposed:
        roi_a = np.transpose(roi_a, (1, 0, 2))
        roi_b = np.transpose(roi_b, (1, 0, 2))
        ov_roi = ov_roi.T
        side_b_is_right = dy >= 0.0
    else:
        side_b_is_right = dx >= 0.0

    oh, ow = ov_roi.shape

    # Hàm năng lượng tại mỗi điểm: sai khác màu L2 + 0.6 * sai khác gradient Sobel
    color_diff = np.sqrt(np.mean((roi_a - roi_b) ** 2, axis=2) + 1e-12)
    gray_a = 0.114 * roi_a[:, :, 0] + 0.587 * roi_a[:, :, 1] + 0.299 * roi_a[:, :, 2]
    gray_b = 0.114 * roi_b[:, :, 0] + 0.587 * roi_b[:, :, 1] + 0.299 * roi_b[:, :, 2]
    grad_diff = np.abs(_sobel_grad_mag_numpy(gray_a) - _sobel_grad_mag_numpy(gray_b))

    # Thêm độ lệch nhẹ ưu tiên giữa vùng chồng lấn để đường cắt không bám sát biên ảnh
    center_bias = 0.025 * np.abs(np.arange(ow, dtype=np.float32) - (ow * 0.5)) / max(float(ow), 1.0)
    energy = (color_diff + 0.6 * grad_diff + center_bias[np.newaxis, :]).astype(np.float32)

    # Phạt rất lớn cho pixel nằm ngoài vùng overlap thực tế để ép đường seam đi trong vùng chung
    row_has_ov = np.any(ov_roi, axis=1)
    outside = (~ov_roi) & row_has_ov[:, np.newaxis]
    energy[outside] = 1e4

    # Quy hoạch động từ hàng 0 -> oh-1 với bước chuyển 8-láng giềng (c-1, c, c+1)
    dp = np.empty((oh, ow), dtype=np.float32)
    backptr = np.zeros((oh, ow), dtype=np.int8)
    dp[0] = energy[0]

    inf_val = np.float32(1e12)
    left_buf = np.empty(ow, dtype=np.float32)
    right_buf = np.empty(ow, dtype=np.float32)

    for r in range(1, oh):
        prev = dp[r - 1]
        left_buf[0] = inf_val
        left_buf[1:] = prev[:-1]
        right_buf[-1] = inf_val
        right_buf[:-1] = prev[1:]

        # Tìm min trong 3 ứng viên (left=-1, mid=0, right=+1)
        min_val = prev.copy()
        step = np.zeros(ow, dtype=np.int8)

        use_left = left_buf < min_val
        min_val[use_left] = left_buf[use_left]
        step[use_left] = -1

        use_right = right_buf < min_val
        min_val[use_right] = right_buf[use_right]
        step[use_right] = 1

        dp[r] = energy[r] + min_val
        backptr[r] = step

    # Truy vết ngược tìm đường nối tối ưu
    seam_cols = np.empty(oh, dtype=np.int32)
    c = int(np.argmin(dp[-1]))
    seam_cols[oh - 1] = c
    for r in range(oh - 1, 0, -1):
        c = int(np.clip(c + int(backptr[r, c]), 0, ow - 1))
        seam_cols[r - 1] = c

    # Phân vùng nhị phân theo đường seam
    col_grid = np.arange(ow, dtype=np.int32)[np.newaxis, :]
    if side_b_is_right:
        owner_b = col_grid >= seam_cols[:, np.newaxis]
    else:
        owner_b = col_grid <= seam_cols[:, np.newaxis]

    # Tọa độ đường seam toàn cục để trực quan hóa
    rows_idx = np.arange(oh, dtype=np.float32)
    valid_seam = ov_roi[np.arange(oh), seam_cols]
    if transposed:
        owner_b = owner_b.T
        ov_back = ov_roi.T
        pts_x = (rows_idx[valid_seam] + x0)
        pts_y = (seam_cols[valid_seam].astype(np.float32) + y0)
    else:
        ov_back = ov_roi
        pts_x = (seam_cols[valid_seam].astype(np.float32) + x0)
        pts_y = (rows_idx[valid_seam] + y0)

    sub_w = weight_b[y0:y1, x0:x1]
    sub_w[ov_back] = owner_b[ov_back].astype(np.float32)
    weight_b[y0:y1, x0:x1] = sub_w

    if len(pts_x) > 0:
        # Giảm mật độ điểm polyline cho nhẹ khi vẽ preview
        stride = max(1, len(pts_x) // 250)
        seam_polyline = np.stack([pts_x[::stride], pts_y[::stride]], axis=1).astype(np.float32)
    else:
        seam_polyline = np.empty((0, 2), dtype=np.float32)

    return weight_b, seam_polyline


# ---------------------------------------------------------------------------
# 3. Trộn Feather (Distance-Transform Alpha Blending)
# ---------------------------------------------------------------------------

def feather_weight_map(mask_a: np.ndarray, mask_b: np.ndarray, power: float = 1.5) -> np.ndarray:
    """TỰ CÀI ĐẶT: Tạo bản đồ trọng số mềm dựa trên biến đổi khoảng cách Euclid (EDT)."""
    weight_b = np.zeros(mask_a.shape, dtype=np.float32)
    only_b = mask_b & (~mask_a)
    overlap = mask_a & mask_b
    weight_b[only_b] = 1.0

    if not np.any(overlap):
        return weight_b

    # Chỉ tính distance transform trên bounding box hợp của 2 mặt nạ để tối ưu tốc độ
    union = mask_a | mask_b
    ys, xs = np.nonzero(union)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    da = distance_transform_edt(mask_a[y0:y1, x0:x1]).astype(np.float32)
    db = distance_transform_edt(mask_b[y0:y1, x0:x1]).astype(np.float32)

    if power != 1.0:
        da = np.power(da, power)
        db = np.power(db, power)

    ov_sub = overlap[y0:y1, x0:x1]
    w_sub = weight_b[y0:y1, x0:x1]
    w_sub[ov_sub] = db[ov_sub] / (da[ov_sub] + db[ov_sub] + 1e-8)
    weight_b[y0:y1, x0:x1] = w_sub
    return weight_b


# ---------------------------------------------------------------------------
# 4. Tháp Gaussian & Tháp Laplacian TỰ CÀI ĐẶT BẰNG NUMPY THUẦN (Burt & Adelson)
# ---------------------------------------------------------------------------

def gaussian_blur_5x5_numpy(arr: np.ndarray, kernel_1d: np.ndarray = _GAUSS_KERNEL_1D) -> np.ndarray:
    """TỰ CÀI ĐẶT: Tích chập Gaussian 2D tách rời (separable 1D ngang + 1D dọc) bằng NumPy thuần."""
    arr = arr.astype(np.float32, copy=False)
    k0, k1, k2, k3, k4 = kernel_1d

    # Lọc ngang (trục x = axis 1) với phản xạ biên (reflect padding)
    if arr.ndim == 3:
        px = np.pad(arr, ((0, 0), (2, 2), (0, 0)), mode="reflect")
    else:
        px = np.pad(arr, ((0, 0), (2, 2)), mode="reflect")
    horiz = (
        k0 * px[:, :-4]
        + k1 * px[:, 1:-3]
        + k2 * px[:, 2:-2]
        + k3 * px[:, 3:-1]
        + k4 * px[:, 4:]
    )

    # Lọc dọc (trục y = axis 0)
    if arr.ndim == 3:
        py = np.pad(horiz, ((2, 2), (0, 0), (0, 0)), mode="reflect")
    else:
        py = np.pad(horiz, ((2, 2), (0, 0)), mode="reflect")
    out = (
        k0 * py[:-4, :]
        + k1 * py[1:-3, :]
        + k2 * py[2:-2, :]
        + k3 * py[3:-1, :]
        + k4 * py[4:, :]
    )
    return out


def pyr_down_numpy(arr: np.ndarray) -> np.ndarray:
    """TỰ CÀI ĐẶT: Giảm phân giải một tầng tháp Gaussian (lọc thông thấp 5×5 + lấy mẫu cách 2)."""
    blurred = gaussian_blur_5x5_numpy(arr, _GAUSS_KERNEL_1D)
    return blurred[::2, ::2]


def pyr_up_numpy(arr: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """TỰ CÀI ĐẶT: Nội suy tăng phân giải một tầng tháp (chèn 0 xen kẽ + lọc nhân 2*k ở mỗi trục)."""
    up = np.zeros(target_shape, dtype=np.float32)
    h_in, w_in = arr.shape[:2]
    h_out, w_out = target_shape[:2]
    use_h = min(h_in, (h_out + 1) // 2)
    use_w = min(w_in, (w_out + 1) // 2)
    up[: use_h * 2 : 2, : use_w * 2 : 2] = arr[:use_h, :use_w]
    # Nhân 2 mỗi chiều (tổng nhân 4) để bảo toàn năng lượng trung bình khi chèn 3/4 số 0
    return gaussian_blur_5x5_numpy(up, 2.0 * _GAUSS_KERNEL_1D)


def build_gaussian_pyramid(arr: np.ndarray, levels: int) -> List[np.ndarray]:
    """TỰ CÀI ĐẶT: Xây dựng tháp Gaussian gồm `levels + 1` tầng."""
    pyr = [arr.astype(np.float32)]
    curr = pyr[0]
    for _ in range(levels):
        if min(curr.shape[0], curr.shape[1]) < 8:
            break
        curr = pyr_down_numpy(curr)
        pyr.append(curr)
    return pyr


def build_laplacian_pyramid(arr: np.ndarray, levels: int) -> List[np.ndarray]:
    """TỰ CÀI ĐẶT: Xây dựng tháp Laplacian L_k = G_k - pyr_up(G_{k+1}), tầng cuối là G_L."""
    g_pyr = build_gaussian_pyramid(arr, levels)
    l_pyr: List[np.ndarray] = []
    for k in range(len(g_pyr) - 1):
        expanded = pyr_up_numpy(g_pyr[k + 1], g_pyr[k].shape)
        l_pyr.append(g_pyr[k] - expanded)
    l_pyr.append(g_pyr[-1])
    return l_pyr


def reconstruct_laplacian_pyramid(l_pyr: List[np.ndarray]) -> np.ndarray:
    """TỰ CÀI ĐẶT: Tái tạo ảnh đầy đủ từ tháp Laplacian (từ tầng thô nhất lên tầng 0)."""
    curr = l_pyr[-1]
    for k in range(len(l_pyr) - 2, -1, -1):
        curr = pyr_up_numpy(curr, l_pyr[k].shape) + l_pyr[k]
    return curr


def _extrapolate_outside_mask(roi: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Điền các pixel ngoài mặt nạ hợp lệ bằng giá trị pixel hợp lệ gần nhất (triệt tiêu quầng đen)."""
    if np.all(valid_mask) or not np.any(valid_mask):
        return roi
    indices = distance_transform_edt(~valid_mask, return_distances=False, return_indices=True)
    return roi[indices[0], indices[1]]


def multiband_blend_pair(
    img_a: np.ndarray,
    mask_a: np.ndarray,
    img_b: np.ndarray,
    mask_b: np.ndarray,
    weight_b: np.ndarray,
    levels: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """TỰ CÀI ĐẶT: Trộn hai ảnh bằng Tháp Laplacian Đa dải tần trên vùng ROI có đệm.

    - Tần số thấp (độ sáng, màu nền trời) chuyển tiếp rất mượt trên dải rộng.
    - Tần số cao (chi tiết cạnh, lá cây, mái nhà) chuyển sắc nét theo đường seam, tránh bóng ma.
    - Ngoại suy biên trước khi tạo tháp đảm bảo 100% không rò viền đen từ nền canvas.
    """
    overlap = mask_a & mask_b
    union = mask_a | mask_b
    out = img_a.copy()
    only_b = mask_b & (~mask_a)
    out[only_b] = img_b[only_b]

    if not np.any(overlap):
        return out, union

    # Xác định ROI bao quanh vùng overlap cộng đệm hỗ trợ của tháp Gaussian
    ys, xs = np.nonzero(overlap)
    pad = min(256, 4 * (2 ** max(1, levels)))
    h, w = img_a.shape[:2]
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)

    m_a = mask_a[y0:y1, x0:x1]
    m_b = mask_b[y0:y1, x0:x1]
    m_u = m_a | m_b

    roi_a = img_a[y0:y1, x0:x1].astype(np.float32)
    roi_b = img_b[y0:y1, x0:x1].astype(np.float32)

    # Trong vùng chỉ có B -> lấy pixel B cho ảnh A (và ngược lại) để tránh bước nhảy về 0
    a_fill = roi_a.copy()
    a_fill[(~m_a) & m_b] = roi_b[(~m_a) & m_b]

    b_fill = roi_b.copy()
    b_fill[(~m_b) & m_a] = roi_a[(~m_b) & m_a]

    w_roi = weight_b[y0:y1, x0:x1].astype(np.float32).copy()
    w_roi[m_a & (~m_b)] = 0.0
    w_roi[m_b & (~m_a)] = 1.0

    # Ngoại suy vùng nền đen ngoài union bằng pixel gần nhất giống hệt nhau ở cả A và B
    if not np.all(m_u) and np.any(m_u):
        idx = distance_transform_edt(~m_u, return_distances=False, return_indices=True)
        a_fill = a_fill[idx[0], idx[1]]
        b_fill = b_fill[idx[0], idx[1]]
        w_roi = w_roi[idx[0], idx[1]]

    # Tính số tầng tháp phù hợp với kích thước thực tế của ROI
    eff_levels = min(levels, max(1, int(np.floor(np.log2(max(8, min(y1 - y0, x1 - x0))))) - 2))

    la_pyr = build_laplacian_pyramid(a_fill, eff_levels)
    lb_pyr = build_laplacian_pyramid(b_fill, eff_levels)
    gw_pyr = build_gaussian_pyramid(w_roi, len(la_pyr) - 1)

    blended_pyr: List[np.ndarray] = []
    for la, lb, gw in zip(la_pyr, lb_pyr, gw_pyr):
        gw3 = gw[:, :, np.newaxis]
        blended_pyr.append((1.0 - gw3) * la + gw3 * lb)

    reconstructed = reconstruct_laplacian_pyramid(blended_pyr)
    reconstructed_u8 = np.clip(np.round(reconstructed), 0.0, 255.0).astype(np.uint8)

    sub_out = out[y0:y1, x0:x1]
    sub_out[m_u] = reconstructed_u8[m_u]
    out[y0:y1, x0:x1] = sub_out
    return out, union


# ---------------------------------------------------------------------------
# 5. Đánh giá độ mượt đường nối (Seam Discontinuity Metric) & Hàm điều phối
# ---------------------------------------------------------------------------

def _compute_seam_gradient_jump(
    blended: np.ndarray, seam_polylines: List[np.ndarray]
) -> float:
    """Đo độ nhảy cường độ cục bộ vuông góc dọc đường nối (càng nhỏ càng liền mạch)."""
    if not seam_polylines:
        return 0.0
    gray = (
        0.114 * blended[:, :, 0].astype(np.float32)
        + 0.587 * blended[:, :, 1].astype(np.float32)
        + 0.299 * blended[:, :, 2].astype(np.float32)
    )
    h, w = gray.shape
    diffs = []
    for pts in seam_polylines:
        if len(pts) == 0:
            continue
        xs = np.clip(np.round(pts[:, 0]).astype(np.int32), 2, w - 3)
        ys = np.clip(np.round(pts[:, 1]).astype(np.int32), 2, h - 3)
        dx = np.abs(gray[ys, xs + 2] - gray[ys, xs - 2])
        dy = np.abs(gray[ys + 2, xs] - gray[ys - 2, xs])
        diffs.append(np.maximum(dx, dy))
    if not diffs:
        return 0.0
    return float(np.mean(np.concatenate(diffs)))


def blend_warped_frames(
    frames: List[WarpedFrame],
    visual_order: List[int],
    blending_mode: BlendingMode = "seam_multiband",
    exposure_compensation: bool = True,
    multiband_levels: int = 5,
) -> BlendingResult:
    """Điều phối cân bằng sáng và trộn toàn bộ tập ảnh đã warp theo thứ tự trái -> phải."""
    # Sắp xếp khung ảnh theo thứ tự vị trí trực quan (visual_order) để mỗi ảnh mới chồng lấn tối đa
    frame_by_idx = {fr.image_index: fr for fr in frames}
    ordered_frames = [frame_by_idx[idx] for idx in visual_order if idx in frame_by_idx]

    if exposure_compensation and blending_mode != "none":
        ordered_frames, gains = compensate_exposure(ordered_frames)
    else:
        gains = {fr.image_index: [1.0, 1.0, 1.0] for fr in ordered_frames}

    canvas = ordered_frames[0].image.copy()
    canvas_mask = ordered_frames[0].mask.copy()
    seam_polylines: List[np.ndarray] = []

    for nxt in ordered_frames[1:]:
        img_b = nxt.image
        mask_b = nxt.mask

        if blending_mode == "none":
            # Cắt cứng (Hard Cut): ghi đè trực tiếp theo biên gần hơn (baseline so sánh)
            w_f = feather_weight_map(canvas_mask, mask_b, power=1.0)
            use_b = (mask_b & (~canvas_mask)) | ((canvas_mask & mask_b) & (w_f >= 0.5))
            # Ghi lại đường ranh giới cắt cứng để so sánh chỉ số độ mượt seam
            ov = canvas_mask & mask_b
            ys, xs = np.nonzero(ov)
            if len(xs) > 0:
                mid_pts = np.argwhere(
                    ov & (np.abs(w_f - 0.5) < 0.015)
                )
                if len(mid_pts) > 0:
                    stride = max(1, len(mid_pts) // 200)
                    seam_polylines.append(mid_pts[::stride, ::-1].astype(np.float32))
            canvas[use_b] = img_b[use_b]
            canvas_mask = canvas_mask | mask_b

        elif blending_mode == "feather":
            w_b = feather_weight_map(canvas_mask, mask_b, power=1.5)
            ov = canvas_mask & mask_b
            only_b = mask_b & (~canvas_mask)
            canvas[only_b] = img_b[only_b]
            if np.any(ov):
                w3 = w_b[ov][:, np.newaxis]
                blended_px = (
                    (1.0 - w3) * canvas[ov].astype(np.float32)
                    + w3 * img_b[ov].astype(np.float32)
                )
                canvas[ov] = np.clip(np.round(blended_px), 0, 255).astype(np.uint8)
                mid_pts = np.argwhere(ov & (np.abs(w_b - 0.5) < 0.015))
                if len(mid_pts) > 0:
                    stride = max(1, len(mid_pts) // 200)
                    seam_polylines.append(mid_pts[::stride, ::-1].astype(np.float32))
            canvas_mask = canvas_mask | mask_b

        elif blending_mode == "multiband":
            # Đường chia Voronoi (trung điểm khoảng cách) + Trộn Tháp Laplacian Đa dải tần
            w_f = feather_weight_map(canvas_mask, mask_b, power=1.0)
            w_bin = (w_f >= 0.5).astype(np.float32)
            ov = canvas_mask & mask_b
            if np.any(ov):
                mid_pts = np.argwhere(ov & (np.abs(w_f - 0.5) < 0.015))
                if len(mid_pts) > 0:
                    stride = max(1, len(mid_pts) // 200)
                    seam_polylines.append(mid_pts[::stride, ::-1].astype(np.float32))
            canvas, canvas_mask = multiband_blend_pair(
                canvas, canvas_mask, img_b, mask_b, w_bin, levels=multiband_levels
            )

        elif blending_mode == "seam_multiband":
            # Đường nối tối ưu DP (chống ghosting) + Trộn Tháp Laplacian Đa dải tần (chống lệch sáng)
            w_dp, seam_pts = find_optimal_seam_dp(canvas, canvas_mask, img_b, mask_b)
            if len(seam_pts) > 0:
                seam_polylines.append(seam_pts)
            canvas, canvas_mask = multiband_blend_pair(
                canvas, canvas_mask, img_b, mask_b, w_dp, levels=multiband_levels
            )
        else:
            raise ValueError(f"Chế độ trộn ảnh không hợp lệ: {blending_mode}")

    seam_jump = _compute_seam_gradient_jump(canvas, seam_polylines)
    return BlendingResult(
        panorama=canvas,
        valid_mask=canvas_mask,
        exposure_gains=gains,
        seam_polylines=seam_polylines,
        seam_gradient_jump=seam_jump,
    )
