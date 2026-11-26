"""Module phát hiện và khớp đặc trưng (Feature Detection & Matching).

- Phát hiện đặc trưng: sử dụng SIFT / ORB của OpenCV kết hợp tối ưu tỷ lệ ảnh
  để đạt tốc độ cao trên ảnh độ phân giải lớn (Full HD / 4K) mà vẫn giữ độ chính xác subpixel.
- Khớp đặc trưng: TỰ CÀI ĐẶT hoàn toàn bằng NumPy thuần (vectorized Euclidean / Hamming
  distance matrix, top-2 nearest neighbors bằng argpartition, Lowe's Ratio Test và
  tùy chọn Cross-Check 2 chiều), song song cung cấp chế độ OpenCV BFMatcher để đối chiếu.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import DetectorType, ImplMode
from .exceptions import InsufficientFeaturesError


@dataclass
class ImageFeatures:
    """Lưu trữ tập điểm đặc trưng và vector mô tả của một ảnh."""

    image_index: int
    keypoints: List[cv2.KeyPoint]
    points: np.ndarray  # shape (N, 2), float32, tọa độ trên độ phân giải gốc
    descriptors: np.ndarray  # shape (N, D)
    detector: str
    scale_factor: float  # tỷ lệ thu nhỏ đã dùng khi detect (<= 1.0)


@dataclass
class FeatureMatch:
    """Một cặp điểm khớp giữa ảnh nguồn (query) và ảnh đích (train)."""

    query_idx: int
    train_idx: int
    distance: float
    ratio: float = 0.0


def detect_and_describe(
    image: np.ndarray,
    image_index: int = 0,
    detector: DetectorType = "sift",
    max_features: int = 2500,
    detection_max_dim: int = 1200,
    mask: Optional[np.ndarray] = None,
) -> ImageFeatures:
    """Phát hiện keypoints và trích xuất descriptors cho một ảnh BGR.

    Khi ảnh lớn hơn `detection_max_dim`, ảnh được thu nhỏ tạm thời để trích xuất
    đặc trưng nhanh hơn, sau đó tọa độ keypoint được ánh xạ ngược chính xác về
    hệ tọa độ ảnh gốc.
    """
    h, w = image.shape[:2]
    scale = 1.0
    if detection_max_dim > 0 and max(h, w) > detection_max_dim:
        scale = detection_max_dim / float(max(h, w))
        work_img = cv2.resize(
            image,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        work_mask = (
            cv2.resize(
                mask.astype(np.uint8),
                (work_img.shape[1], work_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            if mask is not None
            else None
        )
    else:
        work_img = image
        work_mask = mask.astype(np.uint8) if mask is not None else None

    gray = (
        cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
        if work_img.ndim == 3
        else work_img
    )

    if detector == "sift":
        det = cv2.SIFT_create(nfeatures=max_features)
    elif detector == "orb":
        det = cv2.ORB_create(nfeatures=max_features, scaleFactor=1.2, nlevels=8)
    else:
        raise ValueError(f"Detector không được hỗ trợ: {detector}")

    keypoints, descriptors = det.detectAndCompute(gray, work_mask)

    if not keypoints or descriptors is None or len(keypoints) < 4:
        raise InsufficientFeaturesError(
            message=f"Ảnh thứ {image_index + 1} có quá ít điểm đặc trưng ({0 if not keypoints else len(keypoints)} điểm).",
            suggestion="Hãy dùng ảnh rõ nét, tránh vùng trơn hoàn toàn (tường trắng, bầu trời trống) hoặc bị nhòe chuyển động.",
            details={"image_index": image_index, "num_keypoints": 0 if not keypoints else len(keypoints)},
        )

    inv_scale = 1.0 / scale
    scaled_kps: List[cv2.KeyPoint] = []
    pts = np.empty((len(keypoints), 2), dtype=np.float32)
    for i, kp in enumerate(keypoints):
        x = kp.pt[0] * inv_scale
        y = kp.pt[1] * inv_scale
        pts[i, 0] = x
        pts[i, 1] = y
        scaled_kps.append(
            cv2.KeyPoint(
                x=float(x),
                y=float(y),
                size=float(kp.size * inv_scale),
                angle=float(kp.angle),
                response=float(kp.response),
                octave=int(kp.octave),
                class_id=int(kp.class_id),
            )
        )

    return ImageFeatures(
        image_index=image_index,
        keypoints=scaled_kps,
        points=pts,
        descriptors=descriptors,
        detector=detector,
        scale_factor=scale,
    )


def _descriptors_to_matrix(descriptors: np.ndarray, is_binary: bool) -> np.ndarray:
    """Chuyển descriptor về ma trận float32 để tính khoảng cách vector hóa bằng đại số ma trận.

    - Với SIFT (float): dùng trực tiếp chuẩn L2.
    - Với ORB (uint8 nhị phân 32 byte = 256 bit): giải nén bit về {0.0, 1.0}.
      Khi a_i, b_i ∈ {0, 1}, khoảng cách Hamming thỏa mãn đồng nhất thức chính xác:
      Hamming(a, b) = ||a||_1 + ||b||_1 - 2 * (a · b).
    """
    if is_binary:
        return np.unpackbits(descriptors.astype(np.uint8), axis=1).astype(np.float32)
    return descriptors.astype(np.float32, copy=False)


def custom_knn_match_ratio_test(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio_threshold: float = 0.75,
    is_binary: bool = False,
    cross_check: bool = False,
    chunk_size: int = 512,
) -> List[FeatureMatch]:
    """TỰ CÀI ĐẶT: Khớp đặc trưng k-NN (k=2) + Lowe's Ratio Test hoàn toàn bằng NumPy thuần.

    Thuật toán:
    1. Chuyển tập vector mô tả D1 (N1×D) và D2 (N2×D) sang biểu diễn float32.
    2. Chia D1 theo từng khối (chunk) để tối ưu bộ nhớ RAM và tận dụng BLAS matmul:
       - L2 bình phương: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * (a @ b^T)
       - Hamming (nhị phân): H(a, b) = sum(a) + sum(b) - 2 * (a @ b^T)
    3. Tìm 2 láng giềng gần nhất (d1 <= d2) bằng `np.argpartition(..., kth=1)` trong O(N2).
    4. Áp dụng Lowe's Ratio Test: giữ lại cặp khớp nếu d1 < ratio_threshold * d2.
    5. Nếu `cross_check=True`: kiểm tra thêm tính đối xứng (láng giềng gần nhất của j
       trong D1 cũng phải là i) để loại bỏ nhiễu hai chiều.
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []

    mat1 = _descriptors_to_matrix(desc1, is_binary=is_binary)
    mat2 = _descriptors_to_matrix(desc2, is_binary=is_binary)

    n1 = mat1.shape[0]
    norm2_sq = np.sum(mat2 * mat2, axis=1)  # (N2,)

    best_train = np.empty(n1, dtype=np.int32)
    d1_arr = np.empty(n1, dtype=np.float32)
    d2_arr = np.empty(n1, dtype=np.float32)

    for start in range(0, n1, chunk_size):
        end = min(start + chunk_size, n1)
        sub1 = mat1[start:end]
        norm1_sq = np.sum(sub1 * sub1, axis=1, keepdims=True)  # (B, 1)

        # Khoảng cách L2 bình phương (hoặc Hamming chính xác với vector {0, 1})
        dist_sq = norm1_sq + norm2_sq[np.newaxis, :] - 2.0 * (sub1 @ mat2.T)
        np.maximum(dist_sq, 0.0, out=dist_sq)

        # Lấy 2 chỉ số nhỏ nhất không cần sắp xếp toàn bộ mảng
        top2_idx = np.argpartition(dist_sq, kth=1, axis=1)[:, :2]
        rows = np.arange(end - start)
        val0 = dist_sq[rows, top2_idx[:, 0]]
        val1 = dist_sq[rows, top2_idx[:, 1]]

        swap = val1 < val0
        idx_first = np.where(swap, top2_idx[:, 1], top2_idx[:, 0])
        min_sq = np.where(swap, val1, val0)
        second_sq = np.where(swap, val0, val1)

        best_train[start:end] = idx_first
        if is_binary:
            d1_arr[start:end] = min_sq
            d2_arr[start:end] = second_sq
        else:
            d1_arr[start:end] = np.sqrt(min_sq)
            d2_arr[start:end] = np.sqrt(second_sq)

    # Lowe's Ratio Test
    valid = d1_arr < (ratio_threshold * np.maximum(d2_arr, 1e-12))
    cand_q = np.flatnonzero(valid)

    if len(cand_q) == 0:
        return []

    # Tùy chọn Cross-Check: chỉ kiểm tra ngược lại trên các ứng viên đã qua ratio test
    if cross_check:
        cand_t = best_train[cand_q]
        unique_t, inv_map = np.unique(cand_t, return_inverse=True)
        norm1_all = np.sum(mat1 * mat1, axis=1)  # (N1,)
        sub2 = mat2[unique_t]
        rev_dist = (
            np.sum(sub2 * sub2, axis=1, keepdims=True)
            + norm1_all[np.newaxis, :]
            - 2.0 * (sub2 @ mat1.T)
        )
        rev_best_q = np.argmin(rev_dist, axis=1)
        mutual = rev_best_q[inv_map] == cand_q
        cand_q = cand_q[mutual]

    matches: List[FeatureMatch] = []
    for q_idx in cand_q:
        q = int(q_idx)
        t = int(best_train[q])
        d1 = float(d1_arr[q])
        d2 = float(d2_arr[q])
        matches.append(
            FeatureMatch(
                query_idx=q,
                train_idx=t,
                distance=d1,
                ratio=d1 / max(d2, 1e-12),
            )
        )

    matches.sort(key=lambda m: m.distance)
    return matches


def opencv_knn_match_ratio_test(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio_threshold: float = 0.75,
    is_binary: bool = False,
    cross_check: bool = False,
) -> List[FeatureMatch]:
    """Chế độ đối chiếu dùng OpenCV BFMatcher + Lowe's Ratio Test."""
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []

    norm = cv2.NORM_HAMMING if is_binary else cv2.NORM_L2
    bf = cv2.BFMatcher(norm, crossCheck=False)
    knn = bf.knnMatch(desc1, desc2, k=2)

    good: List[FeatureMatch] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair[0], pair[1]
        if m.distance < ratio_threshold * max(n.distance, 1e-12):
            good.append(
                FeatureMatch(
                    query_idx=int(m.queryIdx),
                    train_idx=int(m.trainIdx),
                    distance=float(m.distance),
                    ratio=float(m.distance / max(n.distance, 1e-12)),
                )
            )

    if cross_check and good:
        rev_knn = bf.knnMatch(desc2, desc1, k=1)
        rev_map = {
            int(p[0].queryIdx): int(p[0].trainIdx) for p in rev_knn if len(p) > 0
        }
        good = [m for m in good if rev_map.get(m.train_idx) == m.query_idx]

    good.sort(key=lambda m: m.distance)
    return good


def match_features(
    feat1: ImageFeatures,
    feat2: ImageFeatures,
    mode: ImplMode = "custom",
    ratio_threshold: float = 0.75,
    cross_check: bool = False,
) -> Tuple[List[FeatureMatch], np.ndarray, np.ndarray]:
    """Khớp đặc trưng giữa 2 ảnh và trả về (danh sách match, pts1, pts2)."""
    is_binary = feat1.detector == "orb"
    if mode == "custom":
        matches = custom_knn_match_ratio_test(
            feat1.descriptors,
            feat2.descriptors,
            ratio_threshold=ratio_threshold,
            is_binary=is_binary,
            cross_check=cross_check,
        )
    elif mode == "opencv":
        matches = opencv_knn_match_ratio_test(
            feat1.descriptors,
            feat2.descriptors,
            ratio_threshold=ratio_threshold,
            is_binary=is_binary,
            cross_check=cross_check,
        )
    else:
        raise ValueError(f"Chế độ matcher không hợp lệ: {mode}")

    if not matches:
        empty = np.empty((0, 2), dtype=np.float32)
        return matches, empty, empty

    q_idx = np.array([m.query_idx for m in matches], dtype=np.int32)
    t_idx = np.array([m.train_idx for m in matches], dtype=np.int32)
    pts1 = feat1.points[q_idx]
    pts2 = feat2.points[t_idx]
    return matches, pts1, pts2


def draw_keypoints_preview(
    image: np.ndarray,
    features: ImageFeatures,
    max_dim: int = 960,
    max_draw: int = 800,
) -> np.ndarray:
    """Vẽ các điểm đặc trưng nổi bật lên ảnh để minh họa bước Feature Detection."""
    h, w = image.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w))) if max_dim > 0 else 1.0
    canvas = (
        cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image.copy()
    )

    # Chọn các keypoints mạnh nhất theo response để hiển thị rõ nét, không rối mắt
    kps = sorted(features.keypoints, key=lambda k: k.response, reverse=True)[:max_draw]
    for kp in kps:
        cx = int(round(kp.pt[0] * scale))
        cy = int(round(kp.pt[1] * scale))
        radius = max(2, min(14, int(round(kp.size * scale * 0.25))))
        cv2.circle(canvas, (cx, cy), radius, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 1, (255, 80, 0), -1, cv2.LINE_AA)

    label = f"{features.detector.upper()} | {len(features.keypoints)} keypoints"
    cv2.rectangle(canvas, (10, 10), (10 + len(label) * 11 + 16, 44), (15, 23, 42), -1)
    cv2.putText(
        canvas,
        label,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (240, 249, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def draw_matches_preview(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None,
    max_width: int = 1400,
    max_lines: int = 120,
    title: str = "",
) -> np.ndarray:
    """Vẽ trực quan cặp điểm khớp: Inliers màu xanh lá, Outliers màu đỏ cam."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Chuẩn hóa cùng chiều cao để đặt 2 ảnh cạnh nhau cân đối
    target_h = min(h1, h2, 540)
    s1 = target_h / float(h1)
    s2 = target_h / float(h2)
    r1 = cv2.resize(img1, (int(round(w1 * s1)), target_h), interpolation=cv2.INTER_AREA)
    r2 = cv2.resize(img2, (int(round(w2 * s2)), target_h), interpolation=cv2.INTER_AREA)

    gap = 12
    total_w = r1.shape[1] + gap + r2.shape[1]
    scale_all = min(1.0, max_width / float(total_w)) if max_width > 0 else 1.0
    if scale_all < 1.0:
        r1 = cv2.resize(
            r1,
            (int(round(r1.shape[1] * scale_all)), int(round(target_h * scale_all))),
            interpolation=cv2.INTER_AREA,
        )
        r2 = cv2.resize(
            r2,
            (int(round(r2.shape[1] * scale_all)), int(round(target_h * scale_all))),
            interpolation=cv2.INTER_AREA,
        )
        s1 *= scale_all
        s2 *= scale_all

    out_h = max(r1.shape[0], r2.shape[0])
    out_w = r1.shape[1] + gap + r2.shape[1]
    canvas = np.full((out_h, out_w, 3), 18, dtype=np.uint8)
    canvas[: r1.shape[0], : r1.shape[1]] = r1
    x_off = r1.shape[1] + gap
    canvas[: r2.shape[0], x_off : x_off + r2.shape[1]] = r2

    n = len(pts1)
    if n > 0:
        if inlier_mask is None:
            inlier_mask = np.ones(n, dtype=bool)
        else:
            inlier_mask = inlier_mask.astype(bool).ravel()

        inlier_idx = np.flatnonzero(inlier_mask)
        outlier_idx = np.flatnonzero(~inlier_mask)

        # Vẽ tối đa 30% là outliers (nếu có) và phần lớn là inliers để dễ phân tích
        n_out = min(len(outlier_idx), max(10, max_lines // 4))
        n_in = min(len(inlier_idx), max_lines - n_out)

        if n_out > 0:
            sel_out = outlier_idx[np.linspace(0, len(outlier_idx) - 1, n_out, dtype=int)]
            for idx in sel_out:
                p1 = (int(round(pts1[idx, 0] * s1)), int(round(pts1[idx, 1] * s1)))
                p2 = (int(round(pts2[idx, 0] * s2)) + x_off, int(round(pts2[idx, 1] * s2)))
                cv2.line(canvas, p1, p2, (70, 70, 230), 1, cv2.LINE_AA)
                cv2.circle(canvas, p1, 3, (70, 70, 230), -1, cv2.LINE_AA)
                cv2.circle(canvas, p2, 3, (70, 70, 230), -1, cv2.LINE_AA)

        if n_in > 0:
            sel_in = inlier_idx[np.linspace(0, len(inlier_idx) - 1, n_in, dtype=int)]
            for idx in sel_in:
                p1 = (int(round(pts1[idx, 0] * s1)), int(round(pts1[idx, 1] * s1)))
                p2 = (int(round(pts2[idx, 0] * s2)) + x_off, int(round(pts2[idx, 1] * s2)))
                cv2.line(canvas, p1, p2, (60, 235, 120), 1, cv2.LINE_AA)
                cv2.circle(canvas, p1, 3, (60, 235, 120), -1, cv2.LINE_AA)
                cv2.circle(canvas, p2, 3, (60, 235, 120), -1, cv2.LINE_AA)

        n_in_total = int(np.sum(inlier_mask))
        ratio_pct = 100.0 * n_in_total / max(1, n)
        banner = f"{title} | Inliers: {n_in_total}/{n} ({ratio_pct:.1f}%)" if title else f"Inliers: {n_in_total}/{n} ({ratio_pct:.1f}%)"
    else:
        banner = f"{title} | 0 matches" if title else "0 matches"

    cv2.rectangle(canvas, (10, 10), (min(out_w - 10, 10 + len(banner) * 10 + 24), 44), (15, 23, 42), -1)
    cv2.putText(
        canvas,
        banner,
        (18, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (240, 249, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas
