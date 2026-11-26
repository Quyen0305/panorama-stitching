"""Module biến đổi hình học và căn chỉnh ảnh (Warping & Cylindrical Projection).

Bao gồm:
- Chiếu trụ (Cylindrical Projection) TỰ CÀI ĐẶT bằng ánh xạ nghịch NumPy giúp giảm
  méo phối cảnh ở rìa khi ghép nhiều ảnh góc rộng (wide-FOV horizontal sweeps).
- Tính khung ảnh toàn cục (global bounding box) từ chuỗi ma trận Homography với
  cơ chế bảo vệ bộ nhớ chống bùng nổ kích thước canvas khi H suy biến.
- Biến đổi phối cảnh (`cv2.warpPerspective`) kèm co nhẹ mask 1px để triệt tiêu
  viền đen nội suy tuyến tính ở rìa ảnh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .exceptions import CanvasOverflowError


@dataclass
class WarpedFrame:
    """Một ảnh nguồn đã được chiếu lên hệ tọa độ chung của panorama."""

    image_index: int
    image: np.ndarray  # BGR uint8 (canvas_h, canvas_w, 3)
    mask: np.ndarray  # bool (canvas_h, canvas_w)
    bbox: Tuple[int, int, int, int]  # (x0, y0, x1, y1) vùng chứa pixel hợp lệ
    H_canvas: np.ndarray  # Ma trận 3×3 từ ảnh nguồn -> canvas


def cylindrical_project_image(
    image: np.ndarray, fov_deg: float = 55.0
) -> Tuple[np.ndarray, np.ndarray]:
    """TỰ CÀI ĐẶT: Chiếu ảnh phẳng lên mặt trụ có bán kính f = W / (2 * tan(FOV/2)).

    Ánh xạ nghịch từ tọa độ trên mặt trụ (x_c, y_c) về ảnh phẳng gốc (x_s, y_s):
        theta = (x_c - cx) / f
        h     = (y_c - cy) / f
        x_s   = f * tan(theta) + cx
        y_s   = (y_c - cy) / cos(theta) + cy
    """
    h, w = image.shape[:2]
    fov_rad = np.deg2rad(np.clip(fov_deg, 15.0, 140.0))
    f = (w * 0.5) / np.tan(fov_rad * 0.5)

    cx = w * 0.5
    cy = h * 0.5

    # Chiều rộng thực trên mặt trụ: 2 * f * atan(w / (2f))
    max_theta = np.arctan(cx / f)
    out_w = int(np.ceil(2.0 * f * max_theta))
    out_h = h
    out_cx = out_w * 0.5
    out_cy = out_h * 0.5

    xs = np.arange(out_w, dtype=np.float32)
    ys = np.arange(out_h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    theta = (grid_x - out_cx) / np.float32(f)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    map_x = (np.float32(f) * (sin_t / np.maximum(cos_t, 1e-6)) + np.float32(cx)).astype(
        np.float32
    )
    map_y = ((grid_y - np.float32(out_cy)) / np.maximum(cos_t, 1e-6) + np.float32(cy)).astype(
        np.float32
    )

    valid = (map_x >= 1.0) & (map_x <= w - 2.0) & (map_y >= 1.0) & (map_y <= h - 2.0)
    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    mask = valid.astype(bool)
    warped[~mask] = 0
    return warped, mask


def compute_canvas_geometry(
    image_shapes: List[Tuple[int, int]],
    global_homographies: List[np.ndarray],
    max_canvas_megapixels: float = 36.0,
) -> Tuple[Tuple[int, int], np.ndarray, float]:
    """Tính kích thước canvas (width, height), ma trận tịnh tiến-tỷ lệ T_canvas và scale.

    Trả về:
        (canvas_w, canvas_h), T_canvas (3×3), output_scale (<= 1.0)
    """
    all_corners = []
    total_input_area = 0.0

    for (h, w), H in zip(image_shapes, global_homographies):
        total_input_area += float(h * w)
        corners = np.array(
            [[0.0, 0.0, 1.0], [w, 0.0, 1.0], [w, h, 1.0], [0.0, h, 1.0]],
            dtype=np.float64,
        )
        proj = (H @ corners.T).T
        if np.any(proj[:, 2] <= 1e-4):
            raise CanvasOverflowError(
                message="Phép biến đổi phối cảnh tạo góc chiếu cực đoan (vượt quá mặt phẳng vô cực).",
                suggestion="Với chuỗi ảnh góc quá rộng (> 100°), hãy chuyển Chế độ chiếu từ 'Planar' sang 'Cylindrical' hoặc giảm số lượng ảnh ghép cùng lúc.",
            )
        pts = proj[:, :2] / proj[:, 2:3]
        all_corners.append(pts)

    stacked = np.vstack(all_corners)
    min_x, min_y = np.floor(np.min(stacked, axis=0))
    max_x, max_y = np.ceil(np.max(stacked, axis=0))

    raw_w = float(max_x - min_x)
    raw_h = float(max_y - min_y)
    raw_area = raw_w * raw_h

    # Kiểm tra trường hợp Homography lỗi gây bùng nổ kích thước bất thường (> 25x tổng diện tích ảnh vào)
    if raw_w <= 0 or raw_h <= 0 or raw_area > max(total_input_area * 25.0, 400e6):
        raise CanvasOverflowError(
            message=f"Kích thước vùng ảnh sau biến đổi quá lớn ({int(raw_w)}×{int(raw_h)} px) do biến dạng phối cảnh mạnh.",
            suggestion="Hãy thử bật chế độ chiếu 'Cylindrical', kiểm tra lại ảnh đầu vào hoặc giảm góc xoay giữa các ảnh.",
            details={"raw_width": int(raw_w), "raw_height": int(raw_h)},
        )

    max_pixels = max_canvas_megapixels * 1e6
    scale = 1.0
    if raw_area > max_pixels:
        scale = float(np.sqrt(max_pixels / raw_area))

    canvas_w = max(16, int(round(raw_w * scale)))
    canvas_h = max(16, int(round(raw_h * scale)))

    T_shift = np.array(
        [[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    T_scale = np.array(
        [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    T_canvas = T_scale @ T_shift
    return (canvas_w, canvas_h), T_canvas, scale


def warp_all_images(
    images: List[np.ndarray],
    global_homographies: List[np.ndarray],
    input_masks: Optional[List[Optional[np.ndarray]]] = None,
    max_canvas_megapixels: float = 36.0,
) -> Tuple[List[WarpedFrame], Tuple[int, int], float]:
    """Warp toàn bộ tập ảnh và mặt nạ hợp lệ về chung một hệ tọa độ canvas."""
    shapes = [(img.shape[0], img.shape[1]) for img in images]
    (canvas_w, canvas_h), T_canvas, out_scale = compute_canvas_geometry(
        shapes, global_homographies, max_canvas_megapixels=max_canvas_megapixels
    )

    kernel_erode = np.ones((3, 3), dtype=np.uint8)
    frames: List[WarpedFrame] = []

    for idx, (img, H) in enumerate(zip(images, global_homographies)):
        H_canvas = T_canvas @ H
        warped_img = cv2.warpPerspective(
            img,
            H_canvas,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        if input_masks is not None and input_masks[idx] is not None:
            src_mask = (input_masks[idx].astype(np.uint8)) * 255
        else:
            src_mask = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)

        warped_mask_u8 = cv2.warpPerspective(
            src_mask,
            H_canvas,
            (canvas_w, canvas_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        # Co nhẹ 1px để loại bỏ dải viền nội suy lẫn màu đen ở biên ngoài của mỗi ảnh
        eroded = cv2.erode(warped_mask_u8, kernel_erode, iterations=1)
        mask_bool = eroded > 127
        if not np.any(mask_bool):
            mask_bool = warped_mask_u8 > 127

        warped_img[~mask_bool] = 0

        ys, xs = np.nonzero(mask_bool)
        if len(xs) > 0:
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        else:
            bbox = (0, 0, 0, 0)

        frames.append(
            WarpedFrame(
                image_index=idx,
                image=warped_img,
                mask=mask_bool,
                bbox=bbox,
                H_canvas=H_canvas,
            )
        )

    return frames, (canvas_w, canvas_h), out_scale


def draw_warped_layout_preview(
    frames: List[WarpedFrame],
    canvas_size: Tuple[int, int],
    seam_points: Optional[List[np.ndarray]] = None,
    max_width: int = 1280,
) -> np.ndarray:
    """Tạo ảnh minh họa các khung ảnh sau khi warp (viền màu khác nhau + đường nối tối ưu)."""
    cw, ch = canvas_size
    scale = min(1.0, max_width / float(max(cw, ch))) if max_width > 0 else 1.0
    out_w = max(1, int(round(cw * scale)))
    out_h = max(1, int(round(ch * scale)))

    palette = [
        (80, 180, 255),
        (100, 240, 140),
        (255, 190, 60),
        (240, 110, 220),
        (90, 230, 240),
        (130, 140, 255),
        (180, 240, 90),
        (255, 130, 130),
        (200, 180, 255),
        (255, 220, 130),
    ]

    acc = np.zeros((out_h, out_w, 3), dtype=np.float32)
    counts = np.zeros((out_h, out_w, 1), dtype=np.float32)

    contours_to_draw = []
    for i, fr in enumerate(frames):
        small_img = cv2.resize(fr.image, (out_w, out_h), interpolation=cv2.INTER_AREA)
        small_m = cv2.resize(
            fr.mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST
        )
        m_bool = small_m > 0
        acc[m_bool] += small_img[m_bool].astype(np.float32)
        counts[m_bool] += 1.0

        cnts, _ = cv2.findContours(small_m * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_to_draw.append((i, fr.image_index, cnts, m_bool))

    vis = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    valid = counts[:, :, 0] > 0
    vis[valid] = np.clip(acc[valid] / counts[valid], 0, 255).astype(np.uint8)

    # Tô sáng nhẹ vùng chồng lấn để dễ quan sát
    overlap = counts[:, :, 0] > 1.0
    vis[overlap] = np.clip(
        vis[overlap].astype(np.float32) * 0.85 + np.array([20, 45, 55], dtype=np.float32),
        0,
        255,
    ).astype(np.uint8)

    for i, orig_idx, cnts, m_bool in contours_to_draw:
        color = palette[i % len(palette)]
        cv2.drawContours(vis, cnts, -1, color, 2, cv2.LINE_AA)
        ys, xs = np.nonzero(m_bool)
        if len(xs) > 0:
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            tag = f"#{orig_idx + 1}"
            cv2.rectangle(vis, (cx - 18, cy - 14), (cx + 22, cy + 8), (15, 23, 42), -1)
            cv2.putText(
                vis, tag, (cx - 14, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )

    # Vẽ các đường seam tối ưu (nếu có) bằng màu đỏ cam nổi bật
    if seam_points:
        for pts in seam_points:
            if len(pts) >= 2:
                scaled_pts = np.round(pts * scale).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [scaled_pts], isClosed=False, color=(40, 80, 255), thickness=2, lineType=cv2.LINE_AA)

    return vis
