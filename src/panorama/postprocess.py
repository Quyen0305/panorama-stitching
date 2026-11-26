"""Module hậu xử lý (Post-processing) & cắt biên tự động.

TỰ CÀI ĐẶT:
- Thuật toán tìm Hình chữ nhật nội tiếp lớn nhất (Largest Inscribed Axis-Aligned
  Rectangle) bên trong mặt nạ hợp lệ bằng ngăn xếp đơn điệu O(H * W), loại bỏ hoàn
  toàn các vùng tam giác đen ở biên sau phép chiếu phối cảnh/trụ trong khi giữ tối đa
  diện tích ảnh thật.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _largest_rectangle_in_binary_matrix(binary: np.ndarray) -> Tuple[int, int, int, int]:
    """Thuật toán O(H * W) tìm hình chữ nhật toàn True có diện tích lớn nhất.

    Trả về (x0, y0, x1, y1) với x1, y1 là cận trên không bao gồm (slice Python).
    """
    h, w = binary.shape
    heights = np.zeros(w, dtype=np.int32)
    best_area = 0
    best_rect = (0, 0, w, h)

    for y in range(h):
        row = binary[y]
        heights = np.where(row, heights + 1, 0)

        # Monotonic increasing stack lưu cặp (start_col, height)
        stack = []
        for x in range(w + 1):
            curr_h = int(heights[x]) if x < w else 0
            start = x
            while stack and stack[-1][1] > curr_h:
                s_col, s_h = stack.pop()
                area = s_h * (x - s_col)
                if area > best_area:
                    best_area = area
                    best_rect = (s_col, y - s_h + 1, x, y + 1)
                start = s_col
            if curr_h > 0:
                stack.append((start, curr_h))

    return best_rect


def auto_crop_largest_rectangle(
    image: np.ndarray, valid_mask: np.ndarray, work_max_dim: int = 360
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """TỰ CÀI ĐẶT: Cắt biên đen tự động bằng Hình chữ nhật nội tiếp lớn nhất.

    1. Giảm phân giải mặt nạ theo phép co bảo thủ (min-pooling: 1 ô nhỏ chỉ True khi
       TẤT CẢ các pixel độ phân giải gốc trong ô đó đều hợp lệ) -> đảm bảo kết quả
       không bao giờ lọt pixel đen vào khung cắt.
    2. Chạy thuật toán ngăn xếp đơn điệu O(h * w) tìm hình chữ nhật diện tích cực đại.
    3. Quy đổi về tọa độ gốc và co tinh chỉnh sát biên nếu còn pixel biên đen.
    """
    h, w = valid_mask.shape
    if not np.any(valid_mask):
        return image.copy(), (0, 0, w, h)

    # Tìm bounding box chứa toàn bộ vùng hợp lệ trước
    ys, xs = np.nonzero(valid_mask)
    by0, by1 = int(ys.min()), int(ys.max()) + 1
    bx0, bx1 = int(xs.min()), int(xs.max()) + 1
    sub_mask = valid_mask[by0:by1, bx0:bx1]
    sh, sw = sub_mask.shape

    step = max(1, int(np.ceil(max(sh, sw) / float(work_max_dim))))
    if step > 1:
        gh = sh // step
        gw = sw // step
        if gh >= 4 and gw >= 4:
            # Min-pooling bảo thủ bằng reshape + all()
            trimmed = sub_mask[: gh * step, : gw * step]
            small = trimmed.reshape(gh, step, gw, step).all(axis=(1, 3))
            rx0, ry0, rx1, ry1 = _largest_rectangle_in_binary_matrix(small)
            x0 = bx0 + rx0 * step
            y0 = by0 + ry0 * step
            x1 = bx0 + rx1 * step
            y1 = by0 + ry1 * step
        else:
            x0, y0, x1, y1 = bx0, by0, bx1, by1
    else:
        rx0, ry0, rx1, ry1 = _largest_rectangle_in_binary_matrix(sub_mask)
        x0, y0, x1, y1 = bx0 + rx0, by0 + ry0, bx0 + rx1, by0 + ry1

    # Mở rộng từng cạnh 1px nếu toàn bộ cạnh kề vẫn 100% hợp lệ (tối ưu sát biên thật)
    while y0 > by0 and np.all(valid_mask[y0 - 1, x0:x1]):
        y0 -= 1
    while y1 < by1 and np.all(valid_mask[y1, x0:x1]):
        y1 += 1
    while x0 > bx0 and np.all(valid_mask[y0:y1, x0 - 1]):
        x0 -= 1
    while x1 < bx1 and np.all(valid_mask[y0:y1, x1]):
        x1 += 1

    # Bảo vệ biên: co nhẹ nếu vẫn còn bất kỳ pixel biên nào ngoài mask
    while (y1 - y0 > 16) and (x1 - x0 > 16) and not np.all(valid_mask[y0:y1, x0:x1]):
        top_bad = int(np.count_nonzero(~valid_mask[y0, x0:x1]))
        bot_bad = int(np.count_nonzero(~valid_mask[y1 - 1, x0:x1]))
        left_bad = int(np.count_nonzero(~valid_mask[y0:y1, x0]))
        right_bad = int(np.count_nonzero(~valid_mask[y0:y1, x1 - 1]))
        worst = max(top_bad, bot_bad, left_bad, right_bad)
        if worst == 0:
            break
        if worst == top_bad:
            y0 += 1
        elif worst == bot_bad:
            y1 -= 1
        elif worst == left_bad:
            x0 += 1
        else:
            x1 -= 1

    # Nếu diện tích crop quá nhỏ bất thường (< 20% diện tích bbox), fallback về bbox sạch
    if (x1 - x0) * (y1 - y0) < 0.2 * max(1, (bx1 - bx0) * (by1 - by0)):
        return image[by0:by1, bx0:bx1].copy(), (bx0, by0, bx1, by1)

    return image[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)
