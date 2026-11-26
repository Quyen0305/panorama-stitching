"""Script tạo 5 bộ dữ liệu kiểm thử không bản quyền (từ 4 cảnh gốc tự sinh)
bằng mô phỏng phép quay 3D của camera pinhole (H = K_master @ R @ K_cam^-1).

Các bộ dữ liệu được tạo:
1. 01_outdoor_5imgs_fhd      : 5 ảnh Full HD (1920×1080) ngoại cảnh hồ núi Alpine,
                               thứ tự bị xáo trộn để kiểm chứng Auto-Order & hiệu năng < 30s.
2. 02_urban_4imgs            : 4 ảnh (960×1200) kiến trúc quảng trường châu Âu (nhiều đường thẳng).
3. 03_indoor_3imgs           : 3 ảnh (1000×1200) nội thất thư viện đại học (ánh sáng trong nhà).
4. 04_exposure_motion_4imgs  : 4 ảnh (960×1200) bến cảng có chênh lệch phơi sáng mạnh (±30%)
                               và vật thể chuyển động trong vùng chồng lấn để kiểm chứng
                               Gain Compensation + DP Seam chống ghosting.
5. 05_negative_non_overlapping: 3 ảnh không liên quan từ 3 cảnh khác nhau để kiểm thử
                                xử lý lỗi an toàn (không crash).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from panorama.pipeline import load_image_unicode, save_image_unicode  # noqa: E402


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Ma trận xoay 3D R = R_y(yaw) @ R_x(pitch) @ R_z(roll)."""
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)

    Ry = np.array(
        [
            [math.cos(y), 0.0, math.sin(y)],
            [0.0, 1.0, 0.0],
            [-math.sin(y), 0.0, math.cos(y)],
        ],
        dtype=np.float64,
    )
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(p), -math.sin(p)],
            [0.0, math.sin(p), math.cos(p)],
        ],
        dtype=np.float64,
    )
    Rz = np.array(
        [
            [math.cos(r), -math.sin(r), 0.0],
            [math.sin(r), math.cos(r), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return Ry @ Rx @ Rz


def camera_matrix(w: int, h: int, focal: float) -> np.ndarray:
    return np.array(
        [[focal, 0.0, w * 0.5], [0.0, focal, h * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def upscale_crisp(img: np.ndarray, target_w: int) -> np.ndarray:
    """Phóng to sắc nét (Lanczos4 + unsharp mask nhẹ) để trích xuất khung Full HD rõ nét."""
    h, w = img.shape[:2]
    scale = target_w / float(w)
    target_h = int(round(h * scale))
    up = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0, 0), sigmaX=1.2)
    sharp = cv2.addWeighted(up, 1.22, blur, -0.22, 0)
    return sharp


def render_camera_view(
    master: np.ndarray,
    master_focal: float,
    cam_size: Tuple[int, int],
    cam_focal: float,
    yaw_deg: float,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    exposure_rgb: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    noise_sigma: float = 0.8,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Chiếu tạo 1 khung ảnh từ cảnh gốc bằng ma trận Homography H_cam_to_master."""
    mh, mw = master.shape[:2]
    cw, ch = cam_size

    Km = camera_matrix(mw, mh, master_focal)
    Kc = camera_matrix(cw, ch, cam_focal)
    R = rotation_matrix(yaw_deg, pitch_deg, roll_deg)
    H_cam_to_master = Km @ R @ np.linalg.inv(Kc)
    H_cam_to_master /= H_cam_to_master[2, 2]

    # Kiểm tra 4 góc camera đều nằm gọn trong ảnh gốc
    corners = np.array(
        [[0, 0, 1], [cw - 1, 0, 1], [cw - 1, ch - 1, 1], [0, ch - 1, 1]],
        dtype=np.float64,
    )
    proj = (H_cam_to_master @ corners.T).T
    pts = proj[:, :2] / proj[:, 2:3]
    inside = bool(
        np.all(pts[:, 0] >= 1.0)
        and np.all(pts[:, 0] <= mw - 2.0)
        and np.all(pts[:, 1] >= 1.0)
        and np.all(pts[:, 1] <= mh - 2.0)
    )

    frame = cv2.warpPerspective(
        master,
        H_cam_to_master,
        (cw, ch),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Mô phỏng khác biệt phơi sáng & cân bằng trắng giữa các lần chụp
    b_g, g_g, r_g = exposure_rgb[2], exposure_rgb[1], exposure_rgb[0]
    arr = frame.astype(np.float32)
    arr[:, :, 0] *= b_g
    arr[:, :, 1] *= g_g
    arr[:, :, 2] *= r_g

    if noise_sigma > 0 and rng is not None:
        arr += rng.normal(0.0, noise_sigma, size=arr.shape).astype(np.float32)

    out = np.clip(arr, 0, 255).astype(np.uint8)
    return out, H_cam_to_master, inside


def inject_moving_boat(frame: np.ndarray) -> np.ndarray:
    """Chèn 1 vật thể chuyển động nổi bật (phao & ca-nô đỏ trắng) vào vùng chồng lấn
    của 1 khung ảnh duy nhất để kiểm thử khả năng né ghosting của DP Seam Finding.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    # Đặt ở 1/3 phải (vùng chồng lấn giữa ảnh 2 và ảnh 3), trên mặt nước cảng
    cx, cy = int(w * 0.72), int(h * 0.68)

    # Bóng phản chiếu dưới mặt nước
    cv2.ellipse(out, (cx, cy + 28), (62, 14), 0, 0, 360, (70, 50, 40), -1, cv2.LINE_AA)
    # Sóng rẽ nước màu trắng
    cv2.ellipse(out, (cx - 10, cy + 18), (74, 10), 0, 0, 360, (235, 245, 250), 2, cv2.LINE_AA)
    # Thân ca-nô màu đỏ tươi
    hull = np.array(
        [
            [cx - 64, cy],
            [cx + 68, cy - 4],
            [cx + 50, cy + 20],
            [cx - 52, cy + 20],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(out, [hull], (35, 45, 225), cv2.LINE_AA)
    cv2.polylines(out, [hull], True, (25, 25, 140), 2, cv2.LINE_AA)
    # Dải trắng thân tàu
    cv2.line(out, (cx - 56, cy + 6), (cx + 56, cy + 4), (245, 248, 252), 4, cv2.LINE_AA)
    # Khoang lái màu trắng + kính xanh đậm
    cabin = np.array(
        [[cx - 24, cy - 2], [cx + 26, cy - 2], [cx + 14, cy - 26], [cx - 18, cy - 24]],
        dtype=np.int32,
    )
    cv2.fillPoly(out, [cabin], (242, 245, 248), cv2.LINE_AA)
    window = np.array(
        [[cx - 14, cy - 6], [cx + 18, cy - 6], [cx + 11, cy - 20], [cx - 11, cy - 19]],
        dtype=np.int32,
    )
    cv2.fillPoly(out, [window], (65, 50, 35), cv2.LINE_AA)
    # Cột cờ nhỏ màu vàng cam
    cv2.line(out, (cx - 46, cy), (cx - 46, cy - 36), (220, 225, 230), 2, cv2.LINE_AA)
    flag = np.array([[cx - 46, cy - 36], [cx - 24, cy - 30], [cx - 46, cy - 24]], dtype=np.int32)
    cv2.fillPoly(out, [flag], (20, 195, 255), cv2.LINE_AA)
    return out


def build_all_datasets() -> None:
    masters_dir = PROJECT_ROOT / "datasets" / "_masters"
    datasets_root = PROJECT_ROOT / "datasets"
    rng = np.random.default_rng(2026)

    # ------------------------------------------------------------------
    # Dataset 1: Ngoại cảnh Hồ Alpine - 5 ảnh Full HD (1920x1080), XÁO TRỘN
    # ------------------------------------------------------------------
    m1 = upscale_crisp(load_image_unicode(masters_dir / "master_01_outdoor_lake.jpg"), target_w=3840)
    d1 = datasets_root / "01_outdoor_5imgs_fhd"
    d1.mkdir(parents=True, exist_ok=True)

    mh, mw = m1.shape[:2]
    f_m1 = mw * 0.56  # Master H-FOV ~ 83.5 deg
    cam_w1, cam_h1 = 1920, 1080
    f_c1 = cam_w1 * 1.48  # Camera H-FOV ~ 37.3 deg, V-FOV ~ 21.4 deg

    # 5 góc chụp từ trái sang phải, sau đó lưu theo thứ tự xáo trộn để test Auto-Order
    specs_1 = [
        {"true_order": 0, "file": "img_03.jpg", "yaw": -21.0, "pitch": 0.4, "roll": -1.1, "exp": (0.96, 0.97, 0.98)},
        {"true_order": 1, "file": "img_01.jpg", "yaw": -10.5, "pitch": -0.3, "roll": 0.8, "exp": (1.02, 1.01, 1.00)},
        {"true_order": 2, "file": "img_05.jpg", "yaw": 0.0, "pitch": 0.2, "roll": -0.4, "exp": (1.00, 1.00, 1.00)},
        {"true_order": 3, "file": "img_02.jpg", "yaw": 10.5, "pitch": -0.4, "roll": 1.0, "exp": (0.94, 0.96, 0.98)},
        {"true_order": 4, "file": "img_04.jpg", "yaw": 21.0, "pitch": 0.3, "roll": -0.9, "exp": (1.05, 1.04, 1.02)},
    ]
    meta_1 = []
    for sp in specs_1:
        fr, H_gt, inside = render_camera_view(
            m1, f_m1, (cam_w1, cam_h1), f_c1, sp["yaw"], sp["pitch"], sp["roll"], sp["exp"], 0.6, rng
        )
        assert inside, f"Dataset 1 frame {sp['file']} exceeded master bounds!"
        save_image_unicode(d1 / sp["file"], fr, jpg_quality=94)
        meta_1.append({**sp, "H_cam_to_master": H_gt.tolist()})

    (d1 / "metadata.json").write_text(
        json.dumps(
            {
                "name": "01 - Ngoại cảnh Hồ núi Alpine (5 ảnh Full HD 1920×1080, xáo trộn)",
                "description": "Kiểm chứng ghép 5 ảnh Full HD dưới 30s và khả năng tự động sắp xếp thứ tự ảnh (Auto-Order).",
                "master_size": [mw, mh],
                "frames": meta_1,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Dataset 2: Kiến trúc Đô thị Quảng trường Châu Âu - 4 ảnh (960x1200)
    # ------------------------------------------------------------------
    m2 = upscale_crisp(load_image_unicode(masters_dir / "master_02_urban_square.jpg"), target_w=2800)
    d2 = datasets_root / "02_urban_4imgs"
    d2.mkdir(parents=True, exist_ok=True)

    mh2, mw2 = m2.shape[:2]
    f_m2 = mw2 * 0.50
    cam_w2, cam_h2 = 960, 1080
    f_c2 = cam_w2 * 1.48

    specs_2 = [
        {"true_order": 0, "file": "urban_1.jpg", "yaw": -21.0, "pitch": 0.4, "roll": -1.3, "exp": (0.95, 0.96, 0.97)},
        {"true_order": 1, "file": "urban_2.jpg", "yaw": -7.0, "pitch": -0.3, "roll": 0.9, "exp": (1.03, 1.02, 1.00)},
        {"true_order": 2, "file": "urban_3.jpg", "yaw": 7.0, "pitch": 0.3, "roll": -0.7, "exp": (0.97, 0.98, 1.00)},
        {"true_order": 3, "file": "urban_4.jpg", "yaw": 21.0, "pitch": -0.4, "roll": 1.1, "exp": (1.05, 1.03, 1.01)},
    ]
    meta_2 = []
    for sp in specs_2:
        fr, H_gt, inside = render_camera_view(
            m2, f_m2, (cam_w2, cam_h2), f_c2, sp["yaw"], sp["pitch"], sp["roll"], sp["exp"], 0.6, rng
        )
        assert inside, f"Dataset 2 frame {sp['file']} exceeded master bounds!"
        save_image_unicode(d2 / sp["file"], fr, jpg_quality=94)
        meta_2.append({**sp, "H_cam_to_master": H_gt.tolist()})

    (d2 / "metadata.json").write_text(
        json.dumps(
            {
                "name": "02 - Kiến trúc Quảng trường Châu Âu (4 ảnh, chồng lấn ~40%)",
                "description": "Cảnh kiến trúc nhiều đường thẳng và kết cấu sắc nét để kiểm định sai số Homography.",
                "master_size": [mw2, mh2],
                "frames": meta_2,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Dataset 3: Trong nhà - Thư viện Đại học - 3 ảnh (1080x1200)
    # ------------------------------------------------------------------
    m3 = upscale_crisp(load_image_unicode(masters_dir / "master_03_indoor_library.jpg"), target_w=2800)
    d3 = datasets_root / "03_indoor_3imgs"
    d3.mkdir(parents=True, exist_ok=True)

    mh3, mw3 = m3.shape[:2]
    f_m3 = mw3 * 0.50
    cam_w3, cam_h3 = 1080, 1080
    f_c3 = cam_w3 * 1.38

    specs_3 = [
        {"true_order": 0, "file": "library_left.jpg", "yaw": -18.5, "pitch": 0.3, "roll": -1.0, "exp": (0.91, 0.92, 0.94)},
        {"true_order": 1, "file": "library_center.jpg", "yaw": 0.0, "pitch": -0.2, "roll": 0.5, "exp": (1.02, 1.00, 0.98)},
        {"true_order": 2, "file": "library_right.jpg", "yaw": 18.5, "pitch": 0.3, "roll": -1.0, "exp": (1.10, 1.08, 1.05)},
    ]
    meta_3 = []
    for sp in specs_3:
        fr, H_gt, inside = render_camera_view(
            m3, f_m3, (cam_w3, cam_h3), f_c3, sp["yaw"], sp["pitch"], sp["roll"], sp["exp"], 0.7, rng
        )
        assert inside, f"Dataset 3 frame {sp['file']} exceeded master bounds!"
        save_image_unicode(d3 / sp["file"], fr, jpg_quality=94)
        meta_3.append({**sp, "H_cam_to_master": H_gt.tolist()})

    (d3 / "metadata.json").write_text(
        json.dumps(
            {
                "name": "03 - Trong nhà: Thư viện Đại học (3 ảnh nội thất)",
                "description": "Bối cảnh trong nhà có ánh sáng ấm, kệ sách và chiều sâu kiến trúc phức tạp.",
                "master_size": [mw3, mh3],
                "frames": meta_3,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Dataset 4: Bến cảng Địa Trung Hải - Chênh lệch phơi sáng mạnh + Vật chuyển động
    # ------------------------------------------------------------------
    m4 = upscale_crisp(load_image_unicode(masters_dir / "master_04_coastal_harbor.jpg"), target_w=2800)
    d4 = datasets_root / "04_exposure_motion_4imgs"
    d4.mkdir(parents=True, exist_ok=True)

    mh4, mw4 = m4.shape[:2]
    f_m4 = mw4 * 0.50
    cam_w4, cam_h4 = 960, 1080
    f_c4 = cam_w4 * 1.48

    specs_4 = [
        {"true_order": 0, "file": "harbor_1.jpg", "yaw": -21.0, "pitch": 0.3, "roll": -0.9, "exp": (0.74, 0.76, 0.80)},
        {"true_order": 1, "file": "harbor_2.jpg", "yaw": -7.0, "pitch": -0.3, "roll": 0.7, "exp": (1.18, 1.15, 1.10), "moving_obj": True},
        {"true_order": 2, "file": "harbor_3.jpg", "yaw": 7.0, "pitch": 0.2, "roll": -0.6, "exp": (0.80, 0.82, 0.85)},
        {"true_order": 3, "file": "harbor_4.jpg", "yaw": 21.0, "pitch": -0.3, "roll": 1.0, "exp": (1.14, 1.12, 1.08)},
    ]
    meta_4 = []
    for sp in specs_4:
        fr, H_gt, inside = render_camera_view(
            m4, f_m4, (cam_w4, cam_h4), f_c4, sp["yaw"], sp["pitch"], sp["roll"], sp["exp"], 0.7, rng
        )
        assert inside, f"Dataset 4 frame {sp['file']} exceeded master bounds!"
        if sp.get("moving_obj"):
            fr = inject_moving_boat(fr)
        save_image_unicode(d4 / sp["file"], fr, jpg_quality=94)
        meta_4.append({**sp, "H_cam_to_master": H_gt.tolist()})

    (d4 / "metadata.json").write_text(
        json.dumps(
            {
                "name": "04 - Chênh lệch sáng mạnh & Vật chuyển động (4 ảnh bến cảng)",
                "description": "Thử thách khó với phơi sáng lệch ±30% giữa các ảnh và ca-nô chuyển động trong vùng chồng lấn.",
                "master_size": [mw4, mh4],
                "frames": meta_4,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Dataset 5: Bộ kiểm thử lỗi - 3 ảnh không chồng lấn từ 3 bối cảnh khác nhau
    # ------------------------------------------------------------------
    d5 = datasets_root / "05_negative_non_overlapping"
    d5.mkdir(parents=True, exist_ok=True)
    save_image_unicode(d5 / "scene_lake.jpg", load_image_unicode(d1 / "img_01.jpg"))
    save_image_unicode(d5 / "scene_urban.jpg", load_image_unicode(d2 / "urban_1.jpg"))
    save_image_unicode(d5 / "scene_library.jpg", load_image_unicode(d3 / "library_center.jpg"))
    (d5 / "metadata.json").write_text(
        json.dumps(
            {
                "name": "05 - Kiểm thử lỗi: 3 ảnh không chồng lấn (Negative Test)",
                "description": "Gồm 3 ảnh từ 3 địa điểm hoàn toàn khác nhau nhằm kiểm chứng xử lý lỗi an toàn.",
                "expected_error": True,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Đã khởi tạo thành công toàn bộ 5 bộ dữ liệu kiểm thử tại:", datasets_root)


if __name__ == "__main__":
    build_all_datasets()
