"""Cấu hình tham số cho pipeline ghép ảnh toàn cảnh."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal


DetectorType = Literal["sift", "orb"]
ImplMode = Literal["custom", "opencv"]
ProjectionType = Literal["planar", "cylindrical"]
BlendingMode = Literal["seam_multiband", "multiband", "feather", "none"]


@dataclass
class StitchingConfig:
    """Tham số cấu hình toàn bộ quy trình ghép ảnh toàn cảnh."""

    # 1. Phát hiện đặc trưng (Feature Detection)
    detector: DetectorType = "sift"
    max_features: int = 2500
    detection_max_dim: int = 1200  # Resize nội bộ khi detect để tối ưu tốc độ trên ảnh Full HD

    # 2. Khớp đặc trưng (Feature Matching - Tự cài đặt bằng NumPy hoặc OpenCV)
    matcher_mode: ImplMode = "custom"
    ratio_threshold: float = 0.75
    cross_check: bool = False

    # 3. Ước lượng Homography (Tự cài đặt Normalized DLT + RANSAC hoặc OpenCV)
    homography_mode: ImplMode = "custom"
    ransac_threshold: float = 4.0
    ransac_max_iters: int = 2000
    ransac_confidence: float = 0.995
    min_inliers: int = 15
    min_inlier_ratio: float = 0.12

    # 4. Sắp xếp ảnh & Chiếu hình học (Auto-Ordering & Projection)
    auto_order: bool = True
    projection: ProjectionType = "planar"
    cylindrical_fov_deg: float = 55.0

    # 5. Trộn ảnh & Cân bằng sáng (Tự cài đặt - Điểm nhấn kỹ thuật)
    exposure_compensation: bool = True
    blending_mode: BlendingMode = "seam_multiband"
    multiband_levels: int = 5

    # 6. Hậu xử lý & Bảo vệ bộ nhớ
    auto_crop: bool = True
    max_canvas_megapixels: float = 36.0
    max_input_images: int = 10

    # 7. Xuất ảnh trung gian phục vụ báo cáo / giao diện
    generate_visuals: bool = True
    visual_max_dim: int = 1280
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StitchingConfig":
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
