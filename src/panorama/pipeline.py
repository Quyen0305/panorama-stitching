"""Pipeline tổng thể điều phối 6 bước ghép ảnh toàn cảnh (Panorama Stitching)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .blending import BlendingResult, blend_warped_frames
from .config import StitchingConfig
from .exceptions import ImageLoadError, InsufficientImagesError
from .features import (
    ImageFeatures,
    detect_and_describe,
    draw_keypoints_preview,
    draw_matches_preview,
)
from .ordering import AlignmentGraph, build_alignment_graph, draw_match_graph_preview
from .postprocess import auto_crop_largest_rectangle
from .warping import (
    WarpedFrame,
    cylindrical_project_image,
    draw_warped_layout_preview,
    warp_all_images,
)


ProgressCallback = Callable[[str, int, str], None]


@dataclass
class StitchingResult:
    """Gói kết quả đầy đủ sau khi chạy pipeline ghép ảnh toàn cảnh."""

    panorama: np.ndarray  # Ảnh kết quả cuối (đã cắt biên nếu auto_crop=True)
    panorama_uncropped: np.ndarray  # Ảnh toàn cảnh trước khi cắt biên đen
    valid_mask: np.ndarray
    crop_box: Tuple[int, int, int, int]
    config: StitchingConfig
    image_names: List[str]
    visual_order: List[int]
    anchor_index: int
    exposure_gains: Dict[int, List[float]]
    timings_ms: Dict[str, float]
    metrics: Dict[str, Any]
    visuals: Dict[str, np.ndarray] = field(default_factory=dict)

    def summary_dict(self) -> Dict[str, Any]:
        """Trả về siêu dữ liệu gọn nhẹ dạng JSON phục vụ API và báo cáo."""
        h, w = self.panorama.shape[:2]
        uh, uw = self.panorama_uncropped.shape[:2]
        return {
            "output_size": {"width": w, "height": h},
            "uncropped_size": {"width": uw, "height": uh},
            "crop_box": list(self.crop_box),
            "num_images": len(self.image_names),
            "image_names": self.image_names,
            "visual_order": self.visual_order,
            "visual_order_names": [self.image_names[i] for i in self.visual_order],
            "anchor_index": self.anchor_index,
            "anchor_name": self.image_names[self.anchor_index],
            "exposure_gains": {
                self.image_names[k]: v for k, v in self.exposure_gains.items()
            },
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
            "metrics": self.metrics,
            "config": self.config.to_dict(),
        }


def load_image_unicode(path: Union[str, Path]) -> np.ndarray:
    """Đọc ảnh an toàn hỗ trợ cả đường dẫn tiếng Việt Unicode trên Windows."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise ImageLoadError(
            message=f"Không tìm thấy tệp ảnh: '{p.name}'.",
            suggestion="Kiểm tra lại đường dẫn tệp ảnh đầu vào.",
        )
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ImageLoadError(
            message=f"Tệp ảnh '{p.name}' bị hỏng hoặc không đúng định dạng ảnh (JPG/PNG/BMP/WEBP).",
            suggestion="Hãy thử mở lại tệp bằng trình xem ảnh hoặc xuất lại sang JPG/PNG.",
        )
    return img


def save_image_unicode(
    path: Union[str, Path], image: np.ndarray, jpg_quality: int = 94
) -> None:
    """Lưu ảnh an toàn hỗ trợ đường dẫn Unicode trên Windows."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower() or ".png"
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(jpg_quality, 50, 100))]
    elif ext == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise IOError(f"Không thể mã hóa ảnh định dạng {ext}")
    buf.tofile(str(p))


class PanoramaStitcher:
    """Bộ ghép ảnh toàn cảnh hoàn chỉnh hỗ trợ 2–10 ảnh, tự sắp xếp và trộn đa dải tần."""

    def __init__(self, config: Optional[StitchingConfig] = None) -> None:
        self.config = config or StitchingConfig()

    def stitch_paths(
        self,
        paths: List[Union[str, Path]],
        progress_cb: Optional[ProgressCallback] = None,
    ) -> StitchingResult:
        images: List[np.ndarray] = []
        names: List[str] = []
        for p in paths:
            path_obj = Path(p)
            images.append(load_image_unicode(path_obj))
            names.append(path_obj.name)
        return self.stitch(images, image_names=names, progress_cb=progress_cb)

    def stitch(
        self,
        images: List[np.ndarray],
        image_names: Optional[List[str]] = None,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> StitchingResult:
        """Chạy đầy đủ 6 giai đoạn ghép ảnh toàn cảnh."""
        cfg = self.config
        t_start = time.perf_counter()
        timings: Dict[str, float] = {}

        def _report(stage: str, pct: int, msg: str) -> None:
            if progress_cb is not None:
                progress_cb(stage, pct, msg)

        if images is None or len(images) < 2:
            raise InsufficientImagesError(
                message=f"Cần tối thiểu 2 ảnh để ghép toàn cảnh (hiện chỉ có {0 if not images else len(images)} ảnh).",
                suggestion="Hãy chọn từ 2 đến 10 ảnh có vùng chồng lấn nhau 20–50%.",
            )
        if len(images) > cfg.max_input_images:
            raise InsufficientImagesError(
                message=f"Số lượng ảnh vượt quá giới hạn cho phép ({len(images)} > {cfg.max_input_images}).",
                suggestion=f"Hãy chọn tối đa {cfg.max_input_images} ảnh trong một phiên ghép để đảm bảo tốc độ và bộ nhớ.",
            )

        names = image_names or [f"image_{i + 1}.jpg" for i in range(len(images))]
        work_images: List[np.ndarray] = []
        work_masks: List[Optional[np.ndarray]] = []

        # --- Bước 0: Tiền xử lý / Chiếu trụ (nếu chọn cylindrical) ---
        _report("preprocess", 5, "Đang chuẩn bị ảnh đầu vào...")
        t0 = time.perf_counter()
        for img in images:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            if cfg.projection == "cylindrical":
                c_img, c_mask = cylindrical_project_image(img, fov_deg=cfg.cylindrical_fov_deg)
                work_images.append(c_img)
                work_masks.append(c_mask)
            else:
                work_images.append(img)
                work_masks.append(None)
        timings["preprocess"] = (time.perf_counter() - t0) * 1000.0

        # --- Bước 1: Phát hiện đặc trưng (Feature Detection) ---
        _report("features", 18, f"Đang phát hiện đặc trưng ({cfg.detector.upper()}) trên {len(work_images)} ảnh...")
        t1 = time.perf_counter()
        features_list: List[ImageFeatures] = []
        for idx, (img, msk) in enumerate(zip(work_images, work_masks)):
            feat = detect_and_describe(
                img,
                image_index=idx,
                detector=cfg.detector,
                max_features=cfg.max_features,
                detection_max_dim=cfg.detection_max_dim,
                mask=msk,
            )
            features_list.append(feat)
        timings["feature_detection"] = (time.perf_counter() - t1) * 1000.0

        # --- Bước 2 & 3: Khớp đặc trưng, Tự sắp xếp ảnh & Ước lượng Homography (RANSAC) ---
        _report(
            "matching",
            40,
            f"Đang khớp đặc trưng ({cfg.matcher_mode}) & ước lượng Homography ({cfg.homography_mode})...",
        )
        t2 = time.perf_counter()
        shapes = [(img.shape[0], img.shape[1]) for img in work_images]
        graph: AlignmentGraph = build_alignment_graph(
            features_list=features_list,
            image_shapes=shapes,
            image_names=names,
            config=cfg,
        )
        timings["matching_and_homography"] = (time.perf_counter() - t2) * 1000.0

        # --- Bước 4: Biến đổi & căn chỉnh hình học (Warping) ---
        _report("warping", 62, "Đang biến đổi phối cảnh (Warping) các ảnh về hệ tọa độ chung...")
        t3 = time.perf_counter()
        warped_frames, canvas_size, out_scale = warp_all_images(
            images=work_images,
            global_homographies=graph.global_homographies,
            input_masks=work_masks,
            max_canvas_megapixels=cfg.max_canvas_megapixels,
        )
        timings["warping"] = (time.perf_counter() - t3) * 1000.0

        # --- Bước 5: Cân bằng sáng & Trộn ảnh (Blending - Tự cài đặt) ---
        _report("blending", 80, f"Đang cân bằng sáng và trộn ảnh ({cfg.blending_mode})...")
        t4 = time.perf_counter()
        blend_res: BlendingResult = blend_warped_frames(
            frames=warped_frames,
            visual_order=graph.visual_order,
            blending_mode=cfg.blending_mode,
            exposure_compensation=cfg.exposure_compensation,
            multiband_levels=cfg.multiband_levels,
        )
        timings["blending"] = (time.perf_counter() - t4) * 1000.0

        # --- Bước 6: Hậu xử lý & Cắt biên đen tự động ---
        _report("postprocess", 92, "Đang hậu xử lý và cắt biên tự động...")
        t5 = time.perf_counter()
        pano_uncropped = blend_res.panorama
        if cfg.auto_crop:
            pano_final, crop_box = auto_crop_largest_rectangle(
                pano_uncropped, blend_res.valid_mask
            )
        else:
            pano_final = pano_uncropped.copy()
            crop_box = (0, 0, pano_uncropped.shape[1], pano_uncropped.shape[0])
        timings["postprocess"] = (time.perf_counter() - t5) * 1000.0

        # --- Tạo ảnh minh họa các bước trung gian (nếu bật) ---
        visuals: Dict[str, np.ndarray] = {}
        t6 = time.perf_counter()
        if cfg.generate_visuals:
            _report("visuals", 96, "Đang kết xuất ảnh minh họa các bước trung gian...")
            for idx in range(len(work_images)):
                visuals[f"keypoints_{idx}"] = draw_keypoints_preview(
                    work_images[idx], features_list[idx], max_dim=min(900, cfg.visual_max_dim)
                )
            for e_idx, edge in enumerate(graph.mst_edges):
                title = f"#{edge.src_idx + 1} ({names[edge.src_idx]}) <-> #{edge.dst_idx + 1} ({names[edge.dst_idx]})"
                visuals[f"matches_{e_idx}"] = draw_matches_preview(
                    work_images[edge.src_idx],
                    work_images[edge.dst_idx],
                    edge.pts_src,
                    edge.pts_dst,
                    inlier_mask=edge.inlier_mask,
                    max_width=cfg.visual_max_dim,
                    title=title,
                )
            visuals["match_graph"] = draw_match_graph_preview(graph, names)
            visuals["warped_layout"] = draw_warped_layout_preview(
                warped_frames,
                canvas_size,
                seam_points=blend_res.seam_polylines,
                max_width=cfg.visual_max_dim,
            )
        timings["visualizations"] = (time.perf_counter() - t6) * 1000.0
        timings["total"] = (time.perf_counter() - t_start) * 1000.0

        # Tổng hợp các chỉ số kỹ thuật đánh giá chất lượng
        mst_inliers = [e.num_inliers for e in graph.mst_edges]
        mst_ratios = [e.inlier_ratio for e in graph.mst_edges]
        mst_rmses = [e.reprojection_rmse for e in graph.mst_edges]
        total_kps = [len(f.keypoints) for f in features_list]

        metrics: Dict[str, Any] = {
            "keypoints_per_image": total_kps,
            "mean_keypoints": round(float(np.mean(total_kps)), 1),
            "mst_pairs": [
                {
                    "src": names[e.src_idx],
                    "dst": names[e.dst_idx],
                    "matches": e.num_matches,
                    "inliers": e.num_inliers,
                    "inlier_ratio_pct": round(e.inlier_ratio * 100.0, 2),
                    "reprojection_rmse_px": round(e.reprojection_rmse, 3),
                }
                for e in graph.mst_edges
            ],
            "mean_inliers": round(float(np.mean(mst_inliers)), 1),
            "mean_inlier_ratio_pct": round(float(np.mean(mst_ratios) * 100.0), 2),
            "mean_reprojection_rmse_px": round(float(np.mean(mst_rmses)), 3),
            "seam_gradient_jump": round(blend_res.seam_gradient_jump, 3),
            "canvas_scale": round(out_scale, 4),
            "crop_retained_pct": round(
                100.0
                * (pano_final.shape[0] * pano_final.shape[1])
                / max(1, int(np.count_nonzero(blend_res.valid_mask))),
                2,
            ),
            "processing_time_sec": round(timings["total"] / 1000.0, 3),
            "core_pipeline_time_sec": round(
                (timings["total"] - timings["visualizations"]) / 1000.0, 3
            ),
        }

        _report("done", 100, f"Hoàn thành ghép {len(images)} ảnh trong {metrics['core_pipeline_time_sec']:.2f}s!")

        return StitchingResult(
            panorama=pano_final,
            panorama_uncropped=pano_uncropped,
            valid_mask=blend_res.valid_mask,
            crop_box=crop_box,
            config=cfg,
            image_names=names,
            visual_order=graph.visual_order,
            anchor_index=graph.anchor_idx,
            exposure_gains=blend_res.exposure_gains,
            timings_ms=timings,
            metrics=metrics,
            visuals=visuals,
        )
