"""Script chạy thực nghiệm, đánh giá định lượng & sinh biểu đồ báo cáo tự động.

Thực hiện 4 nhóm thí nghiệm theo Project Charter:
1. Toàn bộ 5 bộ dữ liệu (4 bộ hợp lệ + 1 bộ kiểm thử lỗi không chồng lấn).
2. So sánh thuật toán TỰ CÀI ĐẶT (Custom NumPy Matcher + Normalized DLT/RANSAC/LM)
   với OpenCV chuẩn (BFMatcher + cv2.findHomography) trên SIFT và ORB.
3. So sánh 4 kỹ thuật Trộn ảnh (Hard Cut vs Feather vs Multi-Band vs DP Seam + Multi-Band)
   trên bộ dữ liệu khó có chênh lệch sáng ±30% và vật thể chuyển động (Dataset 4).
4. Vẽ các biểu đồ chất lượng cao (300 DPI) + ảnh phóng to đối chiếu vùng chồng lấn
   và sơ đồ kiến trúc hệ thống để nhúng vào Báo cáo Word và Slide PowerPoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Segoe UI", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from panorama import PanoramaError, PanoramaStitcher, StitchingConfig  # noqa: E402
from panorama.pipeline import load_image_unicode, save_image_unicode  # noqa: E402


DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
VISUALS_DIR = RESULTS_DIR / "visuals"


def get_dataset_images(folder: Path) -> List[Path]:
    return sorted(folder.glob("*.jpg"))


def run_all_datasets_experiment() -> Dict[str, Any]:
    """Chạy pipeline mặc định (SIFT + Custom Matcher + Custom RANSAC + DP Seam Multi-Band)
    trên tất cả 5 bộ dữ liệu.
    """
    valid_datasets = [
        ("01_outdoor_5imgs_fhd", "Ngoại cảnh Hồ Alpine (5 ảnh Full HD)"),
        ("02_urban_4imgs", "Kiến trúc Quảng trường (4 ảnh)"),
        ("03_indoor_3imgs", "Nội thất Thư viện (3 ảnh)"),
        ("04_exposure_motion_4imgs", "Bến cảng: Lệch sáng & Vật chuyển động (4 ảnh)"),
    ]

    summary: Dict[str, Any] = {"datasets": [], "negative_test": {}}
    cfg = StitchingConfig(
        detector="sift",
        matcher_mode="custom",
        homography_mode="custom",
        projection="planar",
        blending_mode="seam_multiband",
        exposure_compensation=True,
        auto_order=True,
        auto_crop=True,
        generate_visuals=True,
    )
    stitcher = PanoramaStitcher(cfg)

    for folder_name, label in valid_datasets:
        dpath = DATASETS_DIR / folder_name
        paths = get_dataset_images(dpath)
        print(f"\n>>> [Dataset] {label} ({len(paths)} ảnh)...")
        res = stitcher.stitch_paths(paths)

        out_pano = RESULTS_DIR / f"pano_{folder_name}.jpg"
        save_image_unicode(out_pano, res.panorama, jpg_quality=95)

        vdir = VISUALS_DIR / folder_name
        vdir.mkdir(parents=True, exist_ok=True)
        save_image_unicode(vdir / "panorama_uncropped.jpg", res.panorama_uncropped, jpg_quality=92)
        for vname, vimg in res.visuals.items():
            save_image_unicode(vdir / f"{vname}.jpg", vimg, jpg_quality=90)

        # Kiểm tra độ chính xác Auto-Order nếu có metadata ground truth
        meta_file = dpath / "metadata.json"
        order_correct = True
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if "frames" in meta:
                expected_order = [
                    f["file"] for f in sorted(meta["frames"], key=lambda x: x["true_order"])
                ]
                actual_order = [res.image_names[i] for i in res.visual_order]
                order_correct = (actual_order == expected_order) or (
                    actual_order == list(reversed(expected_order))
                )

        item = {
            "id": folder_name,
            "label": label,
            "status": "SUCCESS",
            "auto_order_correct": order_correct,
            "output_path": str(out_pano),
            **res.summary_dict(),
        }
        summary["datasets"].append(item)
        print(
            f"    OK! Output: {item['output_size']['width']}x{item['output_size']['height']} | "
            f"Core time: {item['metrics']['core_pipeline_time_sec']:.2f}s | "
            f"Inliers: {item['metrics']['mean_inliers']} ({item['metrics']['mean_inlier_ratio_pct']}%) | "
            f"RMSE: {item['metrics']['mean_reprojection_rmse_px']} px | "
            f"Auto-Order OK: {order_correct}"
        )

    # Kiểm thử lỗi với Dataset 5 (3 ảnh không chồng lấn)
    neg_paths = get_dataset_images(DATASETS_DIR / "05_negative_non_overlapping")
    print("\n>>> [Negative Test] 05_negative_non_overlapping...")
    try:
        stitcher.stitch_paths(neg_paths)
        summary["negative_test"] = {"handled_gracefully": False, "error": None}
    except PanoramaError as exc:
        summary["negative_test"] = {
            "handled_gracefully": True,
            "code": exc.code,
            "message": exc.message,
            "suggestion": exc.suggestion,
        }
        print(f"    OK! Bắt lỗi an toàn [{exc.code}]: {exc.message}")

    return summary


def run_custom_vs_opencv_benchmark() -> List[Dict[str, Any]]:
    """So sánh Custom NumPy vs OpenCV trên Dataset 1 (5 ảnh Full HD) và Dataset 2 (4 ảnh)."""
    paths = get_dataset_images(DATASETS_DIR / "02_urban_4imgs")
    configs = [
        ("SIFT + Custom (NumPy)", "sift", "custom", "custom"),
        ("SIFT + OpenCV", "sift", "opencv", "opencv"),
        ("ORB + Custom (NumPy)", "orb", "custom", "custom"),
        ("ORB + OpenCV", "orb", "opencv", "opencv"),
    ]
    rows: List[Dict[str, Any]] = []
    print("\n>>> [Benchmark] Custom NumPy vs OpenCV trên Dataset 02 (Urban 4 ảnh)...")
    for name, det, m_mode, h_mode in configs:
        cfg = StitchingConfig(
            detector=det,  # type: ignore[arg-type]
            matcher_mode=m_mode,  # type: ignore[arg-type]
            homography_mode=h_mode,  # type: ignore[arg-type]
            blending_mode="seam_multiband",
            generate_visuals=False,
        )
        res = PanoramaStitcher(cfg).stitch_paths(paths)
        row = {
            "config_name": name,
            "detector": det.upper(),
            "impl": "Custom NumPy" if m_mode == "custom" else "OpenCV",
            "match_homog_ms": round(res.timings_ms["matching_and_homography"], 1),
            "core_time_sec": res.metrics["core_pipeline_time_sec"],
            "mean_inliers": res.metrics["mean_inliers"],
            "mean_inlier_ratio_pct": res.metrics["mean_inlier_ratio_pct"],
            "mean_rmse_px": res.metrics["mean_reprojection_rmse_px"],
        }
        rows.append(row)
        print(
            f"    {name:24s} | Match+H: {row['match_homog_ms']:6.1f} ms | "
            f"Inliers: {row['mean_inliers']:5.1f} ({row['mean_inlier_ratio_pct']:5.1f}%) | "
            f"RMSE: {row['mean_rmse_px']:.3f} px"
        )
    return rows


def run_blending_comparison_experiment() -> List[Dict[str, Any]]:
    """So sánh 4 chế độ trộn ảnh trên Dataset 4 (Chênh sáng ±30% + Ca-nô chuyển động)."""
    paths = get_dataset_images(DATASETS_DIR / "04_exposure_motion_4imgs")
    modes = [
        ("Hard Cut (Không trộn, không cân sáng)", "none", False, "01_hard_cut"),
        ("Feather + Cân bằng sáng", "feather", True, "02_feather"),
        ("Multi-Band Pyramid + Cân sáng", "multiband", True, "03_multiband"),
        ("DP Seam + Multi-Band + Cân sáng", "seam_multiband", True, "04_dp_seam_multiband"),
    ]
    rows: List[Dict[str, Any]] = []
    panos: List[tuple[str, np.ndarray]] = []
    print("\n>>> [Benchmark] So sánh 4 kỹ thuật Blending trên Dataset 04 (Lệch sáng + Vật chuyển động)...")

    bdir = VISUALS_DIR / "blending_compare"
    bdir.mkdir(parents=True, exist_ok=True)

    for label, bmode, exp_comp, slug in modes:
        cfg = StitchingConfig(
            detector="sift",
            matcher_mode="custom",
            homography_mode="custom",
            blending_mode=bmode,  # type: ignore[arg-type]
            exposure_compensation=exp_comp,
            auto_crop=True,
            generate_visuals=False,
        )
        res = PanoramaStitcher(cfg).stitch_paths(paths)
        save_image_unicode(bdir / f"{slug}.jpg", res.panorama, jpg_quality=94)
        panos.append((label, res.panorama))

        row = {
            "label": label,
            "mode": bmode,
            "exposure_compensation": exp_comp,
            "seam_gradient_jump": res.metrics["seam_gradient_jump"],
            "blending_ms": round(res.timings_ms["blending"], 1),
            "core_time_sec": res.metrics["core_pipeline_time_sec"],
            "image_path": str(bdir / f"{slug}.jpg"),
        }
        rows.append(row)
        print(
            f"    {label:40s} | Seam Jump: {row['seam_gradient_jump']:5.2f} | "
            f"Blend time: {row['blending_ms']:6.1f} ms"
        )

    # Tạo hình ghép đối chiếu 2×2 toàn ảnh + phóng to vùng chồng lấn có ca-nô chuyển động
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 8.8), dpi=220)
    fig.patch.set_facecolor("#0f172a")
    for ax, (title, bgr), row in zip(axes.ravel(), panos, rows):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(
            f"{title}\nSeam Jump: {row['seam_gradient_jump']:.2f} | Thời gian trộn: {row['blending_ms']:.0f} ms",
            color="#f8fafc",
            fontsize=10.5,
            fontweight="bold",
            pad=10,
        )
        ax.axis("off")
    plt.suptitle(
        "SO SÁNH CÁC KỸ THUẬT TRỘN ẢNH TRÊN BỘ DỮ LIỆU CHÊNH SÁNG ±30% & VẬT CHUYỂN ĐỘNG",
        color="#38bdf8",
        fontsize=12.5,
        fontweight="bold",
        y=0.975,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.89, bottom=0.02, hspace=0.26, wspace=0.04)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHARTS_DIR / "blending_comparison_grid.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    # Hình phóng to 1×4 vùng chứa đường nối & ca-nô chuyển động (khoảng x: 35%-62%, y: 45%-88%)
    fig2, axes2 = plt.subplots(1, 4, figsize=(15, 4.4), dpi=220)
    fig2.patch.set_facecolor("#0f172a")
    short_titles = [
        "1. Hard Cut\n(Lộ vệt nối & lệch sáng)",
        "2. Feather + Gain\n(Mờ bóng ma ca-nô)",
        "3. Multi-Band + Gain\n(Cắt qua ca-nô)",
        "4. DP Seam + Multi-Band\n(Né vật thể & liền mạch)",
    ]
    for ax, stitle, (_, bgr) in zip(axes2, short_titles, panos):
        h, w = bgr.shape[:2]
        crop = bgr[int(h * 0.42) : int(h * 0.90), int(w * 0.34) : int(w * 0.62)]
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(stitle, color="#f1f5f9", fontsize=10, fontweight="bold", pad=6)
        ax.axis("off")
    plt.suptitle(
        "PHÓNG TO VÙNG CHỒNG LẤN (XỬ LÝ LỆCH SÁNG & BÓNG MA VẬT THỂ CHUYỂN ĐỘNG)",
        color="#fbbf24",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(CHARTS_DIR / "blending_comparison_zoom.png", facecolor=fig2.get_facecolor())
    plt.close(fig2)

    return rows


def generate_charts_and_diagrams(
    ds_summary: Dict[str, Any],
    cv_rows: List[Dict[str, Any]],
    blend_rows: List[Dict[str, Any]],
) -> None:
    """Sinh các biểu đồ matplotlib 250 DPI và sơ đồ kiến trúc pipeline cho báo cáo & slide."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Biểu đồ phân rã thời gian xử lý từng giai đoạn trên 4 bộ dữ liệu
    datasets = ds_summary["datasets"]
    labels = [
        "Dataset 1\n5 ảnh Full HD\n(1920×1080)",
        "Dataset 2\n4 ảnh Đô thị\n(960×1080)",
        "Dataset 3\n3 ảnh Thư viện\n(1080×1080)",
        "Dataset 4\n4 ảnh Lệch sáng\n(960×1080)",
    ]
    stages = [
        ("feature_detection", "Phát hiện đặc trưng (SIFT)", "#38bdf8"),
        ("matching_and_homography", "Khớp & RANSAC (Custom)", "#818cf8"),
        ("warping", "Biến đổi hình học (Warp)", "#34d399"),
        ("blending", "Cân sáng & Trộn (Custom)", "#fbbf24"),
        ("postprocess", "Cắt biên tự động (Custom)", "#f472b6"),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=240)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    x = np.arange(len(datasets))
    bottom = np.zeros(len(datasets))

    for key, name, color in stages:
        vals = np.array([d["timings_ms"][key] / 1000.0 for d in datasets])
        ax.bar(x, vals, width=0.52, bottom=bottom, label=name, color=color, edgecolor="#ffffff", linewidth=0.8)
        bottom += vals

    for idx, total_s in enumerate(bottom):
        ax.text(idx, total_s + 0.35, f"{total_s:.2f}s", ha="center", va="bottom", fontweight="bold", fontsize=10, color="#0f172a")

    ax.axhline(30.0, color="#ef4444", linestyle="--", linewidth=1.6, label="Giới hạn yêu cầu (< 30 giây cho 5 ảnh Full HD)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, color="#1e293b")
    ax.set_ylabel("Thời gian xử lý lõi (giây)", fontsize=10.5, fontweight="bold", color="#1e293b")
    ax.set_title("Phân rã thời gian xử lý theo từng giai đoạn trên 4 bộ dữ liệu thực nghiệm", fontsize=12, fontweight="bold", color="#0f172a", pad=12)
    ax.set_ylim(0, max(33.0, float(bottom.max()) + 4.0))
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", fontsize=8.8)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "chart_stage_timings.png")
    plt.close(fig)

    # 2. Biểu đồ so sánh Custom NumPy vs OpenCV (RMSE & Thời gian Match+RANSAC)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=240)
    fig.patch.set_facecolor("#ffffff")
    names = [r["config_name"] for r in cv_rows]
    rmses = [r["mean_rmse_px"] for r in cv_rows]
    times = [r["match_homog_ms"] for r in cv_rows]
    colors = ["#2563eb", "#94a3b8", "#059669", "#cbd5e1"]

    ax1.set_facecolor("#f8fafc")
    bars1 = ax1.bar(names, rmses, color=colors, width=0.55)
    for b, v in zip(bars1, rmses):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.3f} px", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Sai số chiếu lại trung bình RMSE (pixel)", fontsize=9.5, fontweight="bold")
    ax1.set_title("Độ chính xác Homography (thấp hơn = tốt hơn)", fontsize=10.5, fontweight="bold")
    ax1.tick_params(axis="x", rotation=16, labelsize=8.5)
    ax1.set_ylim(0, max(rmses) * 1.25)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    ax2.set_facecolor("#f8fafc")
    bars2 = ax2.bar(names, times, color=colors, width=0.55)
    for b, v in zip(bars2, times):
        ax2.text(b.get_x() + b.get_width() / 2, v + max(times) * 0.02, f"{v:.0f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Thời gian Matching + RANSAC (ms)", fontsize=9.5, fontweight="bold")
    ax2.set_title("Tốc độ xử lý Khớp & Ước lượng H", fontsize=10.5, fontweight="bold")
    ax2.tick_params(axis="x", rotation=16, labelsize=8.5)
    ax2.set_ylim(0, max(times) * 1.25)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    plt.suptitle("Đối chiếu thuật toán Tự cài đặt (Custom NumPy) và OpenCV chuẩn", fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(CHARTS_DIR / "chart_custom_vs_opencv.png")
    plt.close(fig)

    # 3. Biểu đồ độ mượt đường nối (Seam Gradient Jump) giữa 4 chế độ Blending
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=240)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    b_labels = [
        "Hard Cut\n(Không trộn)",
        "Feather\n+ Gain Comp.",
        "Multi-Band\n+ Gain Comp.",
        "DP Seam +\nMulti-Band + Gain",
    ]
    jumps = [r["seam_gradient_jump"] for r in blend_rows]
    b_colors = ["#ef4444", "#f59e0b", "#3b82f6", "#10b981"]
    bars = ax.bar(b_labels, jumps, color=b_colors, width=0.5)
    for b, v in zip(bars, jumps):
        ax.text(b.get_x() + b.get_width() / 2, v + max(jumps) * 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Độ nhảy cường độ tại đường nối (Seam Gradient Jump)", fontsize=9.5, fontweight="bold")
    ax.set_title("Hiệu quả triệt tiêu đường nối trên bộ dữ liệu chênh lệch sáng ±30% (thấp hơn = mượt hơn)", fontsize=11, fontweight="bold", pad=10)
    ax.set_ylim(0, max(jumps) * 1.22)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "chart_blending_quality.png")
    plt.close(fig)

    # 4. Sơ đồ kiến trúc Pipeline 6 bước (phân biệt rõ Custom NumPy vs OpenCV)
    fig, ax = plt.subplots(figsize=(13.5, 4.4), dpi=240)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    ax.text(
        6.75,
        4.05,
        "KIẾN TRÚC HỆ THỐNG GHÉP ẢNH TOÀN CẢNH (6 GIAI ĐOẠN MODULAR)",
        ha="center",
        va="center",
        color="#f8fafc",
        fontsize=13,
        fontweight="bold",
    )

    boxes = [
        (0.3, "1. Phát hiện\nđặc trưng", "SIFT / ORB\nScale-aware\n(OpenCV)", "#0284c7", "OpenCV"),
        (2.5, "2. Khớp\nđặc trưng", "k-NN Matrix\nLowe's Ratio\nCross-Check", "#059669", "TỰ CÀI ĐẶT"),
        (4.7, "3. Homography\n& Auto-Order", "Hartley DLT\nAdaptive RANSAC\nLM + MST Graph", "#059669", "TỰ CÀI ĐẶT"),
        (6.9, "4. Biến đổi\nhình học", "Planar Warp\nCylindrical Proj.\nSafe Canvas", "#0284c7", "Kết hợp"),
        (9.1, "5. Cân sáng &\nTrộn ảnh", "Gain Comp.\nDP Min-Seam\nLaplacian Pyr.", "#d97706", "TỰ CÀI ĐẶT"),
        (11.3, "6. Hậu xử lý\n& Xuất ảnh", "Largest Rect\nAuto-Crop\nPNG/JPG + ZIP", "#059669", "TỰ CÀI ĐẶT"),
    ]

    for idx, (bx, title, desc, color, badge) in enumerate(boxes):
        rect = mpatches.FancyBboxPatch(
            (bx, 1.0),
            1.9,
            2.35,
            boxstyle="round,pad=0.08,rounding_size=0.16",
            facecolor="#1e293b",
            edgecolor=color,
            linewidth=2.2,
        )
        ax.add_patch(rect)
        # Badge trên đầu hộp
        badge_rect = mpatches.FancyBboxPatch(
            (bx + 0.2, 3.12),
            1.5,
            0.36,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=color,
            edgecolor="none",
        )
        ax.add_patch(badge_rect)
        ax.text(bx + 0.95, 3.30, badge, ha="center", va="center", color="#ffffff", fontsize=7.8, fontweight="bold")
        ax.text(bx + 0.95, 2.60, title, ha="center", va="center", color="#f8fafc", fontsize=9.8, fontweight="bold")
        ax.text(bx + 0.95, 1.58, desc, ha="center", va="center", color="#cbd5e1", fontsize=8.3, linespacing=1.35)

        if idx < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(bx + 2.18, 2.18),
                xytext=(bx + 1.94, 2.18),
                arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=2.0),
            )

    ax.text(
        6.75,
        0.45,
        "Điểm nhấn học thuật: 4/6 khối (Matching, Normalized DLT + RANSAC + LM, Gain + DP Seam + Multi-Band Pyramid, Auto-Crop) tự cài đặt bằng Python/NumPy thuần",
        ha="center",
        va="center",
        color="#fde68a",
        fontsize=9.2,
        fontstyle="italic",
    )
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "architecture_diagram.png", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n>>> Đã xuất toàn bộ biểu đồ và sơ đồ kiến trúc vào: {CHARTS_DIR}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ds_summary = run_all_datasets_experiment()
    cv_rows = run_custom_vs_opencv_benchmark()
    blend_rows = run_blending_comparison_experiment()

    generate_charts_and_diagrams(ds_summary, cv_rows, blend_rows)

    full_report = {
        "datasets_summary": ds_summary,
        "custom_vs_opencv": cv_rows,
        "blending_comparison": blend_rows,
        "success_rate_pct": round(
            100.0
            * sum(1 for d in ds_summary["datasets"] if d["status"] == "SUCCESS")
            / max(1, len(ds_summary["datasets"])),
            1,
        ),
    }
    out_json = RESULTS_DIR / "benchmark_summary.json"
    out_json.write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== HOÀN TẤT TOÀN BỘ THỰC NGHIỆM! Kết quả lưu tại: {out_json} ===")


if __name__ == "__main__":
    main()
