"""Giao diện dòng lệnh (CLI) cho Hệ thống Ghép ảnh Toàn cảnh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from panorama import PanoramaError, PanoramaStitcher, StitchingConfig  # noqa: E402
from panorama.pipeline import save_image_unicode  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hệ thống Ghép ảnh Toàn cảnh (Panorama Stitching) - CLI"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Danh sách đường dẫn ảnh (2-10 ảnh) hoặc 1 thư mục chứa các ảnh.",
    )
    parser.add_argument(
        "-o", "--output", default="output_panorama.jpg", help="Đường dẫn ảnh kết quả."
    )
    parser.add_argument(
        "--visuals-dir", default=None, help="Thư mục lưu các ảnh trung gian (tùy chọn)."
    )
    parser.add_argument("--detector", choices=["sift", "orb"], default="sift")
    parser.add_argument("--matcher", choices=["custom", "opencv"], default="custom")
    parser.add_argument("--homography", choices=["custom", "opencv"], default="custom")
    parser.add_argument(
        "--projection", choices=["planar", "cylindrical"], default="planar"
    )
    parser.add_argument(
        "--blending",
        choices=["seam_multiband", "multiband", "feather", "none"],
        default="seam_multiband",
    )
    parser.add_argument("--no-auto-order", action="store_true")
    parser.add_argument("--no-exposure", action="store_true")
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-thresh", type=float, default=4.0)
    return parser.parse_args()


def collect_paths(raw_inputs: list[str]) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    paths: list[Path] = []
    for item in raw_inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(
                sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts)
            )
        else:
            paths.append(p)
    return paths


def main() -> int:
    args = parse_args()
    paths = collect_paths(args.inputs)

    cfg = StitchingConfig(
        detector=args.detector,
        matcher_mode=args.matcher,
        homography_mode=args.homography,
        projection=args.projection,
        blending_mode=args.blending,
        auto_order=not args.no_auto_order,
        exposure_compensation=not args.no_exposure,
        auto_crop=not args.no_crop,
        ratio_threshold=args.ratio,
        ransac_threshold=args.ransac_thresh,
        generate_visuals=args.visuals_dir is not None,
    )

    stitcher = PanoramaStitcher(cfg)

    def _progress(stage: str, pct: int, msg: str) -> None:
        print(f"[{pct:3d}%] ({stage}) {msg}")

    try:
        result = stitcher.stitch_paths(paths, progress_cb=_progress)
    except PanoramaError as exc:
        print(f"\n[LỖI - {exc.code}] {exc.message}")
        if exc.suggestion:
            print(f"[GỢI Ý] {exc.suggestion}")
        return 1

    save_image_unicode(args.output, result.panorama)
    print(f"\nĐã lưu ảnh toàn cảnh: {args.output}")
    print(json.dumps(result.summary_dict(), indent=2, ensure_ascii=False))

    if args.visuals_dir:
        vdir = Path(args.visuals_dir)
        vdir.mkdir(parents=True, exist_ok=True)
        for name, img in result.visuals.items():
            save_image_unicode(vdir / f"{name}.jpg", img)
        save_image_unicode(vdir / "panorama_uncropped.jpg", result.panorama_uncropped)
        print(f"Đã lưu {len(result.visuals)} ảnh minh họa trung gian vào: {vdir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
