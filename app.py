"""Ứng dụng Web (FastAPI) cho Hệ thống Ghép ảnh Toàn cảnh (Panorama Stitching)."""

from __future__ import annotations

import json
import shutil
import sys
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from panorama import PanoramaError, PanoramaStitcher, StitchingConfig  # noqa: E402
from panorama.pipeline import load_image_unicode, save_image_unicode  # noqa: E402

WEB_DIR = PROJECT_ROOT / "web"
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"
JOBS_DIR = PROJECT_ROOT / "web_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="PanoramaStudio — Hệ thống Ghép ảnh Toàn cảnh",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
if RESULTS_DIR.exists():
    app.mount("/results-files", StaticFiles(directory=str(RESULTS_DIR)), name="results-files")

# Lưu trạng thái các phiên xử lý trong bộ nhớ (thread-safe)
JOBS_LOCK = threading.Lock()
JOBS: Dict[str, Dict[str, Any]] = {}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _safe_filename(name: str) -> str:
    return Path(name).name


def _list_dataset_entries() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not DATASETS_DIR.exists():
        return entries

    for folder in sorted(DATASETS_DIR.iterdir()):
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        imgs = sorted(
            f.name for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )
        if not imgs:
            continue
        meta_path = folder / "metadata.json"
        meta = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.exists()
            else {}
        )
        entries.append(
            {
                "id": folder.name,
                "name": meta.get("name", folder.name),
                "description": meta.get("description", ""),
                "expected_error": bool(meta.get("expected_error", False)),
                "num_images": len(imgs),
                "files": imgs,
                "thumbnails": [f"/api/datasets/{folder.name}/image/{fn}" for fn in imgs],
            }
        )
    return entries


@app.get("/", response_class=HTMLResponse)
def serve_index() -> HTMLResponse:
    index_path = WEB_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/api/datasets")
def api_list_datasets() -> JSONResponse:
    return JSONResponse({"datasets": _list_dataset_entries()})


@app.get("/api/datasets/{dataset_id}/image/{filename}")
def api_dataset_image(dataset_id: str, filename: str) -> FileResponse:
    safe_ds = _safe_filename(dataset_id)
    safe_fn = _safe_filename(filename)
    target = DATASETS_DIR / safe_ds / safe_fn
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh mẫu.")
    return FileResponse(str(target))


def _run_stitch_job(
    job_id: str,
    images: List[np.ndarray],
    names: List[str],
    config: StitchingConfig,
) -> None:
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    def _on_progress(stage: str, pct: int, msg: str) -> None:
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["stage"] = stage
                JOBS[job_id]["progress"] = pct
                JOBS[job_id]["message"] = msg

    try:
        stitcher = PanoramaStitcher(config)
        res = stitcher.stitch(images, image_names=names, progress_cb=_on_progress)

        # Lưu ảnh đầu ra PNG, JPG và bản chưa cắt biên
        save_image_unicode(job_dir / "panorama.png", res.panorama)
        save_image_unicode(job_dir / "panorama.jpg", res.panorama, jpg_quality=95)
        save_image_unicode(
            job_dir / "panorama_uncropped.jpg", res.panorama_uncropped, jpg_quality=92
        )

        visual_entries: Dict[str, Any] = {
            "keypoints": [],
            "matches": [],
            "match_graph": None,
            "warped_layout": None,
        }
        for key, img in res.visuals.items():
            fn = f"{key}.jpg"
            save_image_unicode(job_dir / fn, img, jpg_quality=90)
            url = f"/api/jobs/{job_id}/file/{fn}"
            if key.startswith("keypoints_"):
                idx = int(key.split("_")[1])
                visual_entries["keypoints"].append(
                    {"index": idx, "label": f"#{idx + 1} {names[idx]}", "url": url}
                )
            elif key.startswith("matches_"):
                idx = int(key.split("_")[1])
                pair_meta = res.metrics["mst_pairs"][idx] if idx < len(res.metrics["mst_pairs"]) else {}
                lbl = (
                    f"{pair_meta.get('src', '')} ↔ {pair_meta.get('dst', '')} "
                    f"({pair_meta.get('inliers', 0)} inliers)"
                )
                visual_entries["matches"].append({"index": idx, "label": lbl, "url": url})
            elif key == "match_graph":
                visual_entries["match_graph"] = url
            elif key == "warped_layout":
                visual_entries["warped_layout"] = url

        visual_entries["keypoints"].sort(key=lambda x: x["index"])
        visual_entries["matches"].sort(key=lambda x: x["index"])

        summary = res.summary_dict()
        (job_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Đóng gói ZIP gồm ảnh kết quả + toàn bộ ảnh trung gian + báo cáo JSON
        zip_path = job_dir / "panorama_report_bundle.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in sorted(job_dir.iterdir()):
                if item.name == zip_path.name:
                    continue
                zf.write(item, arcname=item.name)

        with JOBS_LOCK:
            JOBS[job_id].update(
                {
                    "status": "done",
                    "stage": "done",
                    "progress": 100,
                    "message": f"Hoàn tất ghép {len(images)} ảnh trong {summary['metrics']['core_pipeline_time_sec']:.2f} giây!",
                    "summary": summary,
                    "outputs": {
                        "panorama_jpg": f"/api/jobs/{job_id}/file/panorama.jpg",
                        "panorama_png": f"/api/jobs/{job_id}/file/panorama.png",
                        "panorama_uncropped": f"/api/jobs/{job_id}/file/panorama_uncropped.jpg",
                        "download_jpg": f"/api/jobs/{job_id}/file/panorama.jpg?download=1",
                        "download_png": f"/api/jobs/{job_id}/file/panorama.png?download=1",
                        "download_zip": f"/api/jobs/{job_id}/file/panorama_report_bundle.zip?download=1",
                    },
                    "visuals": visual_entries,
                }
            )

    except PanoramaError as exc:
        with JOBS_LOCK:
            JOBS[job_id].update(
                {
                    "status": "error",
                    "stage": "error",
                    "progress": 100,
                    "error": exc.to_dict(),
                }
            )
    except Exception as exc:  # pragma: no cover - safety guard
        with JOBS_LOCK:
            JOBS[job_id].update(
                {
                    "status": "error",
                    "stage": "error",
                    "progress": 100,
                    "error": {
                        "code": "UNEXPECTED_ERROR",
                        "message": f"Đã xảy ra lỗi không mong đợi: {exc}",
                        "suggestion": "Vui lòng kiểm tra lại định dạng ảnh hoặc thử cấu hình khác.",
                        "details": {},
                    },
                }
            )


@app.post("/api/stitch")
async def api_start_stitch(
    config_json: str = Form("{}"),
    dataset_id: Optional[str] = Form(None),
    dataset_files_json: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
) -> JSONResponse:
    try:
        cfg_dict = json.loads(config_json) if config_json else {}
        config = StitchingConfig.from_dict(cfg_dict)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cấu hình JSON không hợp lệ: {exc}")

    images: List[np.ndarray] = []
    names: List[str] = []

    if dataset_id:
        ds_dir = DATASETS_DIR / _safe_filename(dataset_id)
        if not ds_dir.exists():
            raise HTTPException(status_code=404, detail="Không tìm thấy bộ dữ liệu mẫu.")
        if dataset_files_json:
            requested = json.loads(dataset_files_json)
            paths = [ds_dir / _safe_filename(fn) for fn in requested]
        else:
            paths = sorted(
                f for f in ds_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            )
        for p in paths:
            try:
                images.append(load_image_unicode(p))
                names.append(p.name)
            except PanoramaError as exc:
                return JSONResponse(status_code=400, content={"error": exc.to_dict()})
    elif files:
        for up in files:
            raw = await up.read()
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "code": "IMAGE_LOAD_ERROR",
                            "message": f"Không thể đọc tệp ảnh '{up.filename}'.",
                            "suggestion": "Vui lòng chọn tệp ảnh hợp lệ (JPG, PNG, WEBP, BMP).",
                        }
                    },
                )
            images.append(img)
            names.append(up.filename or f"upload_{len(images)}.jpg")
    else:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "INSUFFICIENT_IMAGES",
                    "message": "Chưa có ảnh đầu vào nào được chọn.",
                    "suggestion": "Hãy tải lên từ 2–10 ảnh hoặc chọn nhanh 1 bộ dữ liệu mẫu.",
                }
            },
        )

    # Dọn bớt job cũ nếu quá 15 phiên để tránh đầy ổ đĩa
    existing_dirs = sorted(
        [d for d in JOBS_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
    )
    for old_dir in existing_dirs[:-15]:
        shutil.rmtree(old_dir, ignore_errors=True)

    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "running",
            "stage": "queued",
            "progress": 2,
            "message": f"Đã tiếp nhận {len(images)} ảnh. Bắt đầu xử lý...",
        }

    worker = threading.Thread(
        target=_run_stitch_job,
        args=(job_id, images, names, config),
        daemon=True,
    )
    worker.start()
    return JSONResponse({"job_id": job_id})


@app.get("/api/jobs/{job_id}")
def api_job_status(job_id: str) -> JSONResponse:
    with JOBS_LOCK:
        data = JOBS.get(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên ghép ảnh.")
    return JSONResponse(data)


@app.get("/api/jobs/{job_id}/file/{filename}")
def api_job_file(job_id: str, filename: str, download: int = 0) -> FileResponse:
    safe_job = _safe_filename(job_id)
    safe_fn = _safe_filename(filename)
    path = JOBS_DIR / safe_job / safe_fn
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy tệp kết quả.")
    if download:
        return FileResponse(str(path), filename=safe_fn)
    return FileResponse(str(path))


@app.get("/api/benchmarks")
def api_benchmarks() -> JSONResponse:
    summary_file = RESULTS_DIR / "benchmark_summary.json"
    if not summary_file.exists():
        return JSONResponse({"available": False})
    data = json.loads(summary_file.read_text(encoding="utf-8"))
    charts = {
        "architecture": "/results-files/charts/architecture_diagram.png",
        "stage_timings": "/results-files/charts/chart_stage_timings.png",
        "custom_vs_opencv": "/results-files/charts/chart_custom_vs_opencv.png",
        "blending_quality": "/results-files/charts/chart_blending_quality.png",
        "blending_grid": "/results-files/charts/blending_comparison_grid.png",
        "blending_zoom": "/results-files/charts/blending_comparison_zoom.png",
    }
    return JSONResponse({"available": True, "data": data, "charts": charts})


if __name__ == "__main__":
    import uvicorn
    import webbrowser

    host = "127.0.0.1"
    port = 8000
    print(f"=== Khởi động PanoramaStudio tại http://{host}:{port} ===")
    threading.Timer(1.2, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run("app:app", host=host, port=port, reload=False)
