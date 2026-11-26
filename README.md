# PanoramaStudio — Hệ thống Ghép ảnh Toàn cảnh (Panorama Stitching)

> **Bài tập lớn môn Xử lý ảnh số**  
> Kết hợp **4/6 mô-đun thuật toán lõi tự cài đặt bằng Python/NumPy/SciPy** cùng OpenCV, ứng dụng Web tương tác **PanoramaStudio**, bộ 5 dữ liệu kiểm thử không bản quyền, kiểm thử tự động `pytest`, Báo cáo Word và Slide PowerPoint hoàn chỉnh.

---

## 1. Điểm nhấn kỹ thuật & Kết quả đạt được

| Tiêu chí | Yêu cầu Project Charter | Kết quả đạt được |
|---|---|---|
| **Tự cài đặt thuật toán lõi** | Tối thiểu 2/5 bước (không gọi `cv2.Stitcher`) | **4/6 mô-đun cốt lõi tự cài đặt bằng NumPy/SciPy**: Matching (k-NN + Lowe's Ratio), Homography (Hartley DLT + Adaptive RANSAC + LM + Auto-Order MST), Blending (Exposure Gain + DP Seam + Multi-Band Laplacian Pyramid), Auto-Crop (`O(H·W)`) |
| **Hiệu năng Full HD** | 5 ảnh ~1920×1080 trong `< 30 giây` | **9.21 giây** (nhanh hơn gấp 3× giới hạn) |
| **Độ chính xác Homography** | Không lệch hình, đối chiếu OpenCV | **0.158 px RMSE** trên bộ Urban (tương đương tuyệt đối `cv2.findHomography` C++) |
| **Chất lượng trộn ảnh** | Không lộ vết sáng, không ghosting | Cân bằng phơi sáng ±30% mượt mà; DP Seam né hoàn toàn vật thể chuyển động |
| **Tỷ lệ ghép thành công** | ≥ 80% trên ≥ 3 bộ dữ liệu | **100% (4/4 bộ hợp lệ)** + **100% Auto-Order đúng** + bắt lỗi an toàn trên bộ âm tính |
| **Kiểm thử tự động** | Không crash khi lỗi đầu vào | **9/9 unit tests `pytest` đạt 100%** |

---

## 2. Cấu trúc thư mục dự án

```text
panorama-stitching/
├── app.py                          # Máy chủ ứng dụng Web FastAPI (PanoramaStudio)
├── cli.py                          # Giao diện dòng lệnh (CLI)
├── run_app.bat                     # Khởi chạy nhanh 1 chạm trên Windows
├── requirements.txt                # Danh sách thư viện Python
├── src/panorama/
│   ├── config.py                   # Cấu hình tham số pipeline (StitchingConfig)
│   ├── exceptions.py               # Hệ thống ngoại lệ có mã lỗi & gợi ý tiếng Việt
│   ├── features.py                 # SIFT/ORB + TỰ CÀI ĐẶT NumPy k-NN & Lowe's Ratio Test
│   ├── homography.py               # TỰ CÀI ĐẶT Hartley DLT + Adaptive RANSAC/MSAC + LM Jacobian
│   ├── ordering.py                 # TỰ CÀI ĐẶT Auto-Order (Match Graph + Kruskal MST + Center Anchor)
│   ├── warping.py                  # Planar Warp + TỰ CÀI ĐẶT Cylindrical Projection & Safe Canvas
│   ├── blending.py                 # TỰ CÀI ĐẶT Exposure Gain + DP Min-Seam + Feather + Multi-Band Pyramid
│   ├── postprocess.py              # TỰ CÀI ĐẶT Auto-Crop Hình chữ nhật nội tiếp lớn nhất O(H·W)
│   └── pipeline.py                 # Bộ điều phối 6 bước + xuất ảnh trung gian & chỉ số KPI
├── web/                            # Giao diện Web Studio (index.html, styles.css, app.js)
├── datasets/                       # 5 bộ dữ liệu kiểm thử 100% không bản quyền
│   ├── 01_outdoor_5imgs_fhd/       # 5 ảnh Full HD 1920×1080 bị xáo trộn
│   ├── 02_urban_4imgs/             # 4 ảnh kiến trúc quảng trường châu Âu
│   ├── 03_indoor_3imgs/            # 3 ảnh nội thất thư viện đại học
│   ├── 04_exposure_motion_4imgs/   # 4 ảnh lệch sáng ±30% + ca-nô chuyển động
│   └── 05_negative_non_overlapping/# 3 ảnh rời rạc kiểm thử bắt lỗi an toàn
├── scripts/
│   ├── generate_datasets.py        # Tạo lại 5 bộ dữ liệu từ cảnh gốc qua phép quay camera 3D
│   ├── run_benchmarks.py           # Chạy thực nghiệm toàn diện & sinh biểu đồ 300 DPI
│   ├── build_report.py             # Sinh Báo cáo Word (.docx) & Markdown (.md)
│   └── build_slides.py             # Sinh bộ 15 Slide thuyết trình PowerPoint (.pptx)
├── tests/
│   └── test_panorama.py            # 9 kiểm thử đơn vị (pytest)
├── results/                        # Ảnh toàn cảnh đầu ra, ảnh trung gian, biểu đồ & JSON benchmark
└── docs/
    ├── Bao_cao_Ghep_anh_toan_canh.docx
    ├── Bao_cao_Ghep_anh_toan_canh.md
    └── Slide_Thuyet_trinh_Panorama_Stitching.pptx
```

---

## 3. Hướng dẫn cài đặt & khởi chạy

### Cách 1: Chạy nhanh trên Windows (đã cài sẵn `.venv`)
Nhấp đúp tệp `run_app.bat` hoặc chạy lệnh:
```powershell
.\.venv\Scripts\python.exe app.py
```
Trình duyệt sẽ tự mở tại **`http://127.0.0.1:8000`**.

### Cách 2: Cài đặt mới từ đầu trên máy khác
```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

### Cách 3: Sử dụng qua dòng lệnh (CLI)
```powershell
.\.venv\Scripts\python.exe cli.py datasets\01_outdoor_5imgs_fhd -o output.jpg --visuals-dir visuals_out
```

### Chạy kiểm thử tự động & tái lập toàn bộ thực nghiệm / tài liệu
```powershell
# 1. Chạy 9 kiểm thử đơn vị
.\.venv\Scripts\python.exe -m pytest tests\test_panorama.py -v

# 2. Chạy lại toàn bộ thực nghiệm và sinh biểu đồ
.\.venv\Scripts\python.exe scripts\run_benchmarks.py

# 3. Sinh lại Báo cáo Word & Slide PowerPoint từ kết quả mới nhất
.\.venv\Scripts\python.exe scripts\build_report.py
.\.venv\Scripts\python.exe scripts\build_slides.py
```
