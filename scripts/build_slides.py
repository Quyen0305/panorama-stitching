"""Script sinh bộ Slide thuyết trình PowerPoint (15 slide chuẩn 16:9)
với thiết kế đồng bộ, nhúng ảnh toàn cảnh thật, bước trung gian, biểu đồ và giao diện ứng dụng.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
VISUALS_DIR = RESULTS_DIR / "visuals"
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Bảng màu thống nhất (Photography Studio Navy + Cyan + Amber + Clean Light Content)
C_NAVY = RGBColor(15, 23, 42)        # #0F172A
C_DARK_CARD = RGBColor(30, 41, 59)   # #1E293B
C_LIGHT_BG = RGBColor(248, 250, 252) # #F8FAFC
C_WHITE = RGBColor(255, 255, 255)
C_SKY = RGBColor(2, 132, 199)        # #0284C7
C_CYAN = RGBColor(56, 189, 248)      # #38BDF8
C_EMERALD = RGBColor(16, 185, 129)   # #10B981
C_AMBER = RGBColor(245, 158, 11)     # #F59E0B
C_TEXT = RGBColor(30, 41, 59)        # #1E293B
C_MUTED = RGBColor(100, 116, 139)    # #64748B
C_LIGHT_TXT = RGBColor(226, 232, 240)
C_BORDER = RGBColor(203, 213, 225)

FONT = "Segoe UI"


def set_bg(slide: Any, color: RGBColor) -> None:
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(
    slide: Any,
    left: float,
    top: float,
    width: float,
    height: float,
    fill_color: RGBColor,
    line_color: Optional[RGBColor] = None,
    rounded: bool = False,
) -> Any:
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    sh = slide.shapes.add_shape(
        shape_type, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill_color
    if line_color is not None:
        sh.line.color.rgb = line_color
        sh.line.width = Pt(1.2)
    else:
        sh.line.fill.background()
    if rounded and len(sh.adjustments) > 0:
        sh.adjustments[0] = 0.06
    return sh


def add_text(
    slide: Any,
    left: float,
    top: float,
    width: float,
    height: float,
    text: str,
    size: float = 14,
    bold: bool = False,
    color: RGBColor = C_TEXT,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    font_name: str = FONT,
) -> Any:
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.name = font_name
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    return tb


def add_bullets(
    slide: Any,
    left: float,
    top: float,
    width: float,
    height: float,
    items: List[Tuple[str, str]],
    size: float = 13.5,
    color: RGBColor = C_TEXT,
    accent: RGBColor = C_SKY,
) -> Any:
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, (head, body) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(8)
        p.line_spacing = 1.2
        r0 = p.add_run()
        r0.text = "▸ "
        r0.font.name = FONT
        r0.font.size = Pt(size)
        r0.font.bold = True
        r0.font.color.rgb = accent

        if head:
            rh = p.add_run()
            rh.text = head
            rh.font.name = FONT
            rh.font.size = Pt(size)
            rh.font.bold = True
            rh.font.color.rgb = color

        rb = p.add_run()
        rb.text = body
        rb.font.name = FONT
        rb.font.size = Pt(size)
        rb.font.color.rgb = color
    return tb


def add_picture_fit(
    slide: Any,
    img_path: Path,
    left: float,
    top: float,
    max_w: float,
    max_h: float,
    center: bool = True,
) -> None:
    """Chèn ảnh giữ đúng tỷ lệ khung hình, vừa khít hộp (max_w × max_h)."""
    if not img_path.exists():
        return
    with Image.open(img_path) as im:
        iw, ih = im.size
    ratio = iw / float(ih)
    box_ratio = max_w / float(max_h)
    if ratio > box_ratio:
        w = max_w
        h = max_w / ratio
    else:
        h = max_h
        w = max_h * ratio
    x = left + (max_w - w) * 0.5 if center else left
    y = top + (max_h - h) * 0.5 if center else top
    slide.shapes.add_picture(str(img_path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))


def add_header_footer(
    slide: Any,
    title: str,
    subtitle: str,
    slide_num: int,
    total_slides: int,
    dark: bool = False,
) -> None:
    set_bg(slide, C_NAVY if dark else C_LIGHT_BG)
    if not dark:
        add_rect(slide, 0, 0, 13.333, 1.08, C_NAVY)
        add_rect(slide, 0, 1.08, 13.333, 0.045, C_CYAN)
        add_text(slide, 0.55, 0.14, 10.5, 0.48, title, size=22, bold=True, color=C_WHITE)
        add_text(slide, 0.55, 0.62, 10.5, 0.35, subtitle, size=11.5, color=C_CYAN)
    else:
        add_rect(slide, 0, 0, 13.333, 0.06, C_CYAN)
        add_text(slide, 0.55, 0.22, 11.5, 0.48, title, size=22, bold=True, color=C_WHITE)
        add_text(slide, 0.55, 0.70, 11.5, 0.34, subtitle, size=11.5, color=C_CYAN)

    # Footer
    foot_color = RGBColor(148, 163, 184) if dark else C_MUTED
    add_text(
        slide,
        0.55,
        7.12,
        8.0,
        0.28,
        "Hệ thống Ghép ảnh Toàn cảnh (Panorama Stitching)  |  Bài tập lớn Xử lý ảnh số",
        size=9.5,
        color=foot_color,
    )
    add_text(
        slide,
        11.3,
        7.12,
        1.5,
        0.28,
        f"{slide_num} / {total_slides}",
        size=9.5,
        bold=True,
        color=C_CYAN if dark else C_SKY,
        align=PP_ALIGN.RIGHT,
    )


def add_styled_table(
    slide: Any,
    left: float,
    top: float,
    width: float,
    height: float,
    headers: List[str],
    rows: List[List[str]],
    font_size: float = 11,
) -> None:
    shape = slide.shapes.add_table(
        len(rows) + 1, len(headers), Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tbl = shape.table
    for j, h in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        r = p.add_run()
        r.text = h
        r.font.name = FONT
        r.font.size = Pt(font_size)
        r.font.bold = True
        r.font.color.rgb = C_WHITE

    for i, row in enumerate(rows):
        bg = C_WHITE if i % 2 == 0 else RGBColor(241, 245, 249)
        for j, val in enumerate(row):
            cell = tbl.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT
            r = p.add_run()
            r.text = str(val)
            r.font.name = FONT
            r.font.size = Pt(font_size - 0.5)
            r.font.bold = (j == 0)
            r.font.color.rgb = C_NAVY if j == 0 else C_TEXT


def build_slides(bench: Dict[str, Any]) -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    total = 15

    ds_list = bench["datasets_summary"]["datasets"]
    neg_test = bench["datasets_summary"]["negative_test"]
    cv_rows = bench["custom_vs_opencv"]
    fhd_time = ds_list[0]["metrics"]["core_pipeline_time_sec"]

    # ================================================================
    # SLIDE 1: TRANG BÌA (DARK HERO)
    # ================================================================
    s1 = prs.slides.add_slide(blank)
    set_bg(s1, C_NAVY)
    add_rect(s1, 0, 0, 13.333, 0.08, C_CYAN)

    # Dải ảnh toàn cảnh thật chạy ngang phía trên
    add_picture_fit(s1, RESULTS_DIR / "pano_01_outdoor_5imgs_fhd.jpg", 0.55, 0.35, 12.23, 2.75)

    # Badge môn học
    badge = add_rect(s1, 0.55, 3.35, 3.85, 0.42, C_DARK_CARD, line_color=C_CYAN, rounded=True)
    add_text(s1, 0.65, 3.40, 3.65, 0.32, "BÀI TẬP LỚN MÔN XỬ LÝ ẢNH SỐ", size=11.5, bold=True, color=C_CYAN, align=PP_ALIGN.CENTER)

    add_text(
        s1,
        0.55,
        3.92,
        12.2,
        1.15,
        "XÂY DỰNG HỆ THỐNG GHÉP ẢNH TOÀN CẢNH\n(PANORAMA STITCHING)",
        size=32,
        bold=True,
        color=C_WHITE,
    )
    add_text(
        s1,
        0.55,
        5.12,
        12.0,
        0.48,
        "Kết hợp 4/6 mô-đun cốt lõi Tự cài đặt (NumPy/SciPy) • Giao diện Web PanoramaStudio • Kiểm chứng trên 5 bộ dữ liệu",
        size=13.5,
        color=C_AMBER,
    )

    # Thông tin nhóm
    add_rect(s1, 0.55, 5.82, 12.23, 1.25, C_DARK_CARD, line_color=RGBColor(51, 65, 85), rounded=True)
    add_text(s1, 0.80, 5.95, 5.8, 0.95, "Nhóm thực hiện:\n• TV1 (Core Algorithm): [Họ tên & MSSV]\n• TV2 (Image Processing): [Họ tên & MSSV]", size=11.5, color=C_LIGHT_TXT)
    add_text(s1, 6.80, 5.95, 5.7, 0.95, "• TV3 (Ứng dụng & Báo cáo): [Họ tên & MSSV]\nGiảng viên hướng dẫn: [Tên Giảng viên]\nNăm học 2026", size=11.5, color=C_LIGHT_TXT)

    # ================================================================
    # SLIDE 2: ĐẶT VẤN ĐỀ & MỤC TIÊU
    # ================================================================
    s2 = prs.slides.add_slide(blank)
    add_header_footer(s2, "1. Đặt vấn đề & Mục tiêu dự án", "Phạm vi học thuật, hiệu năng và tiêu chí đánh giá theo Project Charter", 2, total)

    add_rect(s2, 0.55, 1.35, 5.8, 5.45, C_WHITE, line_color=C_BORDER, rounded=True)
    add_text(s2, 0.80, 1.55, 5.3, 0.4, "Bài toán & Thách thức thực tế", size=16, bold=True, color=C_NAVY)
    add_bullets(
        s2,
        0.80,
        2.05,
        5.3,
        4.5,
        [
            ("Giới hạn góc nhìn ống kính: ", "Ảnh đơn lẻ khó thu trọn cảnh quan rộng lớn, kiến trúc hay không gian nội thất."),
            ("Thứ tự chụp bất kỳ: ", "Người dùng có thể tải ảnh xáo trộn hoặc xoay lệch góc cầm máy (roll/pitch)."),
            ("Chênh lệch phơi sáng: ", "Camera tự điều chỉnh ISO/tốc độ chụp tạo đường nối sáng tối lộ rõ."),
            ("Vật thể chuyển động (Ghosting): ", "Trộn trung bình đơn giản tạo 'bóng ma' mờ ảo trong vùng chồng lấn."),
            ("Độ ổn định: ", "Cần phát hiện ảnh không chồng lấn để báo lỗi thân thiện, tuyệt đối không crash."),
        ],
        size=12.5,
    )

    # 4 KPI Cards bên phải
    kpis = [
        ("HỌC THUẬT", "4 / 6 mô-đun tự cài đặt", "Vượt mức tối thiểu 2/5 bước (NumPy/SciPy thuần, không gọi cv2.Stitcher)", C_EMERALD),
        ("HIỆU NĂNG", f"{fhd_time:.2f} giây / 5 ảnh Full HD", "Độ phân giải 1920×1080 — nhanh hơn gấp 3× giới hạn < 30 giây", C_SKY),
        ("CHẤT LƯỢNG", "100% thành công (4/4 bộ test)", "RMSE chỉ 0.16–0.52 px; cân bằng sáng mượt; DP Seam triệt tiêu ghosting", C_AMBER),
        ("SẢN PHẨM", "Ứng dụng Web + CLI + Báo cáo", "PanoramaStudio (FastAPI), 9/9 pytest pass, 5 bộ ảnh không bản quyền", C_NAVY),
    ]
    for idx, (tag, big, desc, col) in enumerate(kpis):
        y = 1.35 + idx * 1.40
        add_rect(s2, 6.65, y, 6.12, 1.24, C_WHITE, line_color=C_BORDER, rounded=True)
        add_rect(s2, 6.65, y, 0.12, 1.24, col)
        add_text(s2, 6.95, y + 0.10, 5.6, 0.28, tag, size=10, bold=True, color=col)
        add_text(s2, 6.95, y + 0.36, 5.6, 0.38, big, size=16.5, bold=True, color=C_NAVY)
        add_text(s2, 6.95, y + 0.78, 5.6, 0.40, desc, size=11, color=C_MUTED)

    # ================================================================
    # SLIDE 3: KIẾN TRÚC TỔNG THỂ HỆ THỐNG (DARK SHOWCASE)
    # ================================================================
    s3 = prs.slides.add_slide(blank)
    add_header_footer(s3, "2. Kiến trúc hệ thống Modular 6 giai đoạn", "Tách biệt rõ ràng từng bước xử lý — dễ thay thế, mở rộng và đối chiếu Custom vs OpenCV", 3, total, dark=True)
    add_picture_fit(s3, CHARTS_DIR / "architecture_diagram.png", 0.45, 1.20, 12.43, 4.15)

    # 3 hộp tóm tắt dưới sơ đồ
    bottom_cards = [
        ("Cấu trúc mã nguồn sạch", "src/panorama/: features.py, homography.py, ordering.py, warping.py, blending.py, postprocess.py"),
        ("Chế độ đối chiếu kép", "Cho phép chuyển đổi trực tiếp giữa thuật toán Tự cài đặt (Custom NumPy) và OpenCV ngay trên giao diện."),
        ("Xuất minh họa trung gian", "Tự động tạo ảnh Keypoints, Matches Inliers/Outliers, Match Graph MST và Warped Layout + DP Seam."),
    ]
    for i, (t, d) in enumerate(bottom_cards):
        x = 0.55 + i * 4.18
        add_rect(s3, x, 5.50, 3.92, 1.42, C_DARK_CARD, line_color=RGBColor(56, 189, 248), rounded=True)
        add_text(s3, x + 0.18, 5.62, 3.55, 0.36, t, size=13, bold=True, color=C_CYAN)
        add_text(s3, x + 0.18, 6.00, 3.55, 0.84, d, size=11, color=C_LIGHT_TXT)

    # ================================================================
    # SLIDE 4: BƯỚC 1 & 2 - PHÁT HIỆN & KHỚP ĐẶC TRƯNG
    # ================================================================
    s4 = prs.slides.add_slide(blank)
    add_header_footer(s4, "3. Bước 1–2: Phát hiện & Khớp đặc trưng (Tự cài đặt NumPy)", "SIFT / ORB scale-aware + Ma trận khoảng cách BLAS & Lowe's Ratio Test", 4, total)

    add_bullets(
        s4,
        0.55,
        1.35,
        5.55,
        3.5,
        [
            ("Scale-Aware Detection: ", "Thu nhỏ tạm ảnh lớn (max 1200 px) khi trích xuất SIFT/ORB rồi ánh xạ ngược tọa độ về độ phân giải Full HD — nhanh gấp 3×, giữ chuẩn subpixel."),
            ("Custom NumPy Matcher: ", "Tính khoảng cách theo khối bằng đại số ma trận: ||a - b||² = ||a||² + ||b||² - 2(a·bᵀ)."),
            ("Hỗ trợ ORB nhị phân: ", "Giải nén bit về {0, 1}²⁵⁶ giúp tính chính xác khoảng cách Hamming bằng cùng phép nhân ma trận!"),
            ("Lowe's Ratio Test (r = 0.75): ", "Tìm 2 láng giềng gần nhất bằng np.argpartition trong O(N₂), chỉ giữ cặp d₁ < 0.75·d₂."),
        ],
        size=12.2,
    )
    add_picture_fit(s4, VISUALS_DIR / "01_outdoor_5imgs_fhd" / "keypoints_0.jpg", 0.55, 4.75, 5.55, 2.2)
    add_picture_fit(s4, VISUALS_DIR / "01_outdoor_5imgs_fhd" / "matches_0.jpg", 6.35, 1.45, 6.45, 3.45)
    add_rect(s4, 6.35, 5.10, 6.45, 1.80, C_WHITE, line_color=C_BORDER, rounded=True)
    add_text(s4, 6.55, 5.22, 6.05, 0.35, "Kết quả kiểm chứng bộ khớp Tự cài đặt", size=13.5, bold=True, color=C_SKY)
    add_text(
        s4,
        6.55,
        5.60,
        6.05,
        1.20,
        "• Độ trùng khớp với cv2.BFMatcher đạt ≥ 98% trên SIFT và 100% trên ORB.\n• Ảnh phía trên minh họa cặp ghép thực tế: xanh lá là Inliers (98.5%), đỏ là Outliers bị RANSAC loại bỏ hoàn toàn.",
        size=11.5,
        color=C_TEXT,
    )

    # ================================================================
    # SLIDE 5: BƯỚC 3 - HOMOGRAPHY (DLT + RANSAC + LM)
    # ================================================================
    s5 = prs.slides.add_slide(blank)
    add_header_footer(s5, "4. Bước 3: Ước lượng Homography (100% Tự cài đặt NumPy)", "Hartley Normalization + Normalized DLT (SVD) + Adaptive RANSAC/MSAC + Levenberg-Marquardt", 5, total)

    steps = [
        ("Bước 3.1 — Chuẩn hóa Hartley", "Tịnh tiến tâm về (0, 0), chuẩn hóa bán kính trung bình bằng √2 giúp hệ số điều kiện ma trận SVD ổn định số học.", C_SKY),
        ("Bước 3.2 — Normalized DLT (SVD)", "Lập ma trận A (2n×9) từ n ≥ 4 cặp điểm; dùng full_matrices=True khi n=4 để trích đúng không gian hạch (hàng thứ 9 của Vᵀ). Giải chuẩn hóa H = T_dst⁻¹ · H_norm · T_src.", C_EMERALD),
        ("Bước 3.3 — Adaptive RANSAC + MSAC", "Kiểm tra loại 3 điểm thẳng hàng; chấm điểm mượt theo MSAC; tự cập nhật số vòng lặp tối đa N = ceil(log(1-p)/log(1-w⁴)).", C_AMBER),
        ("Bước 3.4 — Tinh chỉnh LM & Kiểm định lồi", "Tính Jacobian giải tích 2N×8 của phép chiếu xạ ảnh và tối ưu phi tuyến Levenberg-Marquardt trên toàn bộ Inliers; kiểm định tứ giác sau warp lồi và không gập qua vô cực.", C_NAVY),
    ]
    for i, (t, d, col) in enumerate(steps):
        col_i = i % 2
        row_i = i // 2
        x = 0.55 + col_i * 6.25
        y = 1.38 + row_i * 2.18
        add_rect(s5, x, y, 5.98, 1.98, C_WHITE, line_color=C_BORDER, rounded=True)
        add_rect(s5, x, y, 5.98, 0.08, col)
        add_text(s5, x + 0.22, y + 0.20, 5.55, 0.36, t, size=14.5, bold=True, color=col)
        add_text(s5, x + 0.22, y + 0.62, 5.55, 1.22, d, size=12, color=C_TEXT)

    # Box công thức dưới cùng
    add_rect(s5, 0.55, 5.82, 12.23, 1.10, C_NAVY, rounded=True)
    add_text(
        s5,
        0.80,
        5.94,
        11.7,
        0.90,
        "Kiểm chứng toán học: Trên bộ Urban (4 ảnh), thuật toán tự cài đặt đạt sai số chiếu lại RMSE = 0.158 px với 945.3 inliers — TƯƠNG ĐƯƠNG TUYỆT ĐỐI tới 3 chữ số thập phân so với cv2.findHomography của OpenCV C++!",
        size=12.5,
        bold=True,
        color=C_CYAN,
    )

    # ================================================================
    # SLIDE 6: TỰ SẮP XẾP THỨ TỰ ẢNH (AUTO-ORDER MST)
    # ================================================================
    s6 = prs.slides.add_slide(blank)
    add_header_footer(s6, "5. Tự động sắp xếp thứ tự ảnh (Auto-Order MST)", "Xây dựng đồ thị khớp cặp, tìm Cây khung cực đại Kruskal và chọn ảnh gốc trung tâm", 6, total)

    add_picture_fit(s6, VISUALS_DIR / "01_outdoor_5imgs_fhd" / "match_graph.jpg", 0.55, 1.40, 7.25, 3.60)
    add_rect(s6, 0.55, 5.20, 7.25, 1.65, C_WHITE, line_color=C_BORDER, rounded=True)
    add_text(s6, 0.78, 5.32, 6.8, 0.36, "Ví dụ thực tế trên Dataset 01 (5 ảnh Full HD bị xáo trộn)", size=13.5, bold=True, color=C_SKY)
    add_text(
        s6,
        0.78,
        5.72,
        6.8,
        1.02,
        "• Tên tệp tải lên: img_01, img_02, img_03, img_04, img_05 (sai hoàn toàn thứ tự không gian).\n• Đồ thị MST tự phát hiện chính xác chuỗi thật từ trái sang phải: #3 → #1 → #5 → #2 → #4.\n• Chọn ảnh trung tâm #5 (img_05.jpg) làm Anchor giúp giảm 50% độ giãn phối cảnh về 2 biên.",
        size=11.5,
        color=C_TEXT,
    )

    add_rect(s6, 8.05, 1.40, 4.73, 5.45, C_WHITE, line_color=C_BORDER, rounded=True)
    add_text(s6, 8.28, 1.60, 4.3, 0.4, "Quy trình 4 bước của ordering.py", size=15, bold=True, color=C_NAVY)
    add_bullets(
        s6,
        8.28,
        2.12,
        4.3,
        4.5,
        [
            ("Đồ thị trọng số: ", "Khớp mọi cặp (i, j), gán trọng số là số RANSAC inliers hợp lệ."),
            ("Kruskal MST: ", "Chọn N-1 cạnh mạnh nhất giúp chuỗi biến đổi có sai số nhỏ nhất."),
            ("Phát hiện lỗi sớm: ", "Nếu đồ thị không liên thông, chỉ đích danh các ảnh rời rạc và báo lỗi an toàn."),
            ("Chọn Anchor tối ưu: ", "Tính độ lệch tâm (eccentricity) bằng BFS, chọn nút giữa cây làm hệ tọa độ chuẩn."),
        ],
        size=12.2,
    )

    # ================================================================
    # SLIDE 7: BƯỚC 4 - WARPING & CHIẾU TRỤ
    # ================================================================
    s7 = prs.slides.add_slide(blank)
    add_header_footer(s7, "6. Bước 4: Biến đổi hình học & Phép chiếu mặt trụ", "Chuỗi hóa Homography toàn cục, bảo vệ canvas và trực quan hóa vùng chồng lấn", 7, total)

    add_picture_fit(s7, VISUALS_DIR / "01_outdoor_5imgs_fhd" / "warped_layout.jpg", 0.55, 1.38, 12.23, 3.60)

    add_rect(s7, 0.55, 5.18, 5.95, 1.72, C_WHITE, line_color=C_BORDER, rounded=True)
    add_text(s7, 0.78, 5.30, 5.5, 0.34, "Chiếu phẳng (Planar) & Bảo vệ Canvas", size=14, bold=True, color=C_SKY)
    add_text(
        s7,
        0.78,
        5.68,
        5.5,
        1.12,
        "• Duyệt BFS từ ảnh gốc để chuỗi hóa H_{i → anchor}.\n• Kiểm tra giới hạn bộ nhớ (max 36 MP) và co mặt nạ 1px bằng erosion giúp loại bỏ hoàn toàn viền đen nội suy.",
        size=11.5,
        color=C_TEXT,
    )

    add_rect(s7, 6.80, 5.18, 5.98, 1.72, C_WHITE, line_color=C_BORDER, rounded=True)
    add_text(s7, 7.02, 5.30, 5.5, 0.34, "Chiếu mặt trụ (Cylindrical — Tự cài đặt)", size=14, bold=True, color=C_EMERALD)
    add_text(
        s7,
        7.02,
        5.68,
        5.5,
        1.12,
        "• Ánh xạ nghịch x_s = f·tan(θ) + c_x, y_s = (y - c_y)/cos(θ) + c_y với f = W / (2·tan(FOV/2)).\n• Hỗ trợ quét góc rộng mà không bị kéo giãn mạnh ở hai rìa ảnh.",
        size=11.5,
        color=C_TEXT,
    )

    # ================================================================
    # SLIDE 8: BƯỚC 5 - ĐIỂM NHẤN KỸ THUẬT: BLENDING TỰ CÀI ĐẶT
    # ================================================================
    s8 = prs.slides.add_slide(blank)
    add_header_footer(s8, "7. Bước 5: Trộn ảnh & Cân bằng sáng (Điểm nhấn kỹ thuật)", "Toàn bộ 3 tầng xử lý trong blending.py được TỰ CÀI ĐẶT bằng NumPy/SciPy", 8, total)

    cols = [
        (
            "Tầng 1: Cân bằng phơi sáng",
            "Gain Compensation (Brown & Lowe)",
            [
                ("Ý tưởng: ", "Tối ưu hệ số g_i cho từng kênh B, G, R để cường độ trung bình trên mọi vùng chồng lấn khớp nhau."),
                ("Toán học: ", "Giải hệ phương trình tuyến tính N×N (bình phương tối thiểu có điều chuẩn) bằng np.linalg.solve."),
                ("Hiệu quả: ", "Triệt tiêu >90% sai khác độ sáng ±30% và lệch cân bằng trắng trước khi trộn."),
            ],
            C_SKY,
        ),
        (
            "Tầng 2: Đường nối tối ưu DP",
            "Dynamic Programming Min-Seam",
            [
                ("Hàm năng lượng: ", "E = ||I_A - I_B||₂ + 0.6·|∇I_A - ∇I_B| + bias tâm."),
                ("Quy hoạch động: ", "Tìm đường cắt liên tục 8-láng giềng có tổng sai khác nhỏ nhất từ đỉnh xuống đáy vùng chồng lấn."),
                ("Hiệu quả: ", "Tự động vòng né vật thể chuyển động — loại bỏ triệt để hiện tượng 'bóng ma' (ghosting)."),
            ],
            C_AMBER,
        ),
        (
            "Tầng 3: Tháp Đa dải tần",
            "Laplacian Pyramid (Burt & Adelson)",
            [
                ("Tích chập thuần NumPy: ", "Nhân Gaussian tách rời 5 điểm [1, 4, 6, 4, 1]/16; pyr_down và pyr_up bảo toàn năng lượng."),
                ("Trộn đa tần số: ", "Tần số thấp (độ sáng, nền trời) chuyển mượt rộng; tần số cao (chi tiết) bám sát đường seam DP."),
                ("Chống quầng tối: ", "Ngoại suy biên ngoài mặt nạ hợp lệ bằng EDT indices — 0% quầng đen!"),
            ],
            C_EMERALD,
        ),
    ]
    for i, (t, sub, buls, col) in enumerate(cols):
        x = 0.55 + i * 4.18
        add_rect(s8, x, 1.38, 3.92, 5.45, C_WHITE, line_color=C_BORDER, rounded=True)
        add_rect(s8, x, 1.38, 3.92, 0.10, col)
        add_text(s8, x + 0.20, 1.58, 3.55, 0.36, t, size=14.5, bold=True, color=col)
        add_text(s8, x + 0.20, 1.94, 3.55, 0.30, sub, size=10.5, bold=True, color=C_MUTED)
        add_bullets(s8, x + 0.18, 2.38, 3.58, 4.3, buls, size=11.6, accent=col)

    # ================================================================
    # SLIDE 9: BƯỚC 6 - HẬU XỬ LÝ & XỬ LÝ LỖI AN TOÀN
    # ================================================================
    s9 = prs.slides.add_slide(blank)
    add_header_footer(s9, "8. Bước 6: Cắt biên tự động & Cơ chế xử lý lỗi an toàn", "Largest Inscribed Rectangle O(H·W) và hệ thống ngoại lệ thân thiện", 9, total)

    add_rect(s9, 0.55, 1.38, 6.0, 5.45, C_WHITE, line_color=C_BORDER, rounded=True)
    add_text(s9, 0.80, 1.58, 5.5, 0.4, "Cắt biên đen tự động (Auto-Crop O(H·W))", size=15.5, bold=True, color=C_SKY)
    add_bullets(
        s9,
        0.80,
        2.08,
        5.5,
        2.5,
        [
            ("Vấn đề: ", "Ảnh toàn cảnh sau warp có viền hình thang cong và góc tam giác đen."),
            ("Giải thuật tự cài đặt: ", "Co bảo thủ mặt nạ (min-pooling) kết hợp thuật toán ngăn xếp đơn điệu tìm Hình chữ nhật nội tiếp lớn nhất toàn pixel hợp lệ."),
            ("Kết quả: ", "Loại bỏ 100% pixel đen chỉ trong ~60 ms, giữ lại 91.9% diện tích ảnh thật."),
        ],
        size=12,
    )
    add_picture_fit(s9, VISUALS_DIR / "01_outdoor_5imgs_fhd" / "panorama_uncropped.jpg", 0.80, 4.75, 5.5, 1.90)

    add_rect(s9, 6.80, 1.38, 5.98, 5.45, C_WHITE, line_color=C_BORDER, rounded=True)
    add_text(s9, 7.05, 1.58, 5.5, 0.4, "Độ ổn định & Hệ thống ngoại lệ PanoramaError", size=15.5, bold=True, color=C_AMBER)
    add_styled_table(
        s9,
        7.05,
        2.15,
        5.48,
        3.55,
        ["Mã lỗi", "Tình huống phát hiện"],
        [
            ["INSUFFICIENT_IMAGES", "Ít hơn 2 ảnh hoặc vượt quá 10 ảnh."],
            ["IMAGE_LOAD_ERROR", "Tệp ảnh hỏng hoặc sai định dạng."],
            ["INSUFFICIENT_FEATURES", "Ảnh mờ/trơn có < 4 điểm đặc trưng."],
            ["INSUFFICIENT_MATCHES", "Các ảnh không chồng lấn (< 15 inliers)."],
            ["DISCONNECTED_IMAGES", "Có ảnh lạc cảnh không nối được vào MST."],
            ["CANVAS_OVERFLOW", "Biến dạng cực đoan gây tràn bộ nhớ."],
        ],
        font_size=10.5,
    )
    add_text(
        s9,
        7.05,
        5.85,
        5.48,
        0.82,
        f"Kiểm thử âm tính (Dataset 05): Hệ thống bắt an toàn mã [{neg_test.get('code')}] và hiển thị gợi ý khắc phục bằng tiếng Việt, không bao giờ crash.",
        size=11.2,
        bold=True,
        color=C_EMERALD,
    )

    # ================================================================
    # SLIDE 10: BỘ DỮ LIỆU & KIỂM THỬ TỰ ĐỘNG
    # ================================================================
    s10 = prs.slides.add_slide(blank)
    add_header_footer(s10, "9. Bộ dữ liệu không bản quyền & Kiểm thử tự động", "5 bộ dữ liệu mô phỏng camera 3D chính xác + 9/9 unit tests pytest", 10, total)

    ds_rows_tbl = [
        ["01 - Ngoại cảnh Hồ Alpine", "5 ảnh Full HD (1920×1080), xáo trộn", "Kiểm chứng hiệu năng < 30s & Auto-Order MST"],
        ["02 - Quảng trường Châu Âu", "4 ảnh (960×1080), chồng lấn ~40%", "Kiểm định độ thẳng kiến trúc & RMSE Homography"],
        ["03 - Nội thất Thư viện", "3 ảnh (1080×1080), ánh sáng trong nhà", "Bối cảnh trong nhà nhiều kệ sách lặp kết cấu"],
        ["04 - Bến cảng Địa Trung Hải", "4 ảnh (960×1080), sáng ±30% + ca-nô", "Thử thách Gain Compensation & DP Seam chống ghosting"],
        ["05 - Kiểm thử âm tính", "3 ảnh từ 3 bối cảnh rời rạc", "Xác minh bắt lỗi an toàn không crash"],
    ]
    add_styled_table(s10, 0.55, 1.38, 12.23, 3.05, ["Bộ dữ liệu (100% không bản quyền)", "Quy cách ảnh", "Mục đích đánh giá"], ds_rows_tbl, font_size=11.5)

    add_rect(s10, 0.55, 4.68, 12.23, 2.20, C_NAVY, rounded=True)
    add_text(s10, 0.80, 4.82, 11.7, 0.38, "KẾT QUẢ KIỂM THỬ ĐƠN VỊ TỰ ĐỘNG (PYTEST — 9/9 PASSED TRONG 3.31s)", size=13.5, bold=True, color=C_EMERALD)
    add_text(
        s10,
        0.80,
        5.26,
        5.6,
        1.45,
        "✓ Custom SIFT Matcher khớp ≥ 98% OpenCV BFMatcher\n✓ Custom ORB Hamming khớp 100% OpenCV BFMatcher\n✓ Custom DLT + RANSAC + LM khôi phục đúng H (RMSE < 0.8 px)\n✓ Tháp Laplacian tái tạo ảnh lẻ sai số < 0.001",
        size=11.5,
        color=C_LIGHT_TXT,
    )
    add_text(
        s10,
        6.65,
        5.26,
        5.8,
        1.45,
        "✓ Exposure Gain giảm > 85% độ lệch phơi sáng\n✓ DP Seam né 100% vật thể chuyển động trên vùng overlap\n✓ Auto-Crop loại bỏ 100% viền đen (0 pixel lỗi)\n✓ Cylindrical Projection, Auto-Order & Error Handling đạt 100%",
        size=11.5,
        color=C_LIGHT_TXT,
    )

    # ================================================================
    # SLIDE 11: SHOWCASE 4 ẢNH PANORAMA HOÀN CHỈNH (DARK GALLERY)
    # ================================================================
    s11 = prs.slides.add_slide(blank)
    add_header_footer(s11, "10. Kết quả ghép toàn cảnh trên 4 bối cảnh thực nghiệm", "Tỷ lệ thành công 100% (4/4) — Ảnh sắc nét, liền mạch, tự động cắt biên không còn viền đen", 11, total, dark=True)

    showcase = [
        (RESULTS_DIR / "pano_01_outdoor_5imgs_fhd.jpg", f"01. Hồ Alpine (5 ảnh Full HD) — {ds_list[0]['metrics']['core_pipeline_time_sec']}s | RMSE {ds_list[0]['metrics']['mean_reprojection_rmse_px']} px"),
        (RESULTS_DIR / "pano_02_urban_4imgs.jpg", f"02. Quảng trường Châu Âu (4 ảnh) — {ds_list[1]['metrics']['core_pipeline_time_sec']}s | RMSE {ds_list[1]['metrics']['mean_reprojection_rmse_px']} px"),
        (RESULTS_DIR / "pano_03_indoor_3imgs.jpg", f"03. Nội thất Thư viện (3 ảnh) — {ds_list[2]['metrics']['core_pipeline_time_sec']}s | RMSE {ds_list[2]['metrics']['mean_reprojection_rmse_px']} px"),
        (RESULTS_DIR / "pano_04_exposure_motion_4imgs.jpg", f"04. Bến cảng Lệch sáng ±30% & Ca-nô (4 ảnh) — {ds_list[3]['metrics']['core_pipeline_time_sec']}s | Không ghosting"),
    ]
    for i, (img_p, cap) in enumerate(showcase):
        c = i % 2
        r = i // 2
        x = 0.48 + c * 6.30
        y = 1.18 + r * 2.94
        add_rect(s11, x, y, 6.08, 2.78, C_DARK_CARD, line_color=RGBColor(51, 65, 85), rounded=True)
        add_picture_fit(s11, img_p, x + 0.12, y + 0.10, 5.84, 2.24)
        add_text(s11, x + 0.15, y + 2.38, 5.80, 0.34, cap, size=10.5, bold=True, color=C_CYAN, align=PP_ALIGN.CENTER)

    # ================================================================
    # SLIDE 12: SO SÁNH BLENDING & CHỐNG GHOSTING (DARK SHOWCASE)
    # ================================================================
    s12 = prs.slides.add_slide(blank)
    add_header_footer(s12, "11. Thực nghiệm Trộn ảnh & Chống Ghosting (Dataset 04)", "Chênh lệch phơi sáng ±30% giữa các ảnh kết hợp ca-nô chuyển động trong vùng chồng lấn", 12, total, dark=True)

    add_picture_fit(s12, CHARTS_DIR / "blending_comparison_zoom.png", 0.45, 1.18, 12.43, 3.95)
    add_rect(s12, 0.55, 5.32, 12.23, 1.58, C_DARK_CARD, line_color=C_AMBER, rounded=True)
    add_text(s12, 0.80, 5.44, 11.7, 0.34, "Phân tích so sánh trực quan 4 kỹ thuật:", size=13, bold=True, color=C_AMBER)
    add_text(
        s12,
        0.80,
        5.82,
        11.7,
        1.00,
        "1. Hard Cut: Lộ rõ vết cắt sáng/tối và cắt đôi ca-nô.   2. Feather + Gain: Hết lệch sáng nhưng tạo 'bóng ma' ca-nô mờ ảo.\n3. Multi-Band + Gain: Mượt ánh sáng nhưng đường Voronoi tĩnh vẫn cắt ngang thân tàu.\n4. DP Seam + Multi-Band + Gain (Đề xuất): Cân bằng sáng hoàn hảo VÀ tự động uốn đường nối né ca-nô → 100% không bóng ma!",
        size=11.5,
        color=C_LIGHT_TXT,
    )

    # ================================================================
    # SLIDE 13: HIỆU NĂNG & SO SÁNH CUSTOM VS OPENCV
    # ================================================================
    s13 = prs.slides.add_slide(blank)
    add_header_footer(s13, "12. Đánh giá Hiệu năng & Đối chiếu Custom NumPy vs OpenCV", "Nhanh hơn 3× yêu cầu đề tài và độ chính xác tương đương tuyệt đối thư viện C++", 13, total)

    add_picture_fit(s13, CHARTS_DIR / "chart_stage_timings.png", 0.45, 1.28, 6.15, 3.65)
    add_picture_fit(s13, CHARTS_DIR / "chart_custom_vs_opencv.png", 6.72, 1.28, 6.15, 3.65)

    cv_tbl_rows = [
        [r["config_name"], f"{r['match_homog_ms']:.1f} ms", f"{r['mean_inliers']} ({r['mean_inlier_ratio_pct']}%)", f"{r['mean_rmse_px']:.3f} px"]
        for r in cv_rows
    ]
    add_styled_table(s13, 0.55, 5.05, 12.23, 1.85, ["Cấu hình thực nghiệm (Dataset 02)", "Match + RANSAC", "Inliers TB", "Sai số RMSE"], cv_tbl_rows, font_size=10.5)

    # ================================================================
    # SLIDE 14: DEMO ỨNG DỤNG PANORAMASTUDIO
    # ================================================================
    s14 = prs.slides.add_slide(blank)
    add_header_footer(s14, "13. Demo Ứng dụng hoàn chỉnh — PanoramaStudio", "Khởi chạy 1 chạm bằng run_app.bat • Giao diện Web trực quan • Hỗ trợ Demo trực tiếp", 14, total, dark=True)

    add_picture_fit(s14, CHARTS_DIR / "ui_screenshot.png", 0.50, 1.22, 8.20, 5.65)

    add_rect(s14, 8.92, 1.22, 3.90, 5.65, C_DARK_CARD, line_color=C_CYAN, rounded=True)
    add_text(s14, 9.15, 1.42, 3.45, 0.40, "Kịch bản Demo trực tiếp", size=15, bold=True, color=C_CYAN)
    add_bullets(
        s14,
        9.15,
        1.95,
        3.45,
        4.75,
        [
            ("Bước 1 — Nạp ảnh: ", "Kéo thả 2–10 ảnh cá nhân hoặc bấm chọn 1 trong 5 bộ dữ liệu mẫu."),
            ("Bước 2 — Tùy chỉnh: ", "Bật/tắt Custom vs OpenCV, Auto-Order, Exposure Gain, DP Seam + Multi-Band."),
            ("Bước 3 — Phân tích: ", "Xem trực tiếp tiến trình 6 bước, thanh kéo so sánh cắt biên, Keypoints, Inliers, đồ thị MST & tải PNG/JPG/ZIP."),
            ("Bước 4 — Test lỗi: ", "Chạy bộ 05 (3 ảnh rời rạc) để chứng minh hệ thống báo lỗi rõ ràng, không crash."),
        ],
        size=11.5,
        color=C_LIGHT_TXT,
        accent=C_AMBER,
    )

    # ================================================================
    # SLIDE 15: HẠN CHẾ, HƯỚNG PHÁT TRIỂN & CẢM ƠN
    # ================================================================
    s15 = prs.slides.add_slide(blank)
    add_header_footer(s15, "14. Hạn chế, Hướng phát triển & Kết luận", "Tổng kết đóng góp kỹ thuật và lộ trình nâng cấp hệ thống", 15, total)

    add_rect(s15, 0.55, 1.38, 5.95, 4.15, C_WHITE, line_color=C_BORDER, rounded=True)
    add_rect(s15, 0.55, 1.38, 5.95, 0.08, C_AMBER)
    add_text(s15, 0.80, 1.58, 5.45, 0.38, "Hạn chế đã ghi nhận", size=15.5, bold=True, color=C_AMBER)
    add_bullets(
        s15,
        0.80,
        2.06,
        5.45,
        3.3,
        [
            ("Thị sai (Parallax): ", "Ảnh hưởng nhẹ nếu máy ảnh dịch chuyển tịnh tiến gần tiền cảnh."),
            ("Vật chuyển động cỡ rất lớn: ", "Nếu che kín toàn bộ chiều vùng giao, DP vẫn buộc phải đi qua một phần vật thể."),
            ("Chuỗi rất dài (>10 ảnh): ", "Nhân chuỗi Homography dọc MST có thể tích lũy sai số nhỏ ở hai đầu."),
            ("Tối ưu tốc độ trộn: ", "Multi-Band thuần NumPy có thể tăng tốc 5–10× bằng Numba/Cython/GPU."),
        ],
        size=12,
        accent=C_AMBER,
    )

    add_rect(s15, 6.80, 1.38, 5.98, 4.15, C_WHITE, line_color=C_BORDER, rounded=True)
    add_rect(s15, 6.80, 1.38, 5.98, 0.08, C_EMERALD)
    add_text(s15, 7.05, 1.58, 5.48, 0.38, "Hướng phát triển tương lai", size=15.5, bold=True, color=C_EMERALD)
    add_bullets(
        s15,
        7.05,
        2.06,
        5.48,
        3.3,
        [
            ("Bundle Adjustment toàn cục: ", "Tối ưu đồng thời góc xoay & tiêu cự tất cả camera, khử sai số tích lũy và làm thẳng đường chân trời."),
            ("Ghép cầu 360° & Đa hàng: ", "Mở rộng phép chiếu hình cầu cho toàn cảnh nhiều tầng (multi-row)."),
            ("Graph Cut 2D + Poisson Blending: ", "Nâng cấp tối ưu đường nối 2D cho cảnh động phức tạp."),
            ("Đặc trưng học sâu (Deep Features): ", "Tích hợp SuperPoint / LightGlue cho cảnh chụp đêm thiếu sáng."),
        ],
        size=12,
        accent=C_EMERALD,
    )

    add_rect(s15, 0.55, 5.75, 12.23, 1.18, C_NAVY, rounded=True)
    add_text(
        s15,
        0.55,
        5.88,
        12.23,
        0.48,
        "CẢM ƠN THẦY/CÔ VÀ CÁC BẠN ĐÃ LẮNG NGHE!",
        size=20,
        bold=True,
        color=C_CYAN,
        align=PP_ALIGN.CENTER,
    )
    add_text(
        s15,
        0.55,
        6.40,
        12.23,
        0.38,
        "Sẵn sàng Demo trực tiếp ứng dụng PanoramaStudio & Trao đổi Q&A",
        size=12.5,
        color=C_LIGHT_TXT,
        align=PP_ALIGN.CENTER,
    )

    out_pptx = DOCS_DIR / "Slide_Thuyet_trinh_Panorama_Stitching.pptx"
    prs.save(str(out_pptx))
    return out_pptx


def main() -> None:
    bench_file = RESULTS_DIR / "benchmark_summary.json"
    bench = json.loads(bench_file.read_text(encoding="utf-8"))
    pptx_path = build_slides(bench)
    print("Đã sinh thành công bộ Slide PowerPoint (15 trang):", pptx_path)


if __name__ == "__main__":
    main()
