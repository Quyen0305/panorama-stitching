"""Bộ kiểm thử tự động (pytest) cho hệ thống Ghép ảnh Toàn cảnh."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from panorama import (  # noqa: E402
    DisconnectedImagesError,
    InsufficientImagesError,
    InsufficientMatchesError,
    PanoramaStitcher,
    StitchingConfig,
)
from panorama.blending import (  # noqa: E402
    build_laplacian_pyramid,
    compensate_exposure,
    find_optimal_seam_dp,
    reconstruct_laplacian_pyramid,
)
from panorama.features import (  # noqa: E402
    custom_knn_match_ratio_test,
    opencv_knn_match_ratio_test,
)
from panorama.homography import (  # noqa: E402
    compute_reprojection_errors,
    estimate_homography_ransac_custom,
)
from panorama.postprocess import auto_crop_largest_rectangle  # noqa: E402
from panorama.warping import WarpedFrame  # noqa: E402


def test_custom_matcher_sift_matches_opencv() -> None:
    rng = np.random.default_rng(123)
    d1 = rng.random((300, 128), dtype=np.float32) * 100.0
    d2 = d1.copy()
    # Thêm nhiễu nhỏ vào 150 vector đầu và trộn ngẫu nhiên phần còn lại
    d2[:150] += rng.normal(0.0, 2.5, size=(150, 128)).astype(np.float32)
    d2[150:] = rng.random((150, 128), dtype=np.float32) * 100.0
    perm = rng.permutation(300)
    d2 = d2[perm]

    m_custom = custom_knn_match_ratio_test(d1, d2, ratio_threshold=0.75, is_binary=False)
    m_cv = opencv_knn_match_ratio_test(d1, d2, ratio_threshold=0.75, is_binary=False)

    set_custom = {(m.query_idx, m.train_idx) for m in m_custom}
    set_cv = {(m.query_idx, m.train_idx) for m in m_cv}
    overlap = len(set_custom & set_cv) / max(1, len(set_cv))
    assert overlap >= 0.98, f"Độ trùng khớp Custom vs OpenCV SIFT matcher quá thấp: {overlap:.3f}"


def test_custom_matcher_orb_hamming_matches_opencv() -> None:
    rng = np.random.default_rng(456)
    d1 = rng.integers(0, 256, size=(200, 32), dtype=np.uint8)
    d2 = d1.copy()
    # Lật nhẹ 1 vài bit ở 80 vector đầu
    flip_mask = np.zeros_like(d2)
    flip_mask[:80, :2] = 0b00000101
    d2 = np.bitwise_xor(d2, flip_mask)
    d2[80:] = rng.integers(0, 256, size=(120, 32), dtype=np.uint8)

    m_custom = custom_knn_match_ratio_test(d1, d2, ratio_threshold=0.75, is_binary=True)
    m_cv = opencv_knn_match_ratio_test(d1, d2, ratio_threshold=0.75, is_binary=True)

    set_custom = {(m.query_idx, m.train_idx) for m in m_custom}
    set_cv = {(m.query_idx, m.train_idx) for m in m_cv}
    assert set_custom == set_cv


def test_custom_dlt_ransac_lm_recovers_ground_truth_homography() -> None:
    rng = np.random.default_rng(789)
    H_true = np.array(
        [
            [1.08, -0.06, 45.0],
            [0.05, 0.97, -18.0],
            [0.00018, -0.00009, 1.0],
        ],
        dtype=np.float64,
    )

    n_inliers = 160
    n_outliers = 90
    src_in = rng.uniform([20, 20], [900, 700], size=(n_inliers, 2))
    hom = np.hstack([src_in, np.ones((n_inliers, 1))])
    proj = (H_true @ hom.T).T
    dst_in = proj[:, :2] / proj[:, 2:3]
    dst_in += rng.normal(0.0, 0.5, size=dst_in.shape)

    src_out = rng.uniform([20, 20], [900, 700], size=(n_outliers, 2))
    dst_out = rng.uniform([20, 20], [900, 700], size=(n_outliers, 2))

    src_all = np.vstack([src_in, src_out])
    dst_all = np.vstack([dst_in, dst_out])

    res = estimate_homography_ransac_custom(
        src_all,
        dst_all,
        ransac_threshold=3.5,
        max_iters=1500,
        min_inliers=20,
        src_shape=(720, 960),
        seed=42,
    )
    assert res.is_valid and res.H is not None
    assert res.num_inliers >= int(0.95 * n_inliers)
    assert res.reprojection_rmse < 1.2

    # Kiểm tra trực tiếp trên tập điểm sạch không nhiễu
    clean_errs = compute_reprojection_errors(res.H, src_in, proj[:, :2] / proj[:, 2:3])
    assert float(np.mean(clean_errs)) < 0.8


def test_laplacian_pyramid_exact_reconstruction() -> None:
    rng = np.random.default_rng(101)
    # Kích thước lẻ để kiểm chứng căn chỉnh odd-dimension của pyr_down / pyr_up
    img = rng.uniform(0.0, 255.0, size=(157, 233, 3)).astype(np.float32)
    l_pyr = build_laplacian_pyramid(img, levels=4)
    rec = reconstruct_laplacian_pyramid(l_pyr)
    max_err = float(np.max(np.abs(rec - img)))
    assert max_err < 1e-3, f"Sai số tái tạo tháp Laplacian quá cao: {max_err}"


def test_exposure_compensation_equalizes_brightness() -> None:
    h, w = 200, 320
    base = np.full((h, w, 3), 140, dtype=np.uint8)
    # Hai khung chồng lấn nhau ở cột [100:220]
    m1 = np.zeros((h, w), dtype=bool)
    m1[:, :220] = True
    img1 = np.zeros_like(base)
    img1[m1] = 160  # Ảnh 1 sáng hơn

    m2 = np.zeros((h, w), dtype=bool)
    m2[:, 100:] = True
    img2 = np.zeros_like(base)
    img2[m2] = 95  # Ảnh 2 tối hơn ~40%

    frames = [
        WarpedFrame(0, img1, m1, (0, 0, 220, h), np.eye(3)),
        WarpedFrame(1, img2, m2, (100, 0, w, h), np.eye(3)),
    ]
    comp, gains = compensate_exposure(frames)
    ov = m1 & m2
    diff_before = abs(float(img1[ov].mean()) - float(img2[ov].mean()))
    diff_after = abs(float(comp[0].image[ov].mean()) - float(comp[1].image[ov].mean()))
    assert diff_after < diff_before * 0.15, f"Cân bằng sáng chưa đủ tốt: {diff_before:.1f} -> {diff_after:.1f}"


def test_dp_seam_avoids_moving_object() -> None:
    h, w = 160, 260
    bg = np.full((h, w, 3), 120, dtype=np.uint8)
    m_a = np.zeros((h, w), dtype=bool)
    m_a[:, :180] = True
    m_b = np.zeros((h, w), dtype=bool)
    m_b[:, 80:] = True

    img_a = bg.copy()
    img_a[~m_a] = 0
    img_b = bg.copy()
    img_b[~m_b] = 0
    # Chèn vật chuyển động màu đỏ rất sáng ở chính giữa vùng overlap [115:145, 55:105]
    img_b[55:105, 115:145] = (10, 10, 250)

    w_b, seam_pts = find_optimal_seam_dp(img_a, m_a, img_b, m_b)
    # Kiểm tra đường seam KHÔNG đi xuyên qua vùng vật thể chuyển động
    inside_obj = (
        (seam_pts[:, 0] >= 116)
        & (seam_pts[:, 0] <= 144)
        & (seam_pts[:, 1] >= 56)
        & (seam_pts[:, 1] <= 104)
    )
    assert int(np.sum(inside_obj)) == 0, "Đường seam DP không nên cắt xuyên qua vật chuyển động!"
    # Kiểm tra vùng non-overlap vẫn được bảo toàn tuyệt đối
    assert np.all(w_b[m_a & (~m_b)] == 0.0)
    assert np.all(w_b[m_b & (~m_a)] == 1.0)


def test_auto_crop_zero_black_pixels() -> None:
    h, w = 400, 600
    mask = np.zeros((h, w), dtype=bool)
    pts = np.array([[60, 40], [560, 20], [540, 360], [30, 380]], dtype=np.int32)
    mask_u8 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask_u8, pts, 255)
    mask = mask_u8 > 0

    img = np.full((h, w, 3), 200, dtype=np.uint8)
    img[~mask] = 0

    cropped, (x0, y0, x1, y1) = auto_crop_largest_rectangle(img, mask)
    assert np.all(mask[y0:y1, x0:x1]), "Ảnh sau auto-crop không được chứa pixel đen!"
    assert (x1 - x0) * (y1 - y0) > 0.72 * int(np.count_nonzero(mask))


def test_error_handling_invalid_inputs() -> None:
    stitcher = PanoramaStitcher(StitchingConfig(generate_visuals=False))

    with pytest.raises(InsufficientImagesError):
        stitcher.stitch([np.zeros((100, 100, 3), dtype=np.uint8)])

    neg_dir = PROJECT_ROOT / "datasets" / "05_negative_non_overlapping"
    if neg_dir.exists():
        paths = sorted(neg_dir.glob("*.jpg"))
        with pytest.raises((InsufficientMatchesError, DisconnectedImagesError)):
            stitcher.stitch_paths(paths)


def test_auto_order_and_cylindrical_on_real_datasets() -> None:
    d3 = PROJECT_ROOT / "datasets" / "03_indoor_3imgs"
    if not d3.exists():
        pytest.skip("Dataset 03 chưa được tạo")
    paths = sorted(d3.glob("*.jpg"))
    # Đảo ngược / xáo trộn thứ tự đầu vào: [right, left, center]
    shuffled = [paths[2], paths[0], paths[1]]

    cfg = StitchingConfig(
        detector="sift",
        matcher_mode="custom",
        homography_mode="custom",
        projection="cylindrical",
        blending_mode="seam_multiband",
        auto_order=True,
        generate_visuals=False,
    )
    res = PanoramaStitcher(cfg).stitch_paths(shuffled)
    assert res.panorama.shape[1] > res.panorama.shape[0]
    # Kiểm tra khôi phục đúng thứ tự không gian (trái -> giữa -> phải hoặc ngược lại nếu đối xứng)
    ordered_names = [res.image_names[i] for i in res.visual_order]
    assert ordered_names == ["library_left.jpg", "library_center.jpg", "library_right.jpg"]
    assert res.image_names[res.anchor_index] == "library_center.jpg"
