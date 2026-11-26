"""Module tự động sắp xếp thứ tự ảnh & xây dựng đồ thị căn chỉnh toàn cục.

Khi người dùng tải lên N ảnh (2..10) theo thứ tự bất kỳ:
1. Xây dựng đồ thị khớp cặp (Match Graph) với trọng số là số lượng RANSAC inliers.
2. Dùng thuật toán Kruskal tìm Cây khung lớn nhất (Maximum Spanning Tree - MST)
   nhằm ưu tiên các cặp ảnh có độ chồng lấn mạnh và ổn định nhất.
3. Kiểm tra tính liên thông: nếu có ảnh rời rạc (không chồng lấn với cụm chính),
   trả về thông báo lỗi cụ thể chỉ rõ tên ảnh lỗi thay vì làm sập chương trình.
4. Chọn ảnh gốc trung tâm (Center Anchor) bằng cách tối thiểu hóa độ lệch tâm
   (eccentricity) trên cây MST, giúp giảm một nửa độ méo phối cảnh về hai đầu.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import StitchingConfig
from .exceptions import DisconnectedImagesError, InsufficientMatchesError
from .features import FeatureMatch, ImageFeatures, match_features
from .homography import HomographyResult, estimate_homography


@dataclass
class PairwiseEdge:
    """Cạnh liên kết trong đồ thị ghép ảnh: ánh xạ src_idx -> dst_idx."""

    src_idx: int
    dst_idx: int
    H_src_to_dst: np.ndarray
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    reprojection_rmse: float
    pts_src: np.ndarray
    pts_dst: np.ndarray
    inlier_mask: np.ndarray


@dataclass
class AlignmentGraph:
    """Cấu trúc liên kết toàn cục của bộ ảnh đầu vào."""

    num_images: int
    anchor_idx: int
    mst_edges: List[PairwiseEdge]
    all_valid_edges: List[PairwiseEdge]
    inlier_matrix: np.ndarray  # (N, N) ma trận số inliers giữa mọi cặp ảnh
    global_homographies: List[np.ndarray]  # H_i->anchor cho mỗi ảnh i
    visual_order: List[int]  # Thứ tự trái -> phải của các ảnh trong khung toàn cảnh


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _evaluate_pair(
    i: int,
    j: int,
    features_list: List[ImageFeatures],
    image_shapes: List[Tuple[int, int]],
    config: StitchingConfig,
) -> Optional[PairwiseEdge]:
    """Khớp đặc trưng và ước lượng Homography (i -> j) cho một cặp ảnh."""
    matches, pts_i, pts_j = match_features(
        features_list[i],
        features_list[j],
        mode=config.matcher_mode,
        ratio_threshold=config.ratio_threshold,
        cross_check=config.cross_check,
    )
    if len(matches) < config.min_inliers:
        return None

    h_res: HomographyResult = estimate_homography(
        pts_i,
        pts_j,
        mode=config.homography_mode,
        ransac_threshold=config.ransac_threshold,
        max_iters=config.ransac_max_iters,
        confidence=config.ransac_confidence,
        min_inliers=config.min_inliers,
        src_shape=image_shapes[i],
        seed=config.random_seed + i * 31 + j,
    )

    if not h_res.is_valid or h_res.H is None:
        return None
    if h_res.inlier_ratio < config.min_inlier_ratio:
        return None

    return PairwiseEdge(
        src_idx=i,
        dst_idx=j,
        H_src_to_dst=h_res.H,
        num_matches=len(matches),
        num_inliers=h_res.num_inliers,
        inlier_ratio=h_res.inlier_ratio,
        reprojection_rmse=h_res.reprojection_rmse,
        pts_src=pts_i,
        pts_dst=pts_j,
        inlier_mask=h_res.inlier_mask,
    )


def _select_center_anchor(n: int, mst_edges: List[PairwiseEdge]) -> int:
    """Chọn nút trung tâm của cây MST (tối thiểu hóa khoảng cách hop lớn nhất tới mọi nút khác)."""
    adj: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(n)}
    for e in mst_edges:
        adj[e.src_idx].append((e.dst_idx, e.num_inliers))
        adj[e.dst_idx].append((e.src_idx, e.num_inliers))

    best_node = 0
    best_score = (10**9, 10**9, 0)  # (eccentricity, sum_dist, -total_weight)

    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v, _ in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        ecc = max(dist)
        sum_d = sum(dist)
        total_w = sum(w for _, w in adj[start])
        score = (ecc, sum_d, -total_w)
        if score < best_score:
            best_score = score
            best_node = start

    return best_node


def _chain_homographies(
    n: int, anchor_idx: int, mst_edges: List[PairwiseEdge]
) -> List[np.ndarray]:
    """Duyệt BFS từ ảnh gốc trung tâm (anchor_idx) để tính ma trận H_{i -> anchor} cho mọi ảnh."""
    # Lưu danh sách kề (neighbor, H_{neighbor -> current})
    adj: Dict[int, List[Tuple[int, np.ndarray]]] = {i: [] for i in range(n)}
    for e in mst_edges:
        # e.H_src_to_dst ánh xạ src -> dst
        adj[e.dst_idx].append((e.src_idx, e.H_src_to_dst))
        adj[e.src_idx].append((e.dst_idx, np.linalg.inv(e.H_src_to_dst)))

    global_H: List[Optional[np.ndarray]] = [None] * n
    global_H[anchor_idx] = np.eye(3, dtype=np.float64)

    q = deque([anchor_idx])
    while q:
        curr = q.popleft()
        H_curr_to_anchor = global_H[curr]
        assert H_curr_to_anchor is not None
        for nxt, H_nxt_to_curr in adj[curr]:
            if global_H[nxt] is None:
                H_nxt_to_anchor = H_curr_to_anchor @ H_nxt_to_curr
                if abs(H_nxt_to_anchor[2, 2]) > 1e-12:
                    H_nxt_to_anchor = H_nxt_to_anchor / H_nxt_to_anchor[2, 2]
                global_H[nxt] = H_nxt_to_anchor
                q.append(nxt)

    return [h if h is not None else np.eye(3, dtype=np.float64) for h in global_H]


def build_alignment_graph(
    features_list: List[ImageFeatures],
    image_shapes: List[Tuple[int, int]],
    image_names: List[str],
    config: StitchingConfig,
) -> AlignmentGraph:
    """Xây dựng đồ thị khớp, tự động tìm thứ tự ghép (nếu bật auto_order) và tính H toàn cục."""
    n = len(features_list)
    inlier_matrix = np.zeros((n, n), dtype=np.int32)
    all_valid_edges: List[PairwiseEdge] = []

    if config.auto_order:
        # Khớp tất cả các cặp (i, j) để tự động nhận diện cấu trúc chồng lấn bất kể thứ tự tải lên
        for i in range(n):
            for j in range(i + 1, n):
                edge = _evaluate_pair(i, j, features_list, image_shapes, config)
                if edge is not None:
                    all_valid_edges.append(edge)
                    inlier_matrix[i, j] = edge.num_inliers
                    inlier_matrix[j, i] = edge.num_inliers

        if not all_valid_edges:
            raise InsufficientMatchesError(
                message="Không tìm thấy cặp ảnh nào có đủ điểm khớp tin cậy để ghép.",
                suggestion="Kiểm tra xem các ảnh có chụp cùng một cảnh với độ chồng lấn 25–50% hay không, hoặc giảm ngưỡng Lowe's ratio / Min inliers.",
                details={"num_images": n, "min_inliers": config.min_inliers},
            )

        # Kruskal tìm Maximum Spanning Tree (ưu tiên cạnh có nhiều inliers & RMSE thấp)
        sorted_edges = sorted(
            all_valid_edges,
            key=lambda e: (e.num_inliers, -e.reprojection_rmse),
            reverse=True,
        )
        uf = _UnionFind(n)
        mst_edges: List[PairwiseEdge] = []
        for edge in sorted_edges:
            if uf.union(edge.src_idx, edge.dst_idx):
                mst_edges.append(edge)
                if len(mst_edges) == n - 1:
                    break

        # Kiểm tra liên thông: tìm các thành phần liên thông nếu thiếu cạnh
        if len(mst_edges) < n - 1:
            comp_map: Dict[int, List[int]] = {}
            for i in range(n):
                r = uf.find(i)
                comp_map.setdefault(r, []).append(i)
            groups = sorted(comp_map.values(), key=len, reverse=True)
            isolated_indices = [idx for grp in groups[1:] for idx in grp]
            isolated_names = [image_names[idx] for idx in isolated_indices]
            raise DisconnectedImagesError(
                message=f"Không thể liên kết tất cả ảnh thành một toàn cảnh liên tục. Các ảnh không đủ vùng chồng lấn: {', '.join(isolated_names)}.",
                suggestion="Hãy loại bỏ các ảnh không thuộc cùng khung cảnh hoặc bổ sung ảnh trung gian có độ chồng lấn ≥ 25%.",
                details={
                    "disconnected_images": isolated_names,
                    "components": [[image_names[i] for i in grp] for grp in groups],
                },
            )
    else:
        # Chế độ thứ tự thủ công: nối tuần tự (0-1), (1-2), ..., (N-2, N-1)
        mst_edges = []
        for i in range(n - 1):
            edge = _evaluate_pair(i, i + 1, features_list, image_shapes, config)
            if edge is None:
                raise InsufficientMatchesError(
                    message=f"Không thể khớp cặp ảnh liên tiếp '{image_names[i]}' và '{image_names[i + 1]}'.",
                    suggestion="Hãy bật tùy chọn 'Tự động sắp xếp thứ tự ảnh (Auto-Order)' hoặc sắp xếp lại thứ tự ảnh theo đúng chiều chụp.",
                    details={"pair": [image_names[i], image_names[i + 1]]},
                )
            mst_edges.append(edge)
            all_valid_edges.append(edge)
            inlier_matrix[i, i + 1] = edge.num_inliers
            inlier_matrix[i + 1, i] = edge.num_inliers

    anchor_idx = _select_center_anchor(n, mst_edges)
    global_H = _chain_homographies(n, anchor_idx, mst_edges)

    # Sắp xếp thứ tự trực quan từ trái sang phải theo tọa độ x tâm ảnh trên khung chiếu toàn cục
    center_xs: List[Tuple[float, int]] = []
    for i in range(n):
        h, w = image_shapes[i][:2]
        c_hom = global_H[i] @ np.array([w * 0.5, h * 0.5, 1.0], dtype=np.float64)
        cx = c_hom[0] / max(c_hom[2], 1e-9)
        center_xs.append((float(cx), i))
    center_xs.sort(key=lambda t: t[0])
    visual_order = [idx for _, idx in center_xs]

    return AlignmentGraph(
        num_images=n,
        anchor_idx=anchor_idx,
        mst_edges=mst_edges,
        all_valid_edges=all_valid_edges,
        inlier_matrix=inlier_matrix,
        global_homographies=global_H,
        visual_order=visual_order,
    )


def draw_match_graph_preview(
    graph: AlignmentGraph, image_names: List[str]
) -> np.ndarray:
    """Vẽ sơ đồ ma trận & cây khung liên kết ảnh (Auto-Order Match Graph)."""
    n = graph.num_images
    w, h = 920, 420
    canvas = np.full((h, w, 3), (42, 23, 15), dtype=np.uint8)  # BGR of #0f172a

    # Tiêu đề
    cv2.putText(
        canvas,
        f"Auto-Order Alignment Graph (Anchor: #{graph.anchor_idx + 1} {image_names[graph.anchor_idx][:22]})",
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (235, 244, 255),
        1,
        cv2.LINE_AA,
    )

    # Sắp xếp các nút theo thứ tự visual_order từ trái sang phải
    order = graph.visual_order
    pos: Dict[int, Tuple[int, int]] = {}
    margin_x = 90
    usable_w = w - 2 * margin_x
    for rank, img_idx in enumerate(order):
        x = int(margin_x + (usable_w * rank / max(1, n - 1))) if n > 1 else w // 2
        y = 195 + (28 if rank % 2 == 1 else -28)
        pos[img_idx] = (x, y)

    mst_set = {(min(e.src_idx, e.dst_idx), max(e.src_idx, e.dst_idx)) for e in graph.mst_edges}

    # Vẽ các cạnh phụ (màu xám mờ)
    for e in graph.all_valid_edges:
        key = (min(e.src_idx, e.dst_idx), max(e.src_idx, e.dst_idx))
        if key not in mst_set:
            cv2.line(canvas, pos[e.src_idx], pos[e.dst_idx], (75, 85, 105), 1, cv2.LINE_AA)

    # Vẽ các cạnh thuộc cây khung MST (màu cyan nổi bật)
    for e in graph.mst_edges:
        p1, p2 = pos[e.src_idx], pos[e.dst_idx]
        cv2.line(canvas, p1, p2, (240, 200, 60), 3, cv2.LINE_AA)
        mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 10
        txt = f"{e.num_inliers} inliers"
        cv2.rectangle(canvas, (mx - 42, my - 14), (mx + 46, my + 5), (15, 23, 42), -1)
        cv2.putText(
            canvas, txt, (mx - 38, my), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 235, 190), 1, cv2.LINE_AA
        )

    # Vẽ nút ảnh
    for rank, img_idx in enumerate(order):
        cx, cy = pos[img_idx]
        is_anchor = img_idx == graph.anchor_idx
        fill = (255, 165, 30) if is_anchor else (56, 130, 246)
        border = (255, 240, 180) if is_anchor else (180, 215, 255)
        cv2.circle(canvas, (cx, cy), 30, fill, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 30, border, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"#{img_idx + 1}",
            (cx - 15, cy + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        short_name = image_names[img_idx]
        if len(short_name) > 14:
            short_name = short_name[:12] + ".."
        cv2.putText(
            canvas,
            short_name,
            (cx - 46, cy + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (200, 214, 230),
            1,
            cv2.LINE_AA,
        )
        if is_anchor:
            cv2.putText(
                canvas,
                "ANCHOR",
                (cx - 28, cy - 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (120, 220, 255),
                1,
                cv2.LINE_AA,
            )

    order_str = " -> ".join([f"#{i + 1}" for i in order])
    cv2.putText(
        canvas,
        f"Left-to-Right Order: {order_str}",
        (24, h - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (170, 210, 245),
        1,
        cv2.LINE_AA,
    )
    return canvas
