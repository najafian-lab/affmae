from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize


@dataclass
class SegmentPolyline:
    """One basement-membrane segment with the slits found along it.

    Attributes:
        segment_id: index within the image.
        mask: [H, W] bool, the dilated PGBMI component.
        skeleton_mask: [H, W] bool, its one-pixel-wide centreline.
        skeleton_points_xy: [P, 2] centreline points ordered along the segment.
        skeleton_arc_lengths: [P] cumulative arc length at each centreline point.
        slit_points_xy: [S, 2] slit centroids projected onto the centreline,
            ordered by arc length.
        slit_arc_lengths: [S] arc length of each slit along the centreline.
    """
    segment_id: int
    mask: np.ndarray
    skeleton_mask: np.ndarray
    skeleton_points_xy: np.ndarray
    skeleton_arc_lengths: np.ndarray
    slit_points_xy: np.ndarray
    slit_arc_lengths: np.ndarray

    @property
    def fpw_arcs(self) -> np.ndarray:
        """Foot-process widths: gaps between consecutive slits, in pixels.

        Returns:
            [S - 1] array of arc-length differences, empty when fewer than two slits
            were found on this segment.
        """
        return _fpw_arcs_from_slit_arc_lengths(self.slit_arc_lengths)


def _neighbors8(y: int, x: int, shape: tuple[int, int]):
    h, w = shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx, float(np.hypot(dy, dx))


def _build_skeleton_graph(
    skeleton_mask: np.ndarray,
) -> tuple[np.ndarray, dict[tuple[int, int], int], list[list[tuple[int, float]]]]:
    coords_yx = np.argwhere(skeleton_mask)
    coord_to_idx = {tuple(coord): i for i, coord in enumerate(coords_yx)}
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(len(coords_yx))]

    for i, (y, x) in enumerate(coords_yx):
        for ny, nx, dist in _neighbors8(int(y), int(x), skeleton_mask.shape):
            j = coord_to_idx.get((ny, nx))
            if j is not None:
                adjacency[i].append((j, dist))
    return coords_yx, coord_to_idx, adjacency


def _skeleton_degrees(adjacency: list[list[tuple[int, float]]]) -> np.ndarray:
    return np.asarray([len(neighbors) for neighbors in adjacency], dtype=np.int64)


def _dijkstra(
    adjacency: list[list[tuple[int, float]]],
    start_idx: int,
    blocked: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    blocked = blocked or set()
    distances = np.full(len(adjacency), np.inf, dtype=np.float64)
    predecessors = np.full(len(adjacency), -1, dtype=np.int64)
    if start_idx in blocked:
        return distances, predecessors
    distances[start_idx] = 0.0
    heap: list[tuple[float, int]] = [(0.0, start_idx)]
    while heap:
        dist_i, i = heapq.heappop(heap)
        if dist_i > distances[i]:
            continue
        if i in blocked:
            continue
        for j, edge_dist in adjacency[i]:
            if j in blocked:
                continue
            next_dist = dist_i + edge_dist
            if next_dist < distances[j]:
                distances[j] = next_dist
                predecessors[j] = i
                heapq.heappush(heap, (next_dist, j))
    return distances, predecessors


def _reconstruct_path(predecessors: np.ndarray, start_idx: int, end_idx: int) -> list[int]:
    path = [int(end_idx)]
    current = int(end_idx)
    while current != start_idx:
        current = int(predecessors[current])
        if current < 0:
            return [int(start_idx)]
        path.append(current)
    path.reverse()
    return path


def _path_to_xy_and_arcs(coords_yx: np.ndarray, path_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    if not path_indices:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)

    ordered_yx = coords_yx[np.asarray(path_indices, dtype=np.int64)].astype(np.float64)
    ordered_xy = ordered_yx[:, ::-1]
    arc_lengths = np.zeros(len(ordered_xy), dtype=np.float64)
    if len(ordered_xy) > 1:
        steps = np.linalg.norm(np.diff(ordered_xy, axis=0), axis=1)
        arc_lengths[1:] = np.cumsum(steps)
    return ordered_xy, arc_lengths


def _lexicographic_first(indices: np.ndarray, coords_yx: np.ndarray) -> int:
    ordered = indices[np.lexsort((coords_yx[indices, 1], coords_yx[indices, 0]))]
    return int(ordered[0])


def _graph_diameter_path(
    adjacency: list[list[tuple[int, float]]],
    starts: np.ndarray,
) -> list[int]:
    best_distance = -np.inf
    best_path = [int(starts[0])]
    for start_idx in starts:
        distances, predecessors = _dijkstra(adjacency, int(start_idx))
        finite_starts = starts[np.isfinite(distances[starts])]
        if finite_starts.size == 0:
            continue
        end_idx = int(finite_starts[np.argmax(distances[finite_starts])])
        if distances[end_idx] > best_distance:
            best_distance = float(distances[end_idx])
            best_path = _reconstruct_path(predecessors, int(start_idx), end_idx)
    return best_path


def _farthest_point_diameter_path(adjacency: list[list[tuple[int, float]]], start_idx: int) -> list[int]:
    distances, _ = _dijkstra(adjacency, start_idx)
    finite = np.flatnonzero(np.isfinite(distances))
    if finite.size == 0:
        return [start_idx]
    a_idx = int(finite[np.argmax(distances[finite])])
    distances, predecessors = _dijkstra(adjacency, a_idx)
    finite = np.flatnonzero(np.isfinite(distances))
    b_idx = int(finite[np.argmax(distances[finite])])
    return _reconstruct_path(predecessors, a_idx, b_idx)


def _ordered_loop_from_branch_or_break(
    coords_yx: np.ndarray,
    adjacency: list[list[tuple[int, float]]],
    degrees: np.ndarray,
) -> list[int] | None:
    branch_indices = np.flatnonzero(degrees >= 3)
    if branch_indices.size > 0:
        start_idx = _lexicographic_first(branch_indices, coords_yx)
        neighbor_indices = [idx for idx, _ in adjacency[start_idx]]
        best_path = None
        best_distance = -np.inf
        for i, first_neighbor in enumerate(neighbor_indices):
            distances, predecessors = _dijkstra(adjacency, first_neighbor, blocked={start_idx})
            for second_neighbor in neighbor_indices[i + 1 :]:
                if not np.isfinite(distances[second_neighbor]):
                    continue
                if distances[second_neighbor] > best_distance:
                    middle_path = _reconstruct_path(predecessors, first_neighbor, second_neighbor)
                    best_path = [start_idx] + middle_path + [start_idx]
                    best_distance = float(distances[second_neighbor])
        return best_path

    if len(coords_yx) < 3:
        return None

    all_indices = np.arange(len(coords_yx))
    start_idx = _lexicographic_first(all_indices, coords_yx)
    neighbors = sorted(
        adjacency[start_idx],
        key=lambda item: (coords_yx[item[0], 0], coords_yx[item[0], 1]),
    )
    if not neighbors:
        return [start_idx]

    path = [start_idx]
    previous = start_idx
    current = int(neighbors[0][0])
    visited = {start_idx}
    while current != start_idx:
        path.append(current)
        visited.add(current)
        next_candidates = [
            idx
            for idx, _ in sorted(
                adjacency[current],
                key=lambda item: (coords_yx[item[0], 0], coords_yx[item[0], 1]),
            )
            if idx != previous
        ]
        if start_idx in next_candidates and len(path) > 2:
            path.append(start_idx)
            return path
        unvisited = [idx for idx in next_candidates if idx not in visited]
        if not unvisited:
            return None
        previous, current = current, int(unvisited[0])
    return path


def _order_skeleton(
    skeleton_mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    del rng  # Ordering is deterministic; keep the parameter for caller compatibility.
    coords_yx, _, adjacency = _build_skeleton_graph(skeleton_mask)
    if coords_yx.size == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)

    degrees = _skeleton_degrees(adjacency)
    endpoints = np.flatnonzero(degrees <= 1)
    if endpoints.size >= 2:
        path_indices = _graph_diameter_path(adjacency, endpoints)
    else:
        path_indices = _ordered_loop_from_branch_or_break(coords_yx, adjacency, degrees)
        if path_indices is None:
            start_idx = _lexicographic_first(np.arange(len(coords_yx)), coords_yx)
            path_indices = _farthest_point_diameter_path(adjacency, start_idx)

    return _path_to_xy_and_arcs(coords_yx, path_indices)


def _extract_slit_centers(
    slit_mask: np.ndarray,
    min_area: float,
    max_area: float,
    min_circularity: float,
) -> np.ndarray:
    labeled = label(slit_mask, connectivity=2)
    centers: list[tuple[float, float]] = []
    for region in regionprops(labeled):
        area = float(region.area)
        if area < min_area or area > max_area:
            continue

        perimeter = float(region.perimeter)
        circularity = 0.0 if perimeter <= 0 else 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        cy, cx = region.centroid
        centers.append((float(cx), float(cy)))

    if not centers:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(centers, dtype=np.float64)


def _project_slits_to_skeleton(
    slit_points_xy: np.ndarray,
    skeleton_points_xy: np.ndarray,
    skeleton_arc_lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if slit_points_xy.size == 0 or skeleton_points_xy.size == 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    nearest = cdist(slit_points_xy, skeleton_points_xy).argmin(axis=1)
    arc_lengths = skeleton_arc_lengths[nearest]
    order = np.argsort(arc_lengths, kind="mergesort")
    return slit_points_xy[order], arc_lengths[order]


def _fpw_arcs_from_slit_arc_lengths(slit_arc_lengths: np.ndarray) -> np.ndarray:
    if slit_arc_lengths.size < 2:
        return np.empty((0,), dtype=np.float64)
    return np.diff(np.sort(slit_arc_lengths.astype(np.float64)))


def _skeleton_endpoints(skeleton_mask: np.ndarray) -> np.ndarray:
    coords_yx, _, adjacency = _build_skeleton_graph(skeleton_mask)
    if coords_yx.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    degrees = _skeleton_degrees(adjacency)
    endpoint_indices = np.flatnonzero(degrees <= 1)
    if endpoint_indices.size == 0:
        endpoint_indices = np.arange(len(coords_yx))
    return coords_yx[endpoint_indices].astype(np.float64)[:, ::-1]


def _align_pred_to_gt(
    gt_segment: SegmentPolyline,
    pred_segment: SegmentPolyline,
) -> tuple[np.ndarray, np.ndarray, str]:
    if pred_segment.slit_points_xy.size == 0 or pred_segment.slit_arc_lengths.size == 0:
        return pred_segment.slit_points_xy, pred_segment.slit_arc_lengths, "forward"
    if gt_segment.skeleton_points_xy.size == 0 or pred_segment.skeleton_points_xy.size == 0:
        return pred_segment.slit_points_xy, pred_segment.slit_arc_lengths, "forward"

    gt_start_xy = gt_segment.skeleton_points_xy[0]
    candidates_xy = _skeleton_endpoints(pred_segment.skeleton_mask)
    if candidates_xy.size == 0:
        return pred_segment.slit_points_xy, pred_segment.slit_arc_lengths, "forward"

    pred_start_xy = candidates_xy[np.argmin(((candidates_xy - gt_start_xy) ** 2).sum(axis=1))]
    pred_current_start_xy = pred_segment.skeleton_points_xy[0]
    if np.allclose(pred_start_xy, pred_current_start_xy):
        return pred_segment.slit_points_xy, pred_segment.slit_arc_lengths, "forward"

    max_arc = float(pred_segment.skeleton_arc_lengths.max()) if pred_segment.skeleton_arc_lengths.size else 0.0
    aligned_arc_lengths = max_arc - pred_segment.slit_arc_lengths[::-1]
    return pred_segment.slit_points_xy[::-1], aligned_arc_lengths, "reverse"


def extract_segment_polylines(
    mask: np.ndarray,
    pgbmi_class: int = 1,
    slit_class: int = 2,
    pgbmi_dilate: int = 3,
    slit_min_area: float = 4.0,
    slit_max_area: float = 400.0,
    slit_min_circularity: float = 0.4,
    min_pgbmi_area: float = 16.0,
    rng: np.random.Generator | None = None,
) -> list[SegmentPolyline]:
    """Find basement-membrane segments and the slits along each one.

    Dilates and labels the PGBMI class, skeletonizes each component to an ordered
    centreline, then assigns filtered slit blobs to the nearest centreline point so
    they can be ordered by arc length.

    Args:
        mask: [H, W] integer label mask.
        pgbmi_class: label index of the basement membrane.
        slit_class: label index of the slits.
        pgbmi_dilate: dilation radius applied to the PGBMI mask, in pixels; joins
            components broken by segmentation noise.
        slit_min_area: smallest accepted slit blob, in pixels.
        slit_max_area: largest accepted slit blob, in pixels.
        slit_min_circularity: minimum circularity for a slit blob, which rejects
            elongated artifacts.
        min_pgbmi_area: smallest accepted PGBMI component, in pixels.
        rng: generator used to break ties when ordering the centreline.
    Returns:
        List of SegmentPolyline, one per surviving component.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    mask_np = np.asarray(mask)
    # Slits are annotated on top of PGBMI, so include them when recovering the
    # membrane support from the exclusive multiclass mask.
    pgbmi_mask = (mask_np == pgbmi_class) | (mask_np == slit_class)
    slit_mask = mask_np == slit_class
    labeled_pgbmi = label(pgbmi_mask, connectivity=2)

    segments: list[SegmentPolyline] = []
    for region in regionprops(labeled_pgbmi):
        if float(region.area) < min_pgbmi_area:
            continue

        segment_id = int(region.label)
        segment_mask = labeled_pgbmi == segment_id
        skeleton_mask = skeletonize(segment_mask)
        skeleton_points_xy, skeleton_arc_lengths = _order_skeleton(skeleton_mask, rng)
        if skeleton_points_xy.size == 0:
            continue

        if pgbmi_dilate > 0:
            dilated_segment = binary_dilation(segment_mask, iterations=int(pgbmi_dilate))
        else:
            dilated_segment = segment_mask
        segment_slit_mask = slit_mask & dilated_segment
        slit_centers_xy = _extract_slit_centers(
            segment_slit_mask,
            min_area=slit_min_area,
            max_area=slit_max_area,
            min_circularity=slit_min_circularity,
        )
        slit_points_xy, slit_arc_lengths = _project_slits_to_skeleton(
            slit_centers_xy,
            skeleton_points_xy,
            skeleton_arc_lengths,
        )

        segments.append(
            SegmentPolyline(
                segment_id=segment_id,
                mask=segment_mask,
                skeleton_mask=skeleton_mask,
                skeleton_points_xy=skeleton_points_xy,
                skeleton_arc_lengths=skeleton_arc_lengths,
                slit_points_xy=slit_points_xy,
                slit_arc_lengths=slit_arc_lengths,
            )
        )

    return segments


def match_segments_iou(
    gt_segments: list[SegmentPolyline],
    pred_segments: list[SegmentPolyline],
    min_iou: float = 0.1,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Match predicted segments to ground-truth ones by mask IoU.

    Uses a Hungarian assignment rather than greedy matching, so one prediction
    cannot claim several ground-truth segments.

    Args:
        gt_segments: ground-truth SegmentPolyline list.
        pred_segments: predicted SegmentPolyline list.
        min_iou: IoU below which a pairing is rejected.
    Returns:
        ``(matches, unmatched_gt, unmatched_pred)``, where matches holds
        ``(gt_index, pred_index, iou)`` triples.
    """
    if not gt_segments:
        return [], [], list(range(len(pred_segments)))
    if not pred_segments:
        return [], list(range(len(gt_segments))), []

    iou_matrix = np.zeros((len(gt_segments), len(pred_segments)), dtype=np.float64)
    for i, gt in enumerate(gt_segments):
        for j, pred in enumerate(pred_segments):
            intersection = np.logical_and(gt.mask, pred.mask).sum()
            union = np.logical_or(gt.mask, pred.mask).sum()
            iou_matrix[i, j] = 0.0 if union == 0 else intersection / union

    rows, cols = linear_sum_assignment(1.0 - iou_matrix)
    matches: list[tuple[int, int, float]] = []
    matched_gt = set()
    matched_pred = set()
    for row, col in zip(rows, cols):
        iou = float(iou_matrix[row, col])
        if iou >= min_iou:
            matches.append((int(row), int(col), iou))
            matched_gt.add(int(row))
            matched_pred.add(int(col))

    unmatched_gt = [i for i in range(len(gt_segments)) if i not in matched_gt]
    unmatched_pred = [i for i in range(len(pred_segments)) if i not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def segment_metrics(
    gt_segment: SegmentPolyline,
    pred_segment: SegmentPolyline,
    pixel_size: float = 1.0,
    grid_scale_xy: tuple[float, float] = (1.0, 1.0),
) -> dict[str, Any]:
    """Compare one matched pair of segments.

    Args:
        gt_segment: the ground-truth SegmentPolyline.
        pred_segment: the predicted SegmentPolyline.
        pixel_size: nanometres per pixel, for physical widths.
        grid_scale_xy: (sx, sy) rescaling to the reference evaluation grid, so runs
            at different input resolutions stay comparable.
    Returns:
        dict with ``fpw_mean_abs_error`` and the two per-side means it is the
        difference of.

    Note:
        Foot-process width is the reported metric, so it is the only one computed
        here. Discrete Frechet, Chamfer and slit-count error were reported
        alongside it; they measured curve agreement rather than width, and having
        four numbers per segment invited picking whichever looked best.
    """
    scale = float(pixel_size)
    grid_scale = np.asarray(grid_scale_xy, dtype=np.float64)
    arc_scale = float(np.sqrt(grid_scale[0] * grid_scale[1]))

    # Alignment still matters: it decides whether the predicted polyline runs the
    # same direction as the ground truth, which sets the arc lengths the widths
    # are measured along. Its direction label is no longer reported.
    _, pred_aligned_arc_lengths, _ = _align_pred_to_gt(gt_segment, pred_segment)
    gt_arcs = gt_segment.fpw_arcs * arc_scale * scale
    pred_arcs = _fpw_arcs_from_slit_arc_lengths(pred_aligned_arc_lengths) * arc_scale * scale

    fpw_mean_abs_error = float("nan")
    if gt_arcs.size > 0 and pred_arcs.size > 0:
        fpw_mean_abs_error = float(abs(gt_arcs.mean() - pred_arcs.mean()))

    return {
        "fpw_mean_gt": float(gt_arcs.mean()) if gt_arcs.size > 0 else float("nan"),
        "fpw_mean_pred": float(pred_arcs.mean()) if pred_arcs.size > 0 else float("nan"),
        "fpw_mean_abs_error": fpw_mean_abs_error,
    }


def summarize_values(values: list[float]) -> dict[str, float | int]:
    """Count, mean, median and standard deviation, ignoring non-finite entries.

    Args:
        values: list of floats, possibly containing nan or inf.
    Returns:
        dict with ``count``, ``mean``, ``median`` and ``std``; the statistics are
        nan when nothing finite remains.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
    }
