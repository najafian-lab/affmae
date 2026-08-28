import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from affmae.eval.fpw_geometry import (
    SegmentPolyline,
    extract_segment_polylines,
    match_segments_iou,
    segment_metrics,
)


def _draw_disk(mask, center_xy, radius, value):
    cx, cy = center_xy
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    mask[disk] = value


def _make_horizontal_mask(
    slit_centers,
    *,
    y=32,
    x0=8,
    x1=56,
    height=3,
    shape=(72, 72),
):
    mask = np.zeros(shape, dtype=np.int64)
    half = height // 2
    mask[y - half : y + half + 1, x0 : x1 + 1] = 1
    for center in slit_centers:
        _draw_disk(mask, center, radius=1, value=2)
    return mask


def _extract_one(mask):
    segments = extract_segment_polylines(
        mask,
        slit_min_area=1,
        slit_max_area=32,
        slit_min_circularity=0.0,
        min_pgbmi_area=1,
        pgbmi_dilate=1,
        rng=np.random.default_rng(0),
    )
    assert len(segments) == 1
    return segments[0]


def _reversed_segment(segment):
    max_arc = float(segment.skeleton_arc_lengths.max()) if segment.skeleton_arc_lengths.size else 0.0
    return SegmentPolyline(
        segment_id=segment.segment_id,
        mask=segment.mask,
        skeleton_mask=segment.skeleton_mask,
        skeleton_points_xy=segment.skeleton_points_xy[::-1],
        skeleton_arc_lengths=max_arc - segment.skeleton_arc_lengths[::-1],
        slit_points_xy=segment.slit_points_xy[::-1],
        slit_arc_lengths=max_arc - segment.slit_arc_lengths[::-1],
    )


def _max_consecutive_step(points_xy):
    if len(points_xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points_xy, axis=0), axis=1).max())


def test_horizontal_pgbmi_two_slits_identical_masks():
    mask = _make_horizontal_mask([(20, 32), (44, 32)])
    gt = _extract_one(mask)
    pred = _extract_one(mask)
    matches, unmatched_gt, unmatched_pred = match_segments_iou([gt], [pred])

    assert len(matches) == 1
    assert unmatched_gt == []
    assert unmatched_pred == []

    metrics = segment_metrics(gt, pred)
    assert metrics["fpw_mean_abs_error"] == pytest.approx(0.0)
    assert metrics["fpw_mean_gt"] == pytest.approx(metrics["fpw_mean_pred"])


def test_horizontal_pgbmi_predicted_slits_reversed():
    """A reversed prediction must still measure the same widths.

    Alignment is what makes this hold: the arc lengths widths are measured along
    are re-derived after flipping. The direction label it used to return is no
    longer reported, so this asserts the consequence instead.
    """
    mask = _make_horizontal_mask([(20, 32), (44, 32)])
    gt = _extract_one(mask)
    pred = _reversed_segment(_extract_one(mask))

    metrics = segment_metrics(gt, pred)
    assert metrics["fpw_mean_abs_error"] == pytest.approx(0.0)


def test_a_perpendicular_offset_does_not_change_the_width():
    """Shifting every slit one pixel across the membrane leaves spacing intact.

    This used to assert Chamfer 2.0 and Frechet 1.0, which is what those metrics
    are for: they see the displacement. Foot-process width is a spacing *along*
    the membrane, so it should be blind to it -- and that is the reason width is
    the metric worth reporting for this task.
    """
    gt = _extract_one(_make_horizontal_mask([(20, 32), (44, 32)]))
    pred = _extract_one(_make_horizontal_mask([(20, 33), (44, 33)]))
    metrics = segment_metrics(gt, pred)

    assert metrics["fpw_mean_abs_error"] == pytest.approx(0.0)


def test_grid_scale_xy_normalizes_the_reported_width():
    mask = np.ones((1, 11), dtype=bool)
    skeleton_points = np.asarray([(0.0, 0.0), (10.0, 0.0)])
    skeleton_arcs = np.asarray([0.0, 10.0])
    pred_slit_arcs = np.asarray([0.0, 12.0])
    gt = SegmentPolyline(
        segment_id=1,
        mask=mask,
        skeleton_mask=mask,
        skeleton_points_xy=skeleton_points,
        skeleton_arc_lengths=skeleton_arcs,
        slit_points_xy=np.asarray([(0.0, 0.0), (10.0, 0.0)]),
        slit_arc_lengths=skeleton_arcs,
    )
    pred = SegmentPolyline(
        segment_id=1,
        mask=mask,
        skeleton_mask=mask,
        skeleton_points_xy=skeleton_points,
        skeleton_arc_lengths=skeleton_arcs,
        slit_points_xy=np.asarray([(0.0, 1.0), (10.0, 1.0)]),
        slit_arc_lengths=pred_slit_arcs,
    )

    native_metrics = segment_metrics(gt, pred, grid_scale_xy=(1.0, 1.0))
    scaled_metrics = segment_metrics(gt, pred, grid_scale_xy=(2.0, 2.0))

    # The reported number is measured on the reference grid, so a run at half
    # the resolution scales up to the same physical width.
    assert native_metrics["fpw_mean_abs_error"] == pytest.approx(2.0)
    assert scaled_metrics["fpw_mean_abs_error"] == pytest.approx(4.0)


def test_two_disjoint_pgbmi_segments_matched_by_iou():
    mask = np.zeros((96, 96), dtype=np.int64)
    mask[29:32, 8:56] = 1
    mask[65:68, 16:72] = 1
    for center in [(20, 30), (44, 30), (32, 66), (60, 66)]:
        _draw_disk(mask, center, radius=1, value=2)

    gt_segments = extract_segment_polylines(
        mask,
        slit_min_area=1,
        slit_max_area=32,
        slit_min_circularity=0.0,
        min_pgbmi_area=1,
        pgbmi_dilate=1,
        rng=np.random.default_rng(0),
    )
    pred_segments = extract_segment_polylines(
        mask,
        slit_min_area=1,
        slit_max_area=32,
        slit_min_circularity=0.0,
        min_pgbmi_area=1,
        pgbmi_dilate=1,
        rng=np.random.default_rng(0),
    )
    matches, unmatched_gt, unmatched_pred = match_segments_iou(gt_segments, pred_segments)

    assert len(matches) == 2
    assert unmatched_gt == []
    assert unmatched_pred == []
    for gt_idx, pred_idx, _ in matches:
        metrics = segment_metrics(gt_segments[gt_idx], pred_segments[pred_idx])
        assert metrics["fpw_mean_abs_error"] == pytest.approx(0.0)


def test_fpw_arcs_are_nonnegative_after_reversal():
    gt = _extract_one(_make_horizontal_mask([(18, 32), (32, 32), (50, 32)]))
    pred = _reversed_segment(_extract_one(_make_horizontal_mask([(18, 32), (32, 32), (50, 32)])))
    metrics = segment_metrics(gt, pred)

    assert np.all(gt.fpw_arcs >= 0)
    assert np.all(pred.fpw_arcs >= 0)
    assert metrics["fpw_mean_abs_error"] == pytest.approx(0.0)


def test_branched_skeleton_uses_longest_shortest_path():
    mask = np.zeros((80, 80), dtype=np.int64)
    mask[40, 10:70] = 1
    mask[20:41, 40] = 1
    for center in [(18, 40), (62, 40)]:
        _draw_disk(mask, center, radius=1, value=2)

    segment = _extract_one(mask)
    skeleton_pixel_count = int(segment.skeleton_mask.sum())

    assert len(segment.skeleton_points_xy) < skeleton_pixel_count
    assert _max_consecutive_step(segment.skeleton_points_xy) <= np.sqrt(2) + 1e-6


def test_blob_skeleton_path_has_no_fan_jumps():
    mask = np.zeros((96, 96), dtype=np.int64)
    yy, xx = np.ogrid[:96, :96]
    ellipse = ((xx - 48) / 26) ** 2 + ((yy - 48) / 12) ** 2 <= 1
    mask[ellipse] = 1
    for center in [(30, 48), (48, 48), (66, 48)]:
        _draw_disk(mask, center, radius=1, value=2)

    segment = _extract_one(mask)

    assert len(segment.skeleton_points_xy) >= 2
    assert _max_consecutive_step(segment.skeleton_points_xy) <= np.sqrt(2) + 1e-6


def test_loop_skeleton_orders_around_loop_without_fan_jumps():
    mask = np.zeros((96, 96), dtype=np.int64)
    yy, xx = np.ogrid[:96, :96]
    radius = np.sqrt((xx - 48) ** 2 + (yy - 48) ** 2)
    mask[(radius >= 18) & (radius <= 20)] = 1
    for center in [(48, 29), (67, 48), (48, 67)]:
        _draw_disk(mask, center, radius=1, value=2)

    segment = _extract_one(mask)
    steps = np.linalg.norm(np.diff(segment.skeleton_points_xy, axis=0), axis=1)

    assert steps.max() <= np.sqrt(2) + 1e-6
    assert np.linalg.norm(segment.skeleton_points_xy[-1] - segment.skeleton_points_xy[0]) <= np.sqrt(2) + 1e-6


def test_loop_with_single_branch_uses_branch_point_as_start():
    mask = np.zeros((112, 112), dtype=np.int64)
    yy, xx = np.ogrid[:112, :112]
    radius = np.sqrt((xx - 56) ** 2 + (yy - 56) ** 2)
    mask[(radius >= 18) & (radius <= 20)] = 1
    mask[16:38, 56] = 1
    for center in [(56, 37), (75, 56), (56, 75)]:
        _draw_disk(mask, center, radius=1, value=2)

    segment = _extract_one(mask)
    expected_branch_xy = np.asarray([56.0, 37.0])
    steps = np.linalg.norm(np.diff(segment.skeleton_points_xy, axis=0), axis=1)

    assert np.linalg.norm(segment.skeleton_points_xy[0] - expected_branch_xy) <= 3.0
    assert steps.max() <= np.sqrt(2) + 1e-6
    assert np.linalg.norm(segment.skeleton_points_xy[-1] - segment.skeleton_points_xy[0]) <= np.sqrt(2) + 1e-6
