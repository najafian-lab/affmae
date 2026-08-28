"""Segmentation metric tests.

The perfect and disjoint cases pin ``compute_iou``/``compute_dice`` at their
analytic values, which is what catches an unbound or mis-reduced return.
"""

import pytest
import torch

from affmae.eval.metrics import compute_dice, compute_iou


def _one_hot_logits(labels, num_classes):
    """Build logits that argmax exactly to ``labels``.

    Args:
        labels: [B,H,W] int tensor of class indices.
        num_classes: int, channel count of the returned logits.
    Returns:
        [B,C,H,W] float logits.
    """
    b, h, w = labels.shape
    logits = torch.zeros(b, num_classes, h, w)
    logits.scatter_(1, labels.unsqueeze(1), 10.0)
    return logits


@pytest.mark.parametrize("metric", [compute_iou, compute_dice])
class TestMetricsAreCallable:
    def test_returns_finite_values(self, metric):
        """compute_dice raised NameError before the fix; this pins it."""
        targets = torch.zeros(2, 8, 8, dtype=torch.long)
        targets[:, :4, :] = 1
        out = metric(_one_hot_logits(targets, 3), targets)
        assert torch.isfinite(out).all()

    def test_shape_is_batch_by_foreground_classes(self, metric):
        """Background (index 0) is excluded, so C-1 columns."""
        num_classes = 4
        targets = torch.randint(0, num_classes, (3, 8, 8))
        out = metric(_one_hot_logits(targets, num_classes), targets)
        assert out.shape == (3, num_classes - 1)

    def test_perfect_prediction_scores_one(self, metric):
        targets = torch.zeros(1, 8, 8, dtype=torch.long)
        targets[:, :4, :4] = 1
        targets[:, 4:, 4:] = 2
        out = metric(_one_hot_logits(targets, 3), targets)
        torch.testing.assert_close(out, torch.ones_like(out), rtol=1e-3, atol=1e-3)

    def test_disjoint_prediction_scores_zero(self, metric):
        """Predicting the wrong foreground class everywhere scores ~0."""
        targets = torch.ones(1, 8, 8, dtype=torch.long)
        preds = torch.full((1, 8, 8), 2, dtype=torch.long)
        out = metric(_one_hot_logits(preds, 3), targets)
        assert out[0, 0].item() == pytest.approx(0.0, abs=1e-3)

    def test_values_are_bounded(self, metric):
        targets = torch.randint(0, 3, (2, 16, 16))
        preds = torch.randint(0, 3, (2, 16, 16))
        out = metric(_one_hot_logits(preds, 3), targets)
        assert (out >= -1e-6).all() and (out <= 1.0 + 1e-6).all()


def test_dice_exceeds_iou_on_partial_overlap():
    """Dice >= IoU for the same partial overlap; catches a swapped formula."""
    targets = torch.zeros(1, 8, 8, dtype=torch.long)
    targets[:, :4, :] = 1
    preds = torch.zeros(1, 8, 8, dtype=torch.long)
    preds[:, 2:6, :] = 1  # half the target region, half background

    logits = _one_hot_logits(preds, 2)
    iou = compute_iou(logits, targets)[0, 0].item()
    dice = compute_dice(logits, targets)[0, 0].item()

    assert 0.0 < iou < 1.0, iou
    assert dice >= iou - 1e-6, f"dice {dice} < iou {iou}"
