"""Segmentation metrics: per-class IoU and Dice from logits.

Split out of ``affmae/losses.py``. They are metrics, not objectives -- nothing in
that module used them, and keeping them there forced ``eval/`` to import a
training module to score a checkpoint.
"""

import torch
import torch.nn.functional as F


def compute_iou(logits, targets, smooth=1e-6):
    """
    Computes IoU per image, per class.

    Args:
        logits: shape (B, C, H, W)
        targets: shape (B, H, W) containing class indices
    Returns:
        ious: shape (B, C-1) - IoU for each image and each class (excluding BG)
    """
    B, C, H, W = logits.shape

    pred_labels = torch.argmax(logits, dim=1)

    # Convert to one-hot: [B, H, W] -> [B, H, W, C] -> [B, C, H, W]
    pred_oh = F.one_hot(pred_labels, num_classes=C).permute(0, 3, 1, 2).float()
    target_oh = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()

    # Ignore background class (index 0) -> Shape becomes [B, C-1, H, W]
    pred_oh = pred_oh[:, 1:, :, :]
    target_oh = target_oh[:, 1:, :, :]

    # Sum over spatial dimensions (H, W) -> Shape [B, C-1]
    intersection = (pred_oh * target_oh).sum(dim=(2, 3))
    total = pred_oh.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
    union = total - intersection

    # Compute IoU
    ious = (intersection + smooth) / (union + smooth)

    return ious

def compute_dice(logits, targets, smooth=1e-6):
    """
    Computes IoU per image, per class.

    Args:
        logits: shape (B, C, H, W)
        targets: shape (B, H, W) containing class indices
    Returns:
        ious: shape (B, C-1) - IoU for each image and each class (excluding BG)
    """
    B, C, H, W = logits.shape

    pred_labels = torch.argmax(logits, dim=1)

    # Convert to one-hot: [B, H, W] -> [B, H, W, C] -> [B, C, H, W]
    pred_oh = F.one_hot(pred_labels, num_classes=C).permute(0, 3, 1, 2).float()
    target_oh = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()

    # Ignore background class (index 0) -> Shape becomes [B, C-1, H, W]
    pred_oh = pred_oh[:, 1:, :, :]
    target_oh = target_oh[:, 1:, :, :]

    # Sum over spatial dimensions (H, W) -> Shape [B, C-1]
    # NB: cardinality sums the one-hot prediction, not pred_labels; the latter
    # is [B, H, W] and summing it over dim=(2, 3) raises IndexError.
    cardinality = pred_oh.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
    intersection = (pred_oh * target_oh).sum(dim=(2, 3))

    # Dice = 2|A n B| / (|A| + |B|)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth)

    return dice_score
