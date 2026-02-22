import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


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


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 3.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # calculate Cross-Entropy loss without reduction
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)

        # get probabilities of the correct class
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # calculate modulating factor
        modulating_factor = (1 - p_t).pow(self.gamma)

        # cinal pixel-wise focal loss
        pixel_loss = modulating_factor * ce_loss

        if self.reduction == 'mean':
            return pixel_loss.mean()
        elif self.reduction == 'sum':
            return pixel_loss.sum()
        else:
            return pixel_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, reduction="mean"):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    # exclude bg class
    def forward(self, pred, target):
        """
        pred: (B, C, H, W) - Raw Logits
        target: (B, H, W) - Class Indices
        """
        probs = F.softmax(pred, dim=1)

        num_classes = pred.shape[1] 

        targets_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # if self.weights is not None:
        #    weights = self.weights.view(1, -1, 1, 1) # reshape for broadcasting
        #    probs = probs * weights
        #    targets_one_hot = targets_one_hot * weights

        dims = (2, 3)
        intersection = (probs * targets_one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        # remove bg class
        dice_score = dice_score[:, 1:]
        loss = 1.0 - dice_score
        
        return loss.mean() if self.reduction == "mean" else loss


class WeightedTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6, weights=None):
        super(WeightedTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.weights = weights

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) - Raw Logits from the model
            target: (B, H, W) - Ground Truth Class Indices
        """
        probs = F.softmax(pred, dim=1)
        num_classes = probs.shape[1]
        
        # Shape: [B, H, W] -> [B, H, W, C] -> [B, C, H, W]
        targets_oh = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        if self.weights is not None:
             # Reshape weights to [1, C, 1, 1] for broadcasting
             w = self.weights.view(1, -1, 1, 1).to(pred.device)
             probs = probs * w
             targets_oh = targets_oh * w

        dims = (0, 2, 3)
        
        TP = (probs * targets_oh).sum(dim=dims)
        FP = (probs * (1 - targets_oh)).sum(dim=dims)
        FN = ((1 - probs) * targets_oh).sum(dim=dims)
        
        # Shape: [C] (One score per class)
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1.0 - tversky_index.mean()

class ComboLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, 
                reduction: str = 'mean', smooth: float = 1e-6, weights: List[float] = [1, 1]):
        super(ComboLoss, self).__init__()
        self.dice = DiceLoss(smooth, reduction)
        self.focal = FocalLoss(alpha, gamma, reduction)
        self.weights = weights
    
    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        focal_loss = self.focal(pred, target)
        return dice_loss*self.weights[0] + focal_loss*self.weights[1]#, dice_loss, focal_loss


class FocalLossExclude(nn.Module):
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        mask_path_pattern: Optional[str] = None,
        mask_exclude_indices: Optional[List[int]] = None
    ):
        super(FocalLossExclude, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.mask_path_pattern = mask_path_pattern
        self.mask_exclude_indices = mask_exclude_indices

        if (mask_path_pattern is not None) != (mask_exclude_indices is not None):
            raise ValueError("Both mask_path_pattern and mask_exclude_indices must be provided, or neither.")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, paths: Optional[List[str]] = None) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        modulating_factor = (1 - p_t).pow(self.gamma)

        pixel_loss = modulating_factor * ce_loss

        valid_pixel_mask = torch.ones_like(pixel_loss, device=logits.device)

        if self.mask_path_pattern is not None and paths is not None:
            pred_labels = torch.argmax(logits, dim=1)

            if self.mask_path_pattern in paths[0]:
                for i in range(logits.shape[0]):
                    indices_to_ignore = self.mask_exclude_indices + [0] # add 0 class, background/lumen

                    for excluded_class in indices_to_ignore:
                        ignore_locations = (pred_labels[i] == excluded_class)
                        valid_pixel_mask[i][ignore_locations] = 0

        final_pixel_loss = pixel_loss * valid_pixel_mask

        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha.gather(0, targets.flatten()).reshape(targets.shape)
            final_pixel_loss = alpha_t * final_pixel_loss

        if self.reduction == 'mean':
            return final_pixel_loss.sum() / valid_pixel_mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            return final_pixel_loss.sum()
        else:
            return final_pixel_loss