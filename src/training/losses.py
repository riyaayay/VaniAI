"""
Loss Functions for IndicGuard - ENHANCED
Includes EOC-Softmax + Focal Loss for hard examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples.
    
    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes)
            labels: (batch,)
        """
        probs = F.softmax(logits, dim=-1)
        target_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1 - target_probs) ** self.gamma
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        
        focal_loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            alpha_weight = torch.ones_like(labels, dtype=torch.float)
            alpha_weight[labels == 1] = self.alpha
            alpha_weight[labels == 0] = 1 - self.alpha
            focal_loss = alpha_weight * focal_loss
        
        return focal_loss.mean()


class EOCSoftmaxLoss(nn.Module):
    """
    Enhanced Extended One-Class Softmax (EOC-S) Loss
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        feat_dim: int = 256,
        margin: float = 0.4,
        scale: float = 35.0,
        human_class_idx: int = 0,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.margin = margin
        self.scale = scale
        self.human_class_idx = human_class_idx
        
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)
        
        logger.info(f"Enhanced EOC-S Loss: margin={margin}, scale={scale}")
    
    def forward(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        batch_size = features.size(0)
        
        features_norm = F.normalize(features, p=2, dim=1)
        centers_norm = F.normalize(self.centers, p=2, dim=1)
        
        # Force centers to the same device as the features (the GPU)
        distances = torch.cdist(features_norm.unsqueeze(0), self.centers.to(features.device).unsqueeze(0)).squeeze(0)
        similarities = -distances
        
        target_similarities = similarities.clone()
        mask = F.one_hot(labels, num_classes=self.num_classes).bool()
        target_similarities[mask] -= self.margin
        
        scaled_similarities = target_similarities * self.scale
        softmax_loss = F.cross_entropy(scaled_similarities, labels)
        
        center_diff = features_norm - centers_norm[labels]
        center_loss = (center_diff ** 2).sum(dim=1).mean()
        
        human_mask = (labels == self.human_class_idx)
        if human_mask.sum() > 0:
            human_features = features_norm[human_mask]
            human_center = centers_norm[self.human_class_idx]
            compactness_loss = ((human_features - human_center) ** 2).sum(dim=1).mean()
        else:
            compactness_loss = torch.tensor(0.0, device=features.device)
        
        ce_loss = F.cross_entropy(logits, labels)
        
        total_loss = ce_loss + 0.1 * softmax_loss + 0.015 * center_loss + 0.05 * compactness_loss
        
        metrics = {
            'ce_loss': ce_loss.item(),
            'softmax_loss': softmax_loss.item(),
            'center_loss': center_loss.item(),
            'compactness_loss': compactness_loss.item(),
        }
        
        return total_loss, metrics


class LabelSmoothingCrossEntropy(nn.Module):
    """Enhanced Cross-Entropy with Label Smoothing"""
    
    def __init__(self, smoothing: float = 0.15, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        num_classes = logits.size(-1)
        
        with torch.no_grad():
            smooth_labels = torch.zeros_like(logits)
            smooth_labels.fill_(self.smoothing / (num_classes - 1))
            smooth_labels.scatter_(1, labels.unsqueeze(1), 1 - self.smoothing)
        
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_labels * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    ENHANCED Combined loss with Focal Loss support.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        loss_config = config["training"]["loss"]
        training_config = config["training"]
        
        self.label_smoothing = training_config.get("label_smoothing", 0.15)
        self.center_loss_weight = loss_config.get("center_loss_weight", 0.015)
        self.margin = loss_config.get("margin", 0.4)
        self.scale = loss_config.get("scale", 35.0)
        
        # Focal loss configuration
        self.use_focal = loss_config.get("use_focal", True)
        self.focal_gamma = loss_config.get("focal_gamma", 2.0)
        
        # Label-smoothed CE
        self.ce_loss = LabelSmoothingCrossEntropy(
            smoothing=self.label_smoothing
        )
        
        # Focal loss
        if self.use_focal:
            self.focal_loss = FocalLoss(gamma=self.focal_gamma, alpha=0.25)
        
        # EOC-S loss
        self.eoc_loss = EOCSoftmaxLoss(
            num_classes=2,
            feat_dim=config["model"]["liquid"]["hidden_dim"],
            margin=self.margin,
            scale=self.scale,
        )
        
        logger.info(f"Enhanced Combined Loss:")
        logger.info(f"  Label Smoothing: {self.label_smoothing}")
        logger.info(f"  EOC-S Margin: {self.margin}")
        logger.info(f"  Focal Loss: {'Enabled' if self.use_focal else 'Disabled'}")
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        # Base CE loss
        ce_loss = self.ce_loss(logits, labels)
        
        metrics = {
            'ce_loss': ce_loss.item(),
        }
        
        total_loss = ce_loss
        
        # Add focal loss if enabled
        if self.use_focal:
            focal = self.focal_loss(logits, labels)
            total_loss = total_loss + 0.5 * focal
            metrics['focal_loss'] = focal.item()
        
        # EOC-S loss if features provided
        if features is not None:
            eoc_loss, eoc_metrics = self.eoc_loss(features, logits, labels)
            total_loss = total_loss + eoc_loss
            metrics.update(eoc_metrics)
            metrics['eoc_loss'] = (eoc_loss - ce_loss).item()
        
        metrics['total_loss'] = total_loss.item()
        
        return total_loss, metrics


def mixup_criterion(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute loss for mixup-augmented data."""
    if isinstance(criterion, CombinedLoss):
        loss_a, _ = criterion(logits, labels_a, features=None)
        loss_b, _ = criterion(logits, labels_b, features=None)
    else:
        loss_a = criterion(logits, labels_a)
        loss_b = criterion(logits, labels_b)
    
    return lam * loss_a + (1 - lam) * loss_b