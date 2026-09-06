import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Cross Entropy Loss"""
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smoothed labels
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / self.num_classes)
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / self.num_classes)
        
        loss = (-smooth_targets * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class MixupLoss(nn.Module):
    """Loss for Mixup augmentation"""
    
    def __init__(self, loss_fn: nn.Module = None):
        super().__init__()
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        
    def forward(
        self,
        outputs: torch.Tensor,
        targets_a: torch.Tensor,
        targets_b: torch.Tensor,
        lam: float
    ) -> torch.Tensor:
        return lam * self.loss_fn(outputs, targets_a) + (1 - lam) * self.loss_fn(outputs, targets_b)


class CenterLoss(nn.Module):
    """Center Loss for face recognition"""
    
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)
        
        # Expand centers to match batch
        centers_batch = self.centers[labels]
        
        # Calculate distance
        diff = features - centers_batch
        loss = diff.pow(2).sum() / batch_size
        
        return loss


class TripletLoss(nn.Module):
    """Triplet Loss for metric learning"""
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        pos_dist = (anchor - positive).pow(2).sum(dim=1)
        neg_dist = (anchor - negative).pow(2).sum(dim=1)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(features, features.t()) / self.temperature
        
        # Create mask for positive pairs
        labels = labels.unsqueeze(0)
        mask = torch.eq(labels, labels.t()).float()
        
        # Remove diagonal
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=mask.device)
        mask = mask * logits_mask
        
        # Compute log softmax
        exp_sim = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Compute mean log prob for positive pairs
        pos_pairs = mask.sum(dim=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / (pos_pairs + 1e-6)
        
        loss = -mean_log_prob.mean()
        return loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class LovaszLoss(nn.Module):
    """Lovasz-Softmax Loss"""
    
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Simplified Lovasz loss
        num_classes = inputs.size(1)
        
        # Compute probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Compute per-pixel loss
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            pred_c = probs[:, c]
            
            # Compute IoU
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection
            
            loss = 1 - (intersection + 1) / (union + 1)
            losses.append(loss)
        
        return torch.stack(losses).mean()


class CombinedLoss(nn.Module):
    """Combined Loss"""
    
    def __init__(
        self,
        losses: List[nn.Module],
        weights: List[float] = None
    ):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights or [1.0] * len(losses)
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(inputs, targets)
        
        return total_loss


class ArcFaceLoss(nn.Module):
    """ArcFace Loss for face recognition"""
    
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 30.0
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = torch.mm(embeddings, weights.t())
        
        # Get angle
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        
        # Add margin to target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        
        target_logits = torch.cos(theta + self.margin * one_hot)
        logits = target_logits * self.scale
        
        loss = F.cross_entropy(logits, labels)
        return loss


class CBLoss(nn.Module):
    """Class-Balanced Loss"""
    
    def __init__(
        self,
        num_samples_per_class: List[int],
        loss_type: str = 'focal',
        beta: float = 0.9999,
        gamma: float = 2.0
    ):
        super().__init__()
        self.loss_type = loss_type
        self.gamma = gamma
        
        # Calculate effective number of samples
        effective_num = 1.0 - np.power(beta, num_samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / weights.sum() * len(num_samples_per_class)
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = self.weights.to(inputs.device)
        
        if self.loss_type == 'focal':
            focal_loss = FocalLoss(alpha=weights, gamma=self.gamma)
            return focal_loss(inputs, targets)
        else:
            ce_loss = F.cross_entropy(inputs, targets, weight=weights)
            return ce_loss
