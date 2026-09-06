import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class SegmentationMetrics:
    """Segmentation metrics"""
    iou: float
    dice: float
    pixel_accuracy: float
    mean_pixel_accuracy: float
    mean_iou: float
    mean_dice: float
    per_class_iou: Dict[str, float]
    per_class_dice: Dict[str, float]
    confusion_matrix: np.ndarray

class SegmentationEvaluator:
    """Semantic segmentation evaluation"""
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        
    def compute_iou(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        class_id: int
    ) -> float:
        """Compute IoU for a single class"""
        pred_mask = pred == class_id
        target_mask = target == class_id
        
        intersection = np.sum(pred_mask & target_mask)
        union = np.sum(pred_mask | target_mask)
        
        return intersection / union if union > 0 else 0
        
    def compute_dice(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        class_id: int
    ) -> float:
        """Compute Dice coefficient for a single class"""
        pred_mask = pred == class_id
        target_mask = target == class_id
        
        intersection = np.sum(pred_mask & target_mask)
        total = np.sum(pred_mask) + np.sum(target_mask)
        
        return 2 * intersection / total if total > 0 else 0
        
    def compute_pixel_accuracy(
        self,
        pred: np.ndarray,
        target: np.ndarray
    ) -> float:
        """Compute pixel accuracy"""
        return np.mean(pred == target)
        
    def compute_mean_pixel_accuracy(
        self,
        pred: np.ndarray,
        target: np.ndarray
    ) -> float:
        """Compute mean pixel accuracy (per-class)"""
        accuracies = []
        
        for class_id in range(self.num_classes):
            pred_mask = pred == class_id
            target_mask = target == class_id
            
            if np.sum(target_mask) > 0:
                accuracy = np.sum(pred_mask & target_mask) / np.sum(target_mask)
                accuracies.append(accuracy)
                
        return np.mean(accuracies) if accuracies else 0
        
    def compute_confusion_matrix(
        self,
        pred: np.ndarray,
        target: np.ndarray
    ) -> np.ndarray:
        """Compute confusion matrix"""
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        
        for true_class in range(self.num_classes):
            for pred_class in range(self.num_classes):
                cm[true_class, pred_class] = np.sum(
                    (target == true_class) & (pred == pred_class)
                )
                
        return cm
        
    def compute_per_class_metrics(
        self,
        pred: np.ndarray,
        target: np.ndarray
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute per-class IoU and Dice"""
        per_class_iou = {}
        per_class_dice = {}
        
        for class_id, class_name in enumerate(self.class_names):
            per_class_iou[class_name] = self.compute_iou(pred, target, class_id)
            per_class_dice[class_name] = self.compute_dice(pred, target, class_id)
            
        return per_class_iou, per_class_dice
        
    def compute_metrics(
        self,
        pred: np.ndarray,
        target: np.ndarray
    ) -> SegmentationMetrics:
        """Compute all segmentation metrics"""
        # Overall metrics
        pixel_acc = self.compute_pixel_accuracy(pred, target)
        mean_pixel_acc = self.compute_mean_pixel_accuracy(pred, target)
        
        # Per-class metrics
        per_class_iou, per_class_dice = self.compute_per_class_metrics(pred, target)
        
        # Mean metrics
        mean_iou = np.mean(list(per_class_iou.values()))
        mean_dice = np.mean(list(per_class_dice.values()))
        
        # Confusion matrix
        cm = self.compute_confusion_matrix(pred, target)
        
        return SegmentationMetrics(
            iou=mean_iou,
            dice=mean_dice,
            pixel_accuracy=pixel_acc,
            mean_pixel_accuracy=mean_pixel_acc,
            mean_iou=mean_iou,
            mean_dice=mean_dice,
            per_class_iou=per_class_iou,
            per_class_dice=per_class_dice,
            confusion_matrix=cm
        )
        
    def compute_batch_metrics(
        self,
        preds: np.ndarray,
        targets: np.ndarray
    ) -> SegmentationMetrics:
        """Compute metrics for a batch"""
        batch_metrics = []
        
        for i in range(preds.shape[0]):
            metrics = self.compute_metrics(preds[i], targets[i])
            batch_metrics.append(metrics)
            
        # Average across batch
        avg_per_class_iou = {}
        avg_per_class_dice = {}
        
        for class_name in self.class_names:
            avg_per_class_iou[class_name] = np.mean([
                m.per_class_iou[class_name] for m in batch_metrics
            ])
            avg_per_class_dice[class_name] = np.mean([
                m.per_class_dice[class_name] for m in batch_metrics
            ])
            
        return SegmentationMetrics(
            iou=np.mean([m.iou for m in batch_metrics]),
            dice=np.mean([m.dice for m in batch_metrics]),
            pixel_accuracy=np.mean([m.pixel_accuracy for m in batch_metrics]),
            mean_pixel_accuracy=np.mean([m.mean_pixel_accuracy for m in batch_metrics]),
            mean_iou=np.mean([m.mean_iou for m in batch_metrics]),
            mean_dice=np.mean([m.mean_dice for m in batch_metrics]),
            per_class_iou=avg_per_class_iou,
            per_class_dice=avg_per_class_dice,
            confusion_matrix=np.mean([m.confusion_matrix for m in batch_metrics], axis=0).astype(int)
        )
        
    def plot_confusion_matrix(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        normalize: bool = True
    ):
        """Plot confusion matrix"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = self.compute_confusion_matrix(pred, target)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='.2f' if normalize else 'd',
            cmap='Blues', xticklabels=self.class_names,
            yticklabels=self.class_names, ax=ax
        )
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Segmentation Confusion Matrix')
        
        return fig
        
    def visualize_predictions(
        self,
        image: np.ndarray,
        pred: np.ndarray,
        target: np.ndarray,
        num_classes: int = None
    ):
        """Visualize predictions"""
        import matplotlib.pyplot as plt
        
        if num_classes is None:
            num_classes = self.num_classes
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image.transpose(1, 2, 0) if image.shape[0] == 3 else image)
        axes[0].set_title('Image')
        axes[0].axis('off')
        
        axes[1].imshow(pred, cmap='tab20', vmin=0, vmax=num_classes-1)
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        axes[2].imshow(target, cmap='tab20', vmin=0, vmax=num_classes-1)
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        return fig
