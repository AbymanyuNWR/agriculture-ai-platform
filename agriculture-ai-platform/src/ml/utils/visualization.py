import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2

class Visualizer:
    def __init__(self, style: str = 'seaborn-v0_8'):
        plt.style.use(style)
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))
        
    def plot_image_grid(
        self,
        images: List[np.ndarray],
        labels: List[str],
        grid_size: Tuple[int, int] = (2, 5),
        figsize: Tuple[int, int] = (15, 6),
        output_path: str = None
    ):
        """Plot a grid of images"""
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
        
        for i, (image, label) in enumerate(zip(images, labels)):
            row = i // grid_size[1]
            col = i % grid_size[1]
            
            if row < grid_size[0] and col < grid_size[1]:
                axes[row, col].imshow(image)
                axes[row, col].set_title(label, fontsize=10)
                axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Image grid saved to {output_path}")
        
        plt.close()
    
    def plot_class_distribution(
        self,
        class_counts: Dict[str, int],
        figsize: Tuple[int, int] = (10, 6),
        output_path: str = None
    ):
        """Plot class distribution bar chart"""
        fig, ax = plt.subplots(figsize=figsize)
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        bars = ax.bar(classes, counts, color=self.colors[:len(classes)])
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution')
        ax.set_xticklabels(classes, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to {output_path}")
        
        plt.close()
    
    def plot_training_history(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_accs: List[float],
        val_accs: List[float],
        figsize: Tuple[int, int] = (12, 5),
        output_path: str = None
    ):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        epochs = range(1, len(train_losses) + 1)
        
        # Plot losses
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
        ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {output_path}")
        
        plt.close()
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        normalize: bool = True,
        figsize: Tuple[int, int] = (10, 8),
        output_path: str = None
    ):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=figsize)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {output_path}")
        
        plt.close()
    
    def plot_roc_curve(
        self,
        fpr: Dict[str, np.ndarray],
        tpr: Dict[str, np.ndarray],
        roc_auc: Dict[str, float],
        figsize: Tuple[int, int] = (10, 8),
        output_path: str = None
    ):
        """Plot ROC curves"""
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, (class_name, color) in enumerate(zip(fpr.keys(), self.colors)):
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                   label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {output_path}")
        
        plt.close()
    
    def plot_prediction_samples(
        self,
        images: List[np.ndarray],
        true_labels: List[str],
        pred_labels: List[str],
        confidences: List[float],
        num_samples: int = 10,
        figsize: Tuple[int, int] = (15, 6),
        output_path: str = None
    ):
        """Plot sample predictions"""
        fig, axes = plt.subplots(2, num_samples // 2, figsize=figsize)
        
        for i in range(min(num_samples, len(images))):
            row = i // (num_samples // 2)
            col = i % (num_samples // 2)
            
            axes[row, col].imshow(images[i])
            
            # Color based on correctness
            color = 'green' if true_labels[i] == pred_labels[i] else 'red'
            
            axes[row, col].set_title(
                f'True: {true_labels[i]}\n'
                f'Pred: {pred_labels[i]}\n'
                f'Conf: {confidences[i]:.2f}',
                fontsize=8,
                color=color
            )
            axes[row, col].axis('off')
        
        plt.suptitle('Prediction Samples (Green=Correct, Red=Incorrect)')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Prediction samples saved to {output_path}")
        
        plt.close()
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        top_k: int = 20,
        figsize: Tuple[int, int] = (10, 8),
        output_path: str = None
    ):
        """Plot feature importance"""
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_k]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.barh(range(top_k), importances[indices][::-1])
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([feature_names[i] for i in indices][::-1])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_k} Feature Importances')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {output_path}")
        
        plt.close()
    
    def visualize_attention(
        self,
        image: np.ndarray,
        attention_map: np.ndarray,
        figsize: Tuple[int, int] = (12, 5),
        output_path: str = None
    ):
        """Visualize attention map on image"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Original image
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Attention map overlay
        ax2.imshow(image)
        ax2.imshow(attention_map, cmap='jet', alpha=0.5)
        ax2.set_title('Attention Map')
        ax2.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Attention visualization saved to {output_path}")
        
        plt.close()
