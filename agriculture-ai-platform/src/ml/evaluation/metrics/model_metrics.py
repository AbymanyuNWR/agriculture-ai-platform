import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ModelMetrics:
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or [
            'healthy', 'blast', 'brown_spot', 'leaf_blight',
            'bacterial_blight', 'tungro', 'grassy_stunt',
            'ragged_stunt', 'rice_grassy_virus', 'rice_tungro_bacilliform'
        ]
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict:
        """Calculate all classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }
        
        # Per-class metrics
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['per_class'] = {}
        for i, class_name in enumerate(self.class_names):
            if i < len(per_class_precision):
                metrics['per_class'][class_name] = {
                    'precision': per_class_precision[i],
                    'recall': per_class_recall[i],
                    'f1_score': per_class_f1[i]
                }
        
        # AUC-ROC if probabilities provided
        if y_prob is not None:
            try:
                # One-vs-rest AUC
                from sklearn.preprocessing import label_binarize
                y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
                
                if y_true_bin.shape[1] == 1:
                    # Binary case
                    metrics['auc_roc'] = auc(*roc_curve(y_true, y_prob[:, 1])[:2])
                else:
                    # Multi-class
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    
                    for i in range(len(self.class_names)):
                        if i < y_prob.shape[1]:
                            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                            roc_auc[i] = auc(fpr[i], tpr[i])
                    
                    metrics['auc_roc'] = np.mean(list(roc_auc.values()))
                    metrics['auc_roc_per_class'] = roc_auc
            except Exception as e:
                print(f"Error calculating AUC-ROC: {e}")
                metrics['auc_roc'] = 0.0
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: str = None,
        normalize: bool = True
    ) -> np.ndarray:
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {output_path}")
        
        plt.close()
        return cm
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        output_path: str = None
    ):
        """Plot ROC curve"""
        from sklearn.preprocessing import label_binarize
        
        y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
        
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.class_names)))
        
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            if i < y_prob.shape[1]:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, lw=2,
                        label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right", bbox_to_anchor=(1.3, 0))
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {output_path}")
        
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        output_path: str = None
    ):
        """Plot precision-recall curve"""
        from sklearn.preprocessing import label_binarize
        
        y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
        
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.class_names)))
        
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            if i < y_prob.shape[1]:
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                avg_precision = average_precision_score(y_true_bin[:, i], y_prob[:, i])
                plt.plot(recall, precision, color=color, lw=2,
                        label=f'{class_name} (AP = {avg_precision:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower right", bbox_to_anchor=(1.3, 0))
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Precision-recall curve saved to {output_path}")
        
        plt.close()
    
    def plot_metrics_comparison(self, metrics: Dict, output_path: str = None):
        """Plot metrics comparison bar chart"""
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score']
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metric_names, metric_values, color=['#2196F3', '#4CAF50', '#FFC107', '#F44336'])
        
        plt.ylim(0, 1.0)
        plt.ylabel('Score')
        plt.title('Model Performance Metrics')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison saved to {output_path}")
        
        plt.close()
    
    def plot_per_class_metrics(self, metrics: Dict, output_path: str = None):
        """Plot per-class metrics heatmap"""
        if 'per_class' not in metrics:
            return
        
        class_names = list(metrics['per_class'].keys())
        metric_types = ['precision', 'recall', 'f1_score']
        
        data = []
        for class_name in class_names:
            row = [metrics['per_class'][class_name][m] for m in metric_types]
            data.append(row)
        
        data = np.array(data)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(data, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=metric_types, yticklabels=class_names)
        plt.title('Per-Class Metrics')
        plt.xlabel('Metric')
        plt.ylabel('Class')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Per-class metrics saved to {output_path}")
        
        plt.close()
    
    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: str = None
    ) -> str:
        """Generate classification report"""
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4
        )
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Classification report saved to {output_path}")
        
        return report
    
    def generate_full_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray = None,
        output_dir: str = "reports"
    ) -> Dict:
        """Generate complete evaluation report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        
        # Generate plots
        self.plot_confusion_matrix(y_true, y_pred, output_path / "confusion_matrix.png")
        self.plot_metrics_comparison(metrics, output_path / "metrics_comparison.png")
        self.plot_per_class_metrics(metrics, output_path / "per_class_metrics.png")
        
        if y_prob is not None:
            self.plot_roc_curve(y_true, y_prob, output_path / "roc_curve.png")
            self.plot_precision_recall_curve(y_true, y_prob, output_path / "precision_recall_curve.png")
        
        # Generate classification report
        report = self.generate_classification_report(y_true, y_pred, output_path / "classification_report.txt")
        
        # Save metrics
        import json
        with open(output_path / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
