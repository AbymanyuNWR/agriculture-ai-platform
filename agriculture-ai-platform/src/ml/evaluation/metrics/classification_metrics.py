import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, log_loss, cohen_kappa_score,
    matthews_corrcoef, balanced_accuracy_score
)

@dataclass
class ClassificationMetrics:
    """Comprehensive classification metrics"""
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    f1_macro: float
    f1_micro: float
    f1_weighted: float
    auc_roc: float
    auc_pr: float
    log_loss: float
    cohen_kappa: float
    matthews_corrcoef: float
    top_k_accuracy: float
    confusion_matrix: np.ndarray
    classification_report: str

class ClassificationEvaluator:
    """Complete classification evaluation"""
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        top_k: int = 5
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.top_k = top_k
        
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> ClassificationMetrics:
        """Compute all classification metrics"""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # AUC-ROC
        if y_prob is not None and self.num_classes == 2:
            auc_roc = roc_auc_score(y_true, y_prob[:, 1])
            auc_pr = average_precision_score(y_true, y_prob[:, 1])
            logloss = log_loss(y_true, y_prob)
        elif y_prob is not None and self.num_classes > 2:
            auc_roc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            auc_pr = 0.0
            logloss = log_loss(y_true, y_prob)
        else:
            auc_roc = 0.0
            auc_pr = 0.0
            logloss = 0.0
            
        # Other metrics
        cohen_kap = cohen_kappa_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # Top-k accuracy
        if y_prob is not None:
            top_k_acc = self._top_k_accuracy(y_true, y_prob, self.top_k)
        else:
            top_k_acc = 0.0
            
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            zero_division=0
        )
        
        return ClassificationMetrics(
            accuracy=accuracy,
            balanced_accuracy=balanced_acc,
            precision=precision_macro,
            recall=recall_macro,
            f1=precision_macro,
            f1_macro=f1_macro,
            f1_micro=f1_micro,
            f1_weighted=f1_weighted,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            log_loss=logloss,
            cohen_kappa=cohen_kap,
            matthews_corrcoef=mcc,
            top_k_accuracy=top_k_acc,
            confusion_matrix=cm,
            classification_report=report
        )
        
    def _top_k_accuracy(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        k: int
    ) -> float:
        """Compute top-k accuracy"""
        top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
        
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                correct += 1
                
        return correct / len(y_true)
        
    def compute_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-class metrics"""
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        
        per_class = {}
        for i, class_name in enumerate(self.class_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class[class_name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(cm[i, :].sum()),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn)
            }
            
        return per_class
        
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = True
    ):
        """Plot confusion matrix"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        
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
        ax.set_title('Confusion Matrix')
        
        return fig
        
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ):
        """Plot ROC curve"""
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if self.num_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        else:
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            
            for i in range(self.num_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {roc_auc:.3f})')
                
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        
        return fig
