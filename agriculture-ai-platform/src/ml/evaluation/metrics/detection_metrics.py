import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class DetectionMetrics:
    """Detection metrics (mAP, IoU)"""
    mAP: float
    mAP_50: float
    mAP_75: float
    AP_per_class: Dict[str, float]
    mean_iou: float
    iou_per_class: Dict[str, float]
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int

class DetectionEvaluator:
    """Object detection evaluation (mAP, IoU)"""
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.5
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        
    def compute_iou(
        self,
        box1: np.ndarray,
        box2: np.ndarray
    ) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
        
    def compute_iou_batch(
        self,
        boxes1: np.ndarray,
        boxes2: np.ndarray
    ) -> np.ndarray:
        """Compute IoU for batch of boxes"""
        x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0:1].T)
        y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1:2].T)
        x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2:3].T)
        y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3:4].T)
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection
        
        return intersection / np.maximum(union, 1e-6)
        
    def compute_ap(
        self,
        y_true: List[np.ndarray],
        y_pred: List[np.ndarray],
        iou_threshold: float = 0.5
    ) -> float:
        """Compute Average Precision"""
        # Sort predictions by confidence
        all_predictions = []
        for img_idx, preds in enumerate(y_pred):
            for pred in preds:
                all_predictions.append({
                    'image_id': img_idx,
                    'bbox': pred[:4],
                    'confidence': pred[4],
                    'class_id': pred[5] if len(pred) > 5 else 0
                })
                
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Compute precision-recall curve
        tp = np.zeros(len(all_predictions))
        fp = np.zeros(len(all_predictions))
        
        for pred_idx, pred in enumerate(all_predictions):
            img_idx = pred['image_id']
            pred_bbox = pred['bbox']
            
            if img_idx < len(y_true) and len(y_true[img_idx]) > 0:
                # Find best matching ground truth
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_bbox in enumerate(y_true[img_idx]):
                    iou = self.compute_iou(pred_bbox, gt_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                        
                if best_iou >= iou_threshold:
                    tp[pred_idx] = 1
                else:
                    fp[pred_idx] = 1
            else:
                fp[pred_idx] = 1
                
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / len(y_true) if y_true else np.zeros_like(precision)
        
        # Compute AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            precisions_at_recall = precision[recall >= t]
            if len(precisions_at_recall) > 0:
                ap += np.max(precisions_at_recall) / 11
                
        return ap
        
    def compute_map(
        self,
        y_true: List[List[np.ndarray]],
        y_pred: List[List[np.ndarray]],
        iou_threshold: float = 0.5
    ) -> float:
        """Compute mean Average Precision"""
        aps = []
        
        for class_id in range(self.num_classes):
            # Filter by class
            class_true = [
                [box for box in img_boxes if box[5] == class_id]
                for img_boxes in y_true
            ]
            class_pred = [
                [box for box in img_boxes if box[5] == class_id]
                for img_boxes in y_pred
            ]
            
            ap = self.compute_ap(class_true, class_pred, iou_threshold)
            aps.append(ap)
            
        return np.mean(aps)
        
    def compute_map_at_thresholds(
        self,
        y_true: List[List[np.ndarray]],
        y_pred: List[List[np.ndarray]]
    ) -> Dict[str, float]:
        """Compute mAP at different IoU thresholds"""
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
        results = {}
        for thresh in thresholds:
            results[f"mAP@{thresh}"] = self.compute_map(y_true, y_pred, thresh)
            
        results["mAP"] = np.mean(list(results.values()))
        results["mAP_50"] = results.get("mAP@0.5", 0)
        results["mAP_75"] = results.get("mAP@0.75", 0)
        
        return results
        
    def compute_per_class_ap(
        self,
        y_true: List[List[np.ndarray]],
        y_pred: List[List[np.ndarray]]
    ) -> Dict[str, float]:
        """Compute per-class AP"""
        per_class_ap = {}
        
        for class_id, class_name in enumerate(self.class_names):
            class_true = [
                [box for box in img_boxes if box[5] == class_id]
                for img_boxes in y_true
            ]
            class_pred = [
                [box for box in img_boxes if box[5] == class_id]
                for img_boxes in y_pred
            ]
            
            ap = self.compute_ap(class_true, class_pred, self.iou_threshold)
            per_class_ap[class_name] = ap
            
        return per_class_ap
        
    def compute_metrics(
        self,
        y_true: List[List[np.ndarray]],
        y_pred: List[List[np.ndarray]]
    ) -> DetectionMetrics:
        """Compute all detection metrics"""
        # mAP
        map_results = self.compute_map_at_thresholds(y_true, y_pred)
        
        # Per-class AP
        per_class_ap = self.compute_per_class_ap(y_true, y_pred)
        
        # Compute TP, FP, FN
        all_tp = 0
        all_fp = 0
        all_fn = 0
        
        for img_true, img_pred in zip(y_true, y_pred):
            matched = set()
            
            for pred in img_pred:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(img_true):
                    iou = self.compute_iou(pred[:4], gt[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                        
                if best_iou >= self.iou_threshold and best_gt_idx not in matched:
                    all_tp += 1
                    matched.add(best_gt_idx)
                else:
                    all_fp += 1
                    
            all_fn += len(img_true) - len(matched)
            
        # Compute precision, recall, F1
        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return DetectionMetrics(
            mAP=map_results["mAP"],
            mAP_50=map_results["mAP_50"],
            mAP_75=map_results["mAP_75"],
            AP_per_class=per_class_ap,
            mean_iou=0.0,
            iou_per_class={},
            precision=precision,
            recall=recall,
            f1=f1,
            true_positives=all_tp,
            false_positives=all_fp,
            false_negatives=all_fn
        )
        
    def plot_precision_recall_curve(
        self,
        y_true: List[List[np.ndarray]],
        y_pred: List[List[np.ndarray]]
    ):
        """Plot precision-recall curve"""
        import matplotlib.pyplot as plt
        
        # Compute precision-recall for each class
        for class_id, class_name in enumerate(self.class_names):
            # Get predictions for this class
            all_scores = []
            all_labels = []
            
            for img_idx, (img_true, img_pred) in enumerate(zip(y_true, y_pred)):
                for pred in img_pred:
                    if pred[5] == class_id:
                        # Check if correct
                        is_correct = False
                        for gt in img_true:
                            if gt[5] == class_id:
                                iou = self.compute_iou(pred[:4], gt[:4])
                                if iou >= self.iou_threshold:
                                    is_correct = True
                                    break
                                    
                        all_scores.append(pred[4])
                        all_labels.append(1 if is_correct else 0)
                        
            if all_scores:
                from sklearn.metrics import precision_recall_curve
                precision, recall, _ = precision_recall_curve(all_labels, all_scores)
                plt.plot(recall, precision, label=class_name)
                
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        
        return plt.gcf()
