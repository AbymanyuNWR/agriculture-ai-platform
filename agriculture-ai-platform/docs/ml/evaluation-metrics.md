# Model Evaluation Metrics

## Overview

Dokumentasi metrik evaluasi yang digunakan dalam Agriculture AI Platform.

## Classification Metrics

### Accuracy

Accuracy adalah proporsi prediksi yang benar dari total prediksi.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
```

**Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Interpretasi:**
- 0.95 = 95% prediksi benar
- Semakin tinggi, semakin baik
- Tidakcocok untuk dataset tidak seimbang

### Precision

Precision adalah proporsi prediksi positif yang benar.

```python
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred, average='macro')
```

**Formula:**
```
Precision = TP / (TP + FP)
```

**Interpretasi:**
- 0.93 = 93% prediksi positif benar
- Penting untuk mengurangi false positive
- Contoh: Jangan salah diagnosis sehat sebagai sakit

### Recall

Recall adalah proporsi kasus positif yang terdeteksi.

```python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred, average='macro')
```

**Formula:**
```
Recall = TP / (TP + FN)
```

**Interpretasi:**
- 0.94 = 94% kasus positif terdeteksi
- Penting untuk mengurangi false negative
- Contoh: Jangan melewatkan penyakit yang sebenarnya ada

### F1-Score

F1-Score adalah harmonic mean dari precision dan recall.

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred, average='macro')
```

**Formula:**
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Interpretasi:**
- 0.93 = keseimbangan antara precision dan recall
- Metrik utama untuk evaluasi model
- Cocok untuk dataset tidak seimbang

### AUC-ROC

AUC-ROC mengukur kemampuan model membedakan kelas.

```python
from sklearn.metrics import roc_auc_score

auc_roc = roc_auc_score(y_true, y_prob, multi_class='ovr')
```

**Interpretasi:**
- 1.0 = sempurna
- 0.987 = sangat baik
- 0.5 = tidak lebih baik dari random

## Confusion Matrix

Confusion Matrix menunjukkan distribusi prediksi vs actual.

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

**Contoh Output:**

```
              Predicted
              Healthy  Blast  Brown Spot
Actual
Healthy         510      5        5
Blast            10    450       20
Brown Spot        15     20      415
```

## Regression Metrics

### Mean Squared Error (MSE)

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
```

### Mean Absolute Error (MAE)

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
```

### R-Squared (R²)

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
```

## Object Detection Metrics

### Mean Average Precision (mAP)

```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco_gt = COCO(ground_truth_annotations)
coco_dt = coco_gt.loadRes(predictions)

coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
```

### Intersection over Union (IoU)

```python
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    
    return intersection / union
```

## Segmentation Metrics

### Dice Coefficient

```python
def dice_coefficient(pred, target):
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum())
```

### IoU for Segmentation

```python
def iou_segmentation(pred, target):
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    return intersection / union
```

## Model Comparison

### Disease Classifier Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| ResNet-50 | 0.942 | 0.938 | 0.941 | 0.939 | 0.987 |
| EfficientNet-B0 | 0.951 | 0.948 | 0.950 | 0.949 | 0.991 |
| Vision Transformer | 0.958 | 0.955 | 0.957 | 0.956 | 0.993 |

### Inference Time

| Model | CPU (ms) | GPU (ms) | Model Size (MB) |
|-------|----------|----------|-----------------|
| ResNet-50 | 45 | 8 | 98 |
| EfficientNet-B0 | 38 | 6 | 21 |
| Vision Transformer | 62 | 12 | 330 |

## Best Practices

1. **Use Multiple Metrics**: Jangan hanya mengandalkan accuracy
2. **Consider Class Distribution**: Gunakan macro/weighted average untuk dataset tidak seimbang
3. **Report Confidence Intervals**: Laporkan std dev untuk robustness
4. **Compare with Baselines**: Bandingkan dengan model sederhana
5. **Consider Inference Time**: Pertimbangkan latensi untuk production
