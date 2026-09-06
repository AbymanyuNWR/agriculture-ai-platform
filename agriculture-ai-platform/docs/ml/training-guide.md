# Model Training Guide

## Overview

Panduan ini menjelaskan cara melatih model machine learning untuk Agriculture AI Platform.

## Prerequisites

- Python 3.9+
- CUDA 11.8+ (untuk GPU training)
- Minimum 16GB RAM
- Storage: 50GB minimum
- MLflow server running

## Training Pipeline

### 1. Data Preparation

#### Struktur Dataset

```
data/
├── raw/
│   ├── images/
│   │   ├── healthy/
│   │   │   ├── img001.jpg
│   │   │   ├── img002.jpg
│   │   │   └── ...
│   │   ├── blast/
│   │   ├── brown_spot/
│   │   └── ...
│   └── metadata.csv
├── processed/
│   ├── train/
│   ├── val/
│   └── test/
└── features/
```

#### Format Metadata CSV

```csv
image_path,label,crop_type,confidence,source
images/healthy/img001.jpg,healthy,padi,1.0,dataset_v1
images/blast/img002.jpg,blast,padi,1.0,dataset_v1
```

### 2. Data Validation

```python
from src.ml.data.validators.data_validator import DataValidator

validator = DataValidator()
results = validator.validate_dataset("data/raw/")

print(f"Total images: {results['total_images']}")
print(f"Class distribution: {results['class_distribution']}")
print(f"Image quality score: {results['quality_score']}")
```

### 3. Feature Engineering

```python
from src.ml.features.engineering.feature_engineer import FeatureEngineer

engineer = FeatureEngineer()

# Process images
features = engineer.process_images(
    image_dir="data/raw/images/",
    output_dir="data/features/",
    input_size=(224, 224)
)

# Generate metadata
engineer.generate_metadata(
    features_dir="data/features/",
    output_path="data/processed/metadata.csv"
)
```

### 4. Model Training

#### Single GPU Training

```bash
# Train disease classifier
python -m src.ml.pipelines.training_pipeline \
    --model-type disease_classifier \
    --dataset-version v1 \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --device cuda:0
```

#### Multi-GPU Training

```bash
# Train with multiple GPUs
python -m src.ml.pipelines.training_pipeline \
    --model-type disease_classifier \
    --dataset-version v1 \
    --epochs 50 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --distributed \
    --num-gpus 4
```

#### Training Configuration

```yaml
# configs/training/disease_classifier.yaml
model:
  name: disease_classifier
  architecture: resnet50
  num_classes: 10
  pretrained: true

data:
  train_dir: data/processed/train/
  val_dir: data/processed/val/
  test_dir: data/processed/test/
  image_size: [224, 224]
  augment: true

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam
  scheduler: cosine
  early_stopping: 10
  
augmentation:
  random_flip: true
  random_rotation: 45
  random_brightness: 0.2
  random_contrast: 0.2
  
logging:
  mlflow: true
  tensorboard: true
  interval: 100
```

### 5. Model Evaluation

```bash
# Evaluate trained model
python -m src.ml.evaluation.evaluate_model \
    --model-path data/models/disease_classifier.pth \
    --test-dir data/processed/test/ \
    --output-dir reports/
```

#### Evaluation Metrics

```python
from src.ml.evaluation.metrics.model_metrics import ModelMetrics

metrics = ModelMetrics()

# Calculate metrics
results = metrics.calculate_metrics(
    y_true=ground_truth,
    y_pred=predictions,
    y_prob=probabilities
)

# Print results
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall: {results['recall']:.4f}")
print(f"F1-Score: {results['f1_score']:.4f}")
print(f"AUC-ROC: {results['auc_roc']:.4f}")

# Confusion matrix
metrics.plot_confusion_matrix(
    y_true=ground_truth,
    y_pred=predictions,
    class_names=class_names,
    output_path="reports/confusion_matrix.png"
)
```

### 6. Model Registration

```bash
# Register model in MLflow
python -m src.ml.registry.register_model \
    --model-path data/models/disease_classifier.pth \
    --model-name disease_classifier \
    --version 1.0.0 \
    --metrics reports/metrics.json \
    --tags "dataset=v1,architecture=resnet50"
```

## Hyperparameter Tuning

### Using Optuna

```python
import optuna
from src.ml.training.trainers.model_trainer import ModelTrainer

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    
    # Train model
    trainer = ModelTrainer(
        learning_rate=lr,
        batch_size=batch_size,
        dropout=dropout
    )
    
    metrics = trainer.train()
    
    return metrics["f1_score"]

# Create study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Best parameters
print(study.best_params)
```

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'dropout': [0.2, 0.3, 0.4]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

## Model Optimization

### Quantization

```python
import torch.quantization as quantization

# Load model
model = DiseaseClassifier(num_classes=10)
model.load_state_dict(torch.load("model.pth"))

# Quantize
quantized_model = quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), "model_quantized.pth")
```

### ONNX Export

```python
import torch.onnx

# Load model
model = DiseaseClassifier(num_classes=10)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
--batch-size 16

# Use gradient accumulation
--gradient-accumulation-steps 4

# Enable mixed precision
--mixed-precision
```

### Slow Training

```bash
# Enable DataLoader workers
--num-workers 8

# Use pin memory
--pin-memory

# Enable cudnn benchmark
--cudnn-benchmark
```

### Poor Performance

1. Check data quality
2. Increase model capacity
3. Add more data augmentation
4. Try different learning rate schedules
5. Use ensemble methods

## Best Practices

1. **Version Control**: Always version your datasets and models
2. **Experiment Tracking**: Use MLflow to track all experiments
3. **Reproducibility**: Set random seeds for reproducibility
4. **Validation**: Always use a held-out test set
5. **Monitoring**: Monitor training curves for overfitting
6. **Documentation**: Document all experiments and results
