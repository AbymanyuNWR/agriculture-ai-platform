from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml

@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    features_dir: str = "data/features"
    image_size: tuple = (224, 224)
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
    
@dataclass
class AugmentationConfig:
    random_flip: bool = True
    random_rotation: int = 45
    random_brightness: float = 0.2
    random_contrast: float = 0.2
    random_saturation: float = 0.2
    random_hue: float = 0.1
    random_crop: bool = True
    random_erasing: bool = True

@dataclass
class ModelConfig:
    name: str = "disease_classifier"
    architecture: str = "resnet50"
    num_classes: int = 10
    pretrained: bool = True
    dropout: float = 0.3
    feature_dim: int = 256
    
@dataclass
class TrainingConfig:
    model_name: str = "disease_classifier"
    dataset_version: str = "v1"
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    optimizer: str = "adam"
    early_stopping: int = 10
    device: str = "cuda:0"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    
@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1_score", "auc_roc"
    ])
    confusion_matrix: bool = True
    classification_report: bool = True
    roc_curve: bool = True
    precision_recall_curve: bool = True
    
@dataclass
class MLflowConfig:
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "agriculture_ai"
    artifact_location: str = "mlruns"
    log_model: bool = True
    log_params: bool = True
    log_metrics: bool = True
    
@dataclass
class DataCollectionConfig:
    image_sources: List[str] = field(default_factory=lambda: [
        "plantvillage", "custom", "openimages"
    ])
    metadata_format: str = "csv"
    validation_split: float = 0.2
    min_images_per_class: int = 100
    max_images_per_class: int = 10000
    
@dataclass
class FeatureEngineeringConfig:
    image_size: tuple = (224, 224)
    normalize: bool = True
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    feature_extraction: bool = False
    pca_components: Optional[int] = None
    
@dataclass
class MonitoringConfig:
    enabled: bool = True
    drift_detection: bool = True
    performance_monitoring: bool = True
    alert_threshold: float = 0.1
    check_interval_minutes: int = 60
    
@dataclass
class MLConfig:
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    data_collection: DataCollectionConfig = field(default_factory=DataCollectionConfig)
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'MLConfig':
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

ml_config = MLConfig()
