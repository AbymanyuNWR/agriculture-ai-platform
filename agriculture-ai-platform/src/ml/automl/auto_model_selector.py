import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import cross_val_score
import optuna
from enum import Enum
import time

class ModelType(Enum):
    """Available model types"""
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B1 = "efficientnet_b1"
    EFFICIENTNET_B2 = "efficientnet_b2"
    EFFICIENTNET_B3 = "efficientnet_b3"
    EFFICIENTNET_B4 = "efficientnet_b4"
    EFFICIENTNET_B5 = "efficientnet_b5"
    VIT_SMALL = "vit_small"
    VIT_BASE = "vit_base"
    VIT_LARGE = "vit_large"
    SWIN_TINY = "swin_tiny"
    SWIN_SMALL = "swin_small"
    SWIN_BASE = "swin_base"
    CONVNEXT_TINY = "convnext_tiny"
    CONVNEXT_SMALL = "convnext_small"
    CONVNEXT_BASE = "convnext_base"
    RESNEST50 = "resnest50"
    RESNEST101 = "resnest101"

@dataclass
class ModelCandidate:
    """Model candidate for selection"""
    model_type: ModelType
    model: nn.Module
    params: int
    flops: float
    val_score: float
    train_time: float

class AutoModelSelector:
    """Automatic model selection"""
    
    def __init__(
        self,
        num_classes: int,
        input_size: Tuple[int, int] = (224, 224),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        optimization_target: str = "accuracy"  # accuracy, latency, params
    ):
        self.num_classes = num_classes
        self.input_size = input_size
        self.device = device
        self.optimization_target = optimization_target
        
    def get_model(
        self,
        model_type: ModelType,
        pretrained: bool = True
    ) -> nn.Module:
        """Get model by type"""
        # Import from existing model files
        from ..models.advanced.efficientnet import EfficientNetClassifier
        from ..models.advanced.vit_classifier import ViTClassifier
        from ..models.advanced.swin_transformer import SwinTransformerClassifier
        from ..models.advanced.convnext import ConvNeXtClassifier
        
        model_map = {
            ModelType.EFFICIENTNET_B0: (EfficientNetClassifier, {'variant': 'b0'}),
            ModelType.EFFICIENTNET_B1: (EfficientNetClassifier, {'variant': 'b1'}),
            ModelType.EFFICIENTNET_B2: (EfficientNetClassifier, {'variant': 'b2'}),
            ModelType.EFFICIENTNET_B3: (EfficientNetClassifier, {'variant': 'b3'}),
            ModelType.EFFICIENTNET_B4: (EfficientNetClassifier, {'variant': 'b4'}),
            ModelType.EFFICIENTNET_B5: (EfficientNetClassifier, {'variant': 'b5'}),
            ModelType.VIT_SMALL: (ViTClassifier, {'variant': 'small'}),
            ModelType.VIT_BASE: (ViTClassifier, {'variant': 'base'}),
            ModelType.VIT_LARGE: (ViTClassifier, {'variant': 'large'}),
            ModelType.SWIN_TINY: (SwinTransformerClassifier, {'variant': 'tiny'}),
            ModelType.SWIN_SMALL: (SwinTransformerClassifier, {'variant': 'small'}),
            ModelType.SWIN_BASE: (SwinTransformerClassifier, {'variant': 'base'}),
            ModelType.CONVNEXT_TINY: (ConvNeXtClassifier, {'variant': 'tiny'}),
            ModelType.CONVNEXT_SMALL: (ConvNeXtClassifier, {'variant': 'small'}),
            ModelType.CONVNEXT_BASE: (ConvNeXtClassifier, {'variant': 'base'}),
        }
        
        model_class, kwargs = model_map[model_type]
        return model_class(
            num_classes=self.num_classes,
            **kwargs
        )
        
    def count_parameters(self, model: nn.Module) -> int:
        """Count model parameters"""
        return sum(p.numel() for p in model.parameters())
        
    def estimate_flops(self, model: nn.Module) -> float:
        """Estimate FLOPs"""
        # Simple estimation based on input
        dummy = torch.randn(1, 3, *self.input_size).to(self.device)
        model = model.to(self.device)
        
        # Count operations (simplified)
        flops = 0
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                flops += m.weight.numel()
                
        return float(flops)
        
    def evaluate_model(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        num_epochs: int = 10,
        lr: float = 0.001
    ) -> Dict[str, float]:
        """Evaluate a model"""
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training
        start_time = time.time()
        model.train()
        
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        train_time = time.time() - start_time
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        accuracy = correct / total
        
        # Measure latency
        latency = self._measure_latency(model)
        
        return {
            "accuracy": accuracy,
            "train_time": train_time,
            "latency_ms": latency
        }
        
    def _measure_latency(self, model: nn.Module) -> float:
        """Measure inference latency"""
        model.eval()
        dummy = torch.randn(1, 3, *self.input_size).to(self.device)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(dummy)
                
        # Measure
        latencies = []
        for _ in range(20):
            start = time.time()
            with torch.no_grad():
                _ = model(dummy)
            latencies.append((time.time() - start) * 1000)
            
        return np.mean(latencies)
        
    def select_best_model(
        self,
        train_loader,
        val_loader,
        candidate_models: Optional[List[ModelType]] = None,
        num_epochs: int = 10
    ) -> ModelCandidate:
        """Select best model from candidates"""
        if candidate_models is None:
            candidate_models = [
                ModelType.EFFICIENTNET_B0,
                ModelType.EFFICIENTNET_B1,
                ModelType.VIT_SMALL,
                ModelType.SWIN_TINY,
                ModelType.CONVNEXT_TINY,
            ]
            
        candidates = []
        
        for model_type in candidate_models:
            print(f"Evaluating {model_type.value}...")
            
            model = self.get_model(model_type)
            params = self.count_parameters(model)
            flops = self.estimate_flops(model)
            
            metrics = self.evaluate_model(
                model, train_loader, val_loader, num_epochs
            )
            
            candidate = ModelCandidate(
                model_type=model_type,
                model=model,
                params=params,
                flops=flops,
                val_score=metrics["accuracy"],
                train_time=metrics["train_time"]
            )
            candidates.append(candidate)
            
        # Select based on optimization target
        if self.optimization_target == "accuracy":
            best = max(candidates, key=lambda x: x.val_score)
        elif self.optimization_target == "latency":
            best = min(candidates, key=lambda x: x.train_time)
        elif self.optimization_target == "params":
            best = min(candidates, key=lambda x: x.params)
        else:
            best = max(candidates, key=lambda x: x.val_score)
            
        return best


class AutoHyperparameterOptimizer:
    """Automatic hyperparameter optimization"""
    
    def __init__(
        self,
        model_class,
        num_classes: int,
        input_size: Tuple[int, int] = (224, 224),
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_class = model_class
        self.num_classes = num_classes
        self.input_size = input_size
        self.device = device
        
    def objective(
        self,
        trial: optuna.Trial,
        train_loader,
        val_loader,
        num_epochs: int = 10
    ) -> float:
        """Optuna objective function"""
        # Sample hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        
        # Model-specific hyperparameters
        if hasattr(self.model_class, '__name__'):
            if 'ViT' in self.model_class.__name__:
                patch_size = trial.suggest_categorical("patch_size", [8, 16, 32])
                embed_dim = trial.suggest_categorical("embed_dim", [192, 256, 384])
                
            elif 'EfficientNet' in self.model_class.__name__:
                drop_rate = trial.suggest_float("drop_rate", 0.1, 0.5)
                
        # Create model
        model = self.model_class(num_classes=self.num_classes)
        model = model.to(self.device)
        
        # Training
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        
        # Train
        model.train()
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        # Validate
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        accuracy = correct / total
        
        return accuracy
        
    def optimize(
        self,
        train_loader,
        val_loader,
        num_trials: int = 50,
        num_epochs: int = 10
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(
            lambda trial: self.objective(
                trial, train_loader, val_loader, num_epochs
            ),
            n_trials=num_trials
        )
        
        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "study": study
        }


class AutoDataAugmentation:
    """Automatic data augmentation selection"""
    
    def __init__(self):
        self.augmentation_space = {
            "rotation": (0, 45),
            "horizontal_flip": (0, 1),
            "vertical_flip": (0, 1),
            "brightness": (0.5, 1.5),
            "contrast": (0.5, 1.5),
            "saturation": (0.5, 1.5),
            "hue": (-0.1, 0.1),
            "cutout": (0, 20),
            "mixup_alpha": (0.0, 1.0),
            "cutmix_alpha": (0.0, 1.0),
        }
        
    def objective(
        self,
        trial: optuna.Trial,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str = "cuda"
    ) -> float:
        """Optuna objective for augmentation"""
        # Sample augmentation parameters
        rotation = trial.suggest_int("rotation", 0, 45)
        h_flip = trial.suggest_float("horizontal_flip", 0, 1)
        brightness = trial.suggest_float("brightness", 0.5, 1.5)
        cutout = trial.suggest_int("cutout", 0, 20)
        
        # Create augmentation pipeline
        import torchvision.transforms as T
        
        transforms = [
            T.Resize((224, 224)),
            T.RandomRotation(rotation),
            T.RandomHorizontalFlip(h_flip),
            T.ColorJitter(brightness=brightness),
        ]
        
        if cutout > 0:
            transforms.append(T.RandomErasing(p=0.5, scale=(0.02, cutout/100)))
            
        transform = T.Compose(transforms)
        
        # Evaluate with augmentation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Apply augmentation
                augmented = torch.stack([transform(img) for img in inputs])
                augmented = augmented.to(device)
                labels = labels.to(device)
                
                outputs = model(augmented)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        accuracy = correct / total
        return accuracy
        
    def find_best_augmentation(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        num_trials: int = 20,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """Find best augmentation strategy"""
        study = optuna.create_study(direction="maximize")
        
        study.optimize(
            lambda trial: self.objective(
                trial, model, train_loader, val_loader, device
            ),
            n_trials=num_trials
        )
        
        return {
            "best_params": study.best_params,
            "best_value": study.best_value
        }
