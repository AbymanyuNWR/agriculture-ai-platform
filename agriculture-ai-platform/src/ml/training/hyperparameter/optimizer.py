import optuna
from typing import Dict, List, Optional, Tuple, Callable
import torch
import torch.nn as nn
from pathlib import Path
import json

class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(
        self,
        objective_fn: Callable,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        study_name: str = "agriculture_ai",
        storage: Optional[str] = None,
        direction: str = "maximize"
    ):
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        
    def optimize(self, **kwargs) -> Dict:
        """Run optimization"""
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=self.direction,
            load_if_exists=True
        )
        
        study.optimize(
            self.objective_fn,
            n_trials=self.n_trials,
            timeout=self.timeout,
            **kwargs
        )
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial': study.best_trial.number,
            'n_trials': len(study.trials)
        }
    
    def get_study_stats(self) -> Dict:
        """Get study statistics"""
        study = optuna.load_study(
            study_name=self.study_name,
            storage=self.storage
        )
        
        return {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'trial_values': [t.value for t in study.trials if t.value is not None]
        }


class ModelHyperparameterOptimizer:
    """Optimize model hyperparameters"""
    
    def __init__(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        n_trials: int = 50
    ):
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.n_trials = n_trials
        
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization"""
        # Suggest hyperparameters
        config = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd', 'adamw']),
            'scheduler': trial.suggest_categorical('scheduler', ['cosine', 'step', 'linear']),
            'model': trial.suggest_categorical('model', ['resnet50', 'efficientnet_b3', 'vit_small']),
        }
        
        # Train model
        model = self.train_fn(config)
        
        # Evaluate
        score = self.eval_fn(model)
        
        return score
    
    def optimize(self) -> Dict:
        """Run optimization"""
        optimizer = HyperparameterOptimizer(
            objective_fn=self.objective,
            n_trials=self.n_trials,
            study_name="model_optimization"
        )
        
        return optimizer.optimize()


class TrainingHyperparameterOptimizer:
    """Optimize training hyperparameters"""
    
    def __init__(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        n_trials: int = 30
    ):
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.n_trials = n_trials
        
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization"""
        # Suggest training hyperparameters
        config = {
            'epochs': trial.suggest_int('epochs', 10, 100),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'warmup_steps': trial.suggest_int('warmup_steps', 0, 1000),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            'gradient_clip': trial.suggest_float('gradient_clip', 0.1, 10.0),
            'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.3),
            'mixup_alpha': trial.suggest_float('mixup_alpha', 0.0, 1.0),
            'cutmix_alpha': trial.suggest_float('cutmix_alpha', 0.0, 1.0),
        }
        
        # Train model
        model = self.train_fn(config)
        
        # Evaluate
        score = self.eval_fn(model)
        
        return score
    
    def optimize(self) -> Dict:
        """Run optimization"""
        optimizer = HyperparameterOptimizer(
            objective_fn=self.objective,
            n_trials=self.n_trials,
            study_name="training_optimization"
        )
        
        return optimizer.optimize()


class ArchitectureSearch:
    """Neural Architecture Search"""
    
    def __init__(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        n_trials: int = 50
    ):
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.n_trials = n_trials
        
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for architecture search"""
        # Suggest architecture parameters
        config = {
            'num_layers': trial.suggest_int('num_layers', 2, 8),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256, 512, 1024]),
            'num_heads': trial.suggest_categorical('num_heads', [4, 8, 16, 32]),
            'mlp_ratio': trial.suggest_float('mlp_ratio', 2.0, 8.0),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'attention_dropout': trial.suggest_float('attention_dropout', 0.0, 0.3),
            'patch_size': trial.suggest_categorical('patch_size', [8, 16, 32]),
        }
        
        # Train model with suggested architecture
        model = self.train_fn(config)
        
        # Evaluate
        score = self.eval_fn(model)
        
        return score
    
    def optimize(self) -> Dict:
        """Run architecture search"""
        optimizer = HyperparameterOptimizer(
            objective_fn=self.objective,
            n_trials=self.n_trials,
            study_name="architecture_search"
        )
        
        return optimizer.optimize()


class DataAugmentationOptimizer:
    """Optimize data augmentation strategies"""
    
    def __init__(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        n_trials: int = 30
    ):
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.n_trials = n_trials
        
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for augmentation optimization"""
        # Suggest augmentation parameters
        config = {
            'random_flip': trial.suggest_categorical('random_flip', [True, False]),
            'random_rotation': trial.suggest_int('random_rotation', 0, 90),
            'random_crop': trial.suggest_categorical('random_crop', [True, False]),
            'color_jitter': trial.suggest_float('color_jitter', 0.0, 0.5),
            'random_erasing': trial.suggest_float('random_erasing', 0.0, 0.5),
            'mixup_alpha': trial.suggest_float('mixup_alpha', 0.0, 1.0),
            'cutmix_alpha': trial.suggest_float('cutmix_alpha', 0.0, 1.0),
            'auto_augment': trial.suggest_categorical('auto_augment', ['auto', 'rand', 'trivial', None]),
        }
        
        # Train model with suggested augmentation
        model = self.train_fn(config)
        
        # Evaluate
        score = self.eval_fn(model)
        
        return score
    
    def optimize(self) -> Dict:
        """Run augmentation optimization"""
        optimizer = HyperparameterOptimizer(
            objective_fn=self.objective,
            n_trials=self.n_trials,
            study_name="augmentation_optimization"
        )
        
        return optimizer.optimize()


def save_optimization_results(results: Dict, output_path: str):
    """Save optimization results"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def load_optimization_results(input_path: str) -> Dict:
    """Load optimization results"""
    with open(input_path) as f:
        return json.load(f)


def plot_optimization_history(study: optuna.Study, output_path: str):
    """Plot optimization history"""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot optimization history
    optuna.visualization.matplotlib.plot_optimization_history(study, ax=ax1)
    ax1.set_title('Optimization History')
    
    # Plot parameter importances
    optuna.visualization.matplotlib.plot_param_importances(study, ax=ax2)
    ax2.set_title('Parameter Importances')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
