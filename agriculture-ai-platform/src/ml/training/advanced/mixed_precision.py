import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Tuple
import numpy as np

class MixedPrecisionTrainer:
    """Mixed precision training for faster training and lower memory usage"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        max_grad_norm: float = 1.0,
        scale_factor: float = 2.0,
        growth_interval: int = 2000
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.max_grad_norm = max_grad_norm
        
        # Initialize GradScaler for mixed precision
        self.scaler = GradScaler(
            init_scale=2.**16,
            growth_factor=scale_factor,
            backoff_factor=0.5,
            growth_interval=growth_interval
        )
        
        # Training metrics
        self.scaler_history = []
        
    def train_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        accumulation_steps: int = 1
    ) -> Dict[str, float]:
        """Single training step with mixed precision"""
        self.model.train()
        
        # Forward pass with autocast
        with autocast():
            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = self.loss_fn(outputs, labels)
            loss = loss / accumulation_steps
        
        # Backward pass with scaler
        self.scaler.scale(loss).backward()
        
        # Gradient accumulation
        if self.scaler.get_scale() > 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        # Record scaler value
        self.scaler_history.append(self.scaler.get_scale())
        
        # Calculate metrics
        with torch.no_grad():
            _, predicted = outputs.max(1)
            accuracy = (predicted == labels).float().mean()
        
        return {
            'loss': loss.item() * accumulation_steps,
            'accuracy': accuracy.item(),
            'scaler_scale': self.scaler.get_scale()
        }
    
    def validate_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Single validation step"""
        self.model.eval()
        
        with torch.no_grad():
            with autocast():
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = self.loss_fn(outputs, labels)
            
            _, predicted = outputs.max(1)
            accuracy = (predicted == labels).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
    
    def get_scaler_statistics(self) -> Dict[str, float]:
        """Get GradScaler statistics"""
        if not self.scaler_history:
            return {}
        
        return {
            'current_scale': self.scaler.get_scale(),
            'min_scale': min(self.scaler_history),
            'max_scale': max(self.scaler_history),
            'mean_scale': np.mean(self.scaler_history),
            'growth_interval': self.scaler._growth_interval
        }


class GradientAccumulationTrainer:
    """Gradient accumulation for effectively larger batch sizes"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        accumulation_steps: int = 4,
        max_grad_norm: float = 1.0
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        self.current_step = 0
        self.accumulated_loss = 0.0
        
    def train_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Single training step with gradient accumulation"""
        self.model.train()
        
        # Forward pass
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = self.loss_fn(outputs, labels)
        
        # Scale loss for accumulation
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        # Accumulate loss for logging
        self.accumulated_loss += loss.item()
        self.current_step += 1
        
        # Perform optimization step when accumulation is complete
        if self.current_step % self.accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            
            # Update weights
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Reset accumulated loss
            avg_loss = self.accumulated_loss / self.accumulation_steps
            self.accumulated_loss = 0.0
        else:
            avg_loss = loss.item()
        
        # Calculate metrics
        with torch.no_grad():
            _, predicted = outputs.max(1)
            accuracy = (predicted == labels).float().mean()
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy.item(),
            'accumulation_step': self.current_step % self.accumulation_steps,
            'is_update_step': self.current_step % self.accumulation_steps == 0
        }


class EMA:
    """Exponential Moving Average of model parameters"""
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 1000
    ):
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        
        # Create EMA copy of model
        self.ema_model = self._create_ema_model()
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        
        self.step = 0
        
    def _create_ema_model(self):
        """Create EMA copy of model"""
        import copy
        return copy.deepcopy(self.model)
    
    def _get_decay(self):
        """Get current decay with warmup"""
        self.step += 1
        if self.step <= self.warmup_steps:
            return min(self.decay, (1 + self.step) / (self.warmup_steps + self.step))
        return self.decay
    
    def update(self):
        """Update EMA model"""
        decay = self._get_decay()
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = decay * self.shadow[name] + (1 - decay) * param.data
    
    def apply_shadow(self):
        """Apply EMA shadow to model"""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original model parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    def get_ema_model(self):
        """Get EMA model for inference"""
        self.apply_shadow()
        return self.model
    
    def state_dict(self):
        """Get EMA state dict"""
        return {
            'shadow': self.shadow,
            'step': self.step,
            'decay': self.decay
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state dict"""
        self.shadow = state_dict['shadow']
        self.step = state_dict['step']
        self.decay = state_dict['decay']


class GradientCentralization:
    """Gradient centralization for better training"""
    
    def __init__(self, model: nn.Module, use_gc: bool = True, gc_conv_only: bool = False):
        self.model = model
        self.use_gc = use_gc
        self.gc_conv_only = gc_conv_only
        
        if use_gc:
            self._register_hook()
    
    def _register_hook(self):
        """Register gradient hook for centralization"""
        for name, param in self.model.named_parameters():
            if self.gc_conv_only and 'conv' not in name:
                continue
            if param.requires_grad:
                param.register_hook(self._gradient_centralization_hook)
    
    @staticmethod
    def _gradient_centralization_hook(grad):
        """Gradient centralization hook"""
        if len(grad.shape) > 1:
            # For conv and linear layers
            mean = grad.mean(dim=list(range(1, len(grad.shape))), keepdim=True)
            grad = grad - mean
        return grad
    
    def zero_grad(self):
        """Zero gradients with centralization"""
        self.model.zero_grad()


class Lookahead:
    """Lookahead optimizer wrapper"""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        k: int = 5,
        alpha: float = 0.5
    ):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        
        # Create slow weights
        self.slow_weights = []
        for group in optimizer.param_groups:
            slow_group = []
            for param in group['params']:
                if param.requires_grad:
                    slow_group.append(param.data.clone())
                else:
                    slow_group.append(None)
            self.slow_weights.append(slow_group)
        
        self.step_counter = 0
    
    def step(self):
        """Perform Lookahead step"""
        self.optimizer.step()
        self.step_counter += 1
        
        if self.step_counter % self.k == 0:
            for group_idx, group in enumerate(self.optimizer.param_groups):
                for param_idx, param in enumerate(group['params']):
                    if param.requires_grad and self.slow_weights[group_idx][param_idx] is not None:
                        # Update slow weights
                        slow = self.slow_weights[group_idx][param_idx]
                        param.data = slow + self.alpha * (param.data - slow)
                        self.slow_weights[group_idx][param_idx] = param.data.clone()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'slow_weights': self.slow_weights,
            'step_counter': self.step_counter
        }
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.slow_weights = state_dict['slow_weights']
        self.step_counter = state_dict['step_counter']
