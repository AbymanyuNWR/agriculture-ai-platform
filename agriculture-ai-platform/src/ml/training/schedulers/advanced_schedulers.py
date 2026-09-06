import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import List

class CosineAnnealingWarmRestarts(_LRScheduler):
    """Cosine Annealing with Warm Restarts"""
    
    def __init__(
        self,
        optimizer,
        T_0: int = 10,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        T_cur = self.last_epoch % self.T_0
        T_i = self.T_0
        
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
            for base_lr in self.base_lrs
        ]


class OneCycleLR(_LRScheduler):
    """One Cycle Learning Rate Policy"""
    
    def __init__(
        self,
        optimizer,
        max_lr: float,
        steps_per_epoch: int,
        epochs: int,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1
    ):
        self.max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.total_steps = steps_per_epoch * epochs
        self.step_size_up = int(pct_start * self.total_steps)
        
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        step = self.last_epoch
        
        if step < self.step_size_up:
            # Warmup phase
            lr = self.max_lr * (step / self.step_size_up)
        else:
            # Annealing phase
            if self.anneal_strategy == 'cos':
                t = (step - self.step_size_up) / (self.total_steps - self.step_size_up)
                lr = self.max_lr * (1 + math.cos(math.pi * t)) / 2
            else:
                t = (step - self.step_size_up) / (self.total_steps - self.step_size_up)
                lr = self.max_lr * (1 - t)
        
        return [lr for _ in self.base_lrs]


class WarmupCosineScheduler(_LRScheduler):
    """Cosine Scheduler with Linear Warmup"""
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            lr_scale = step / self.warmup_steps
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
        
        return [max(base_lr * lr_scale, self.min_lr) for base_lr in self.base_lrs]


class PolynomialLR(_LRScheduler):
    """Polynomial Learning Rate Schedule"""
    
    def __init__(
        self,
        optimizer,
        total_steps: int,
        power: float = 0.9,
        min_lr: float = 0,
        last_epoch: int = -1
    ):
        self.total_steps = total_steps
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        step = min(self.last_epoch, self.total_steps)
        
        lr_scale = (1 - step / self.total_steps) ** self.power
        
        return [max(base_lr * lr_scale, self.min_lr) for base_lr in self.base_lrs]


class CyclicLR(_LRScheduler):
    """Cyclic Learning Rate Policy"""
    
    def __init__(
        self,
        optimizer,
        base_lr: float,
        max_lr: float,
        step_size_up: int = 2000,
        step_size_down: int = None,
        mode: str = 'triangular',
        gamma: float = 1.0,
        scale_fn=None,
        scale_mode: str = 'cycle',
        last_epoch: int = -1
    ):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        step = self.last_epoch
        
        cycle = math.floor(1 + step / (self.step_size_up + self.step_size_down))
        x = abs(step / (self.step_size_up + self.step_size_down) - cycle)
        
        if self.scale_fn is None:
            if self.mode == 'triangular':
                scale = 1.0
            elif self.mode == 'triangular2':
                scale = 1 / (2 ** (cycle - 1))
            elif self.mode == 'exp_range':
                scale = self.gamma ** step
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
        else:
            scale = self.scale_fn(cycle)
        
        lrs = []
        for base_lr in self.base_lrs:
            base_height = (1 - x) * scale
            if step % 2 == 0:
                lr = base_lr + (self.max_lr - base_lr) * base_height
            else:
                lr = self.max_lr - (self.max_lr - base_lr) * base_height
            lrs.append(lr)
        
        return lrs


class LinearWarmupScheduler(_LRScheduler):
    """Linear Warmup Scheduler"""
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        initial_lr: float = 0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        step = self.last_epoch
        
        if step < self.warmup_steps:
            lr_scale = step / self.warmup_steps
            return [base_lr * lr_scale for base_lr in self.base_lrs]
        
        return self.base_lrs


class ExponentialWarmupScheduler(_LRScheduler):
    """Exponential Warmup Scheduler"""
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        decay: float = 0.99,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.decay = decay
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        step = self.last_epoch
        
        if step < self.warmup_steps:
            lr_scale = math.exp(step / self.warmup_steps * math.log(10))
        else:
            lr_scale = self.decay ** (step - self.warmup_steps)
        
        return [base_lr * lr_scale for base_lr in self.base_lrs]


class CosineWithRestarts(_LRScheduler):
    """Cosine Annealing with Warm Restarts"""
    
    def __init__(
        self,
        optimizer,
        T_max: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1
    ):
        self.T_max = T_max
        self.T_mult = T_mult
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        step = self.last_epoch
        
        # Calculate current cycle
        cycle = 0
        while step > self.T_max:
            step -= self.T_max
            cycle += 1
            self.T_max *= self.T_mult
        
        # Calculate learning rate
        lr_scale = 0.5 * (1 + math.cos(math.pi * step / self.T_max))
        
        return [max(base_lr * lr_scale, self.eta_min) for base_lr in self.base_lrs]


class MultiStepLRWithWarmup(_LRScheduler):
    """Multi-step LR with warmup"""
    
    def __init__(
        self,
        optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_steps: int = 0,
        last_epoch: int = -1
    ):
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        step = self.last_epoch
        
        # Warmup phase
        if step < self.warmup_steps:
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        
        # Milestone phase
        return [
            base_lr * self.gamma ** sum(1 for m in self.milestones if m <= step)
            for base_lr in self.base_lrs
        ]
