import torch
from torch.optim import Optimizer
from typing import List, Optional
import math

class AdamW(Optimizer):
    """AdamW optimizer with weight decay"""
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                state['step'] += 1
                
                # Decay the first and second moment estimates
                beta1, beta2 = group['betas']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


class LAMB(Optimizer):
    """Layer-wise Adaptive Moments optimizer for Batch training"""
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        clamp_value: float = 10.0
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, clamp_value=clamp_value)
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                state['step'] += 1
                
                # Update moments
                beta1, beta2 = group['betas']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Update gradient
                update = (exp_avg / bias_correction1) / (exp_avg_sq / bias_correction2).sqrt().add_(group['eps'])
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    update = update + group['weight_decay'] * p.data
                
                # Clamp update
                update = torch.clamp(update, -group['clamp_value'], group['clamp_value'])
                
                # Compute trust ratio
                if len(state) == 0:
                    state['norm_ratio'] = 1.0
                else:
                    weight_norm = p.data.norm(2)
                    update_norm = update.norm(2)
                    state['norm_ratio'] = weight_norm / update_norm
                
                # Update parameters
                p.data.add_(update, alpha=-group['lr'] * state['norm_ratio'])
        
        return loss


class RAdam(Optimizer):
    """RAdam optimizer"""
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                state['step'] += 1
                
                # Update moments
                beta1, beta2 = group['betas']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute length of approximate SMA
                rho_inf = 2 / (1 - beta2) - 1
                rho_t = rho_t = rho_inf - 2 * state['step'] * (beta2 ** state['step']) / bias_correction2
                
                # Check if we can use RAdam
                if rho_t > 5:
                    # Compute length of the approx. moving average
                    rect = math.sqrt(
                        (rho_t - 4) * (rho_t - 2) * rho_inf /
                        ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                    )
                    
                    # Compute adaptive learning rate
                    adaptive_lr = math.sqrt(bias_correction2) / (exp_avg_sq.sqrt() + group['eps'])
                    
                    # Update parameters
                    update = exp_avg * adaptive_lr * rect
                else:
                    # Use SGD with momentum
                    update = exp_avg / bias_correction1
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    update = update + group['weight_decay'] * p.data
                
                # Update parameters
                p.data.add_(update, alpha=-group['lr'])
        
        return loss


class Ranger(Optimizer):
    """Ranger optimizer (RAdam + Lookahead)"""
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        alpha: float = 0.5,
        k: int = 6
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, alpha=alpha, k=k)
        super().__init__(params, defaults)
        
        # Initialize slow weights
        self.slow_weights = []
        for group in self.param_groups:
            slow_group = []
            for param in group['params']:
                slow_group.append(param.data.clone())
            self.slow_weights.append(slow_group)
        
        self.step_counter = 0
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        # RAdam step
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                state['step'] += 1
                
                beta1, beta2 = group['betas']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                update = exp_avg / bias_correction1
                
                if group['weight_decay'] != 0:
                    update = update + group['weight_decay'] * p.data
                
                p.data.add_(update, alpha=-group['lr'])
        
        # Lookahead step
        self.step_counter += 1
        if self.step_counter % self.param_groups[0]['k'] == 0:
            for group_idx, group in enumerate(self.param_groups):
                for param_idx, param in enumerate(group['params']):
                    slow = self.slow_weights[group_idx][param_idx]
                    param.data = slow + group['alpha'] * (param.data - slow)
                    self.slow_weights[group_idx][param_idx] = param.data.clone()
        
        return loss


class AdamGS(Optimizer):
    """Adam with gradient scale"""
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        grad_scale: float = 1.0
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, grad_scale=grad_scale)
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data * group['grad_scale']
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                state['step'] += 1
                
                beta1, beta2 = group['betas']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                p.data.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group['eps']), value=-step_size)
        
        return loss
