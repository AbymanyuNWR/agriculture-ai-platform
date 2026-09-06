import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from collections import defaultdict
import copy

class ElasticWeightConsolidation:
    """Elastic Weight Consolidation for continual learning"""
    
    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.001,
        gamma: float = 1000.0
    ):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.fisher_dict = {}
        self.optimal_params = {}
        
    def compute_fisher(
        self,
        dataloader: DataLoader,
        num_samples: int = 1000
    ):
        """Compute Fisher information matrix"""
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        
        sample_count = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(next(self.model.parameters()).device), labels.to(...)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
                    
            sample_count += inputs.size(0)
            if sample_count >= num_samples:
                break
                
        # Average
        for n in fisher:
            fisher[n] /= sample_count
            
        self.fisher_dict = fisher
        
    def save_optimal_params(self):
        """Save current optimal parameters"""
        self.optimal_params = {
            n: p.data.clone() for n, p in self.model.named_parameters()
        }
        
    def penalty(self) -> torch.Tensor:
        """Compute EWC penalty"""
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for n, p in self.model.named_parameters():
            if n in self.fisher_dict:
                fisher = self.fisher_dict[n]
                optimal = self.optimal_params[n]
                loss += (fisher * (p - optimal).pow(2)).sum()
                
        return self.gamma * loss / 2
        
    def update(self, dataloader: DataLoader, num_epochs: int = 1):
        """Update model with EWC"""
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        for epoch in range(num_epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(next(self.model.parameters()).device), labels.to(...)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                task_loss = F.cross_entropy(outputs, labels)
                ewc_loss = self.penalty()
                loss = task_loss + ewc_loss
                
                loss.backward()
                optimizer.step()


class ProgressiveNeuralNetworks:
    """Progressive Neural Networks"""
    
    def __init__(self, base_model: nn.Module):
        self.base_model = base_model
        self.columns = [base_model]
        self.handles = []
        
    def add_column(self, new_model: nn.Module):
        """Add new column for new task"""
        self.columns.append(new_model)
        
    def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        """Forward pass with lateral connections"""
        features = []
        
        for i, column in enumerate(self.columns[:task_id + 1]):
            if i == 0:
                out = column(x)
            else:
                # Lateral connections
                lateral = torch.cat(features, dim=1)
                out = column(torch.cat([x, lateral], dim=1))
                
            features.append(out)
            
        return features[-1]


class MemoryReplay:
    """Experience replay with memory buffer"""
    
    def __init__(
        self,
        buffer_size: int = 1000,
        replay_ratio: float = 0.5
    ):
        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        self.buffer = defaultdict(list)
        
    def add_samples(
        self,
        dataloader: DataLoader,
        task_id: int,
        num_samples: int = 100
    ):
        """Add samples to memory buffer"""
        samples_added = 0
        
        for inputs, labels in dataloader:
            batch_size = inputs.size(0)
            samples_to_add = min(
                num_samples - samples_added,
                batch_size,
                self.buffer_size - len(self.buffer[task_id])
            )
            
            if samples_to_add <= 0:
                break
                
            self.buffer[task_id].extend([
                (inputs[i], labels[i]) for i in range(samples_to_add)
            ])
            
            samples_added += samples_to_add
            
    def sample(
        self,
        task_ids: List[int],
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from memory buffer"""
        samples = []
        
        for task_id in task_ids:
            if task_id in self.buffer and self.buffer[task_id]:
                task_samples = self.buffer[task_id]
                indices = np.random.choice(
                    len(task_ids),
                    size=min(batch_size // len(task_ids), len(task_samples)),
                    replace=False
                )
                samples.extend([task_samples[i] for i in indices])
                
        if not samples:
            return None, None
            
        inputs, labels = zip(*samples)
        return torch.stack(inputs), torch.tensor(labels)
        
    def get_replay_batch(
        self,
        current_inputs: torch.Tensor,
        current_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get replay batch mixed with current data"""
        batch_size = current_inputs.size(0)
        replay_size = int(batch_size * self.replay_ratio)
        
        if replay_size > 0:
            task_ids = list(self.buffer.keys())
            replay_inputs, replay_labels = self.sample(task_ids, replay_size)
            
            if replay_inputs is not None:
                combined_inputs = torch.cat([current_inputs, replay_inputs])
                combined_labels = torch.cat([current_labels, replay_labels])
                return combined_inputs, combined_labels
                
        return current_inputs, current_labels


class KnowledgeDistillationReplay:
    """Knowledge distillation for replay"""
    
    def __init__(
        self,
        old_model: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5
    ):
        self.old_model = old_model
        self.temperature = temperature
        self.alpha = alpha
        
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        old_logits: torch.Tensor
    ) -> torch.Tensor:
        """Combined loss with distillation"""
        # New task loss
        new_loss = F.cross_entropy(student_logits, labels)
        
        # Distillation loss
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_old = F.softmax(old_logits / self.temperature, dim=1)
        distill_loss = F.kl_div(
            soft_student, soft_old, reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combined
        return self.alpha * distill_loss + (1 - self.alpha) * new_loss


class PackNet:
    """PackNet for continual learning"""
    
    def __init__(self, model: nn.Module, prune_ratio: float = 0.3):
        self.model = model
        self.prune_ratio = prune_ratio
        self.masks = {}
        self.task_assignments = {}
        
    def prune(self, task_id: int):
        """Prune least important weights"""
        for name, param in self.model.named_parameters():
            # Get importance (gradient magnitude)
            if param.grad is not None:
                importance = param.grad.data.abs()
                
                # Create mask
                threshold = importance.quantile(self.prune_ratio)
                mask = (importance > threshold).float()
                
                # Store mask
                if task_id not in self.masks:
                    self.masks[task_id] = {}
                self.masks[task_id][name] = mask
                
                # Apply mask
                param.data *= mask
                
    def freeze_mask(self, task_id: int):
        """Freeze weights not in current task mask"""
        for name, param in self.model.named_parameters():
            if task_id in self.masks and name in self.masks[task_id]:
                # Only update non-masked weights
                mask = self.masks[task_id][name]
                param.register_hook(lambda grad, m=mask: grad * m)


class GradientEpisodicMemory:
    """Gradient Episodic Memory"""
    
    def __init__(
        self,
        model: nn.Module,
        memory_size: int = 1000,
        device: str = "cuda"
    ):
        self.model = model
        self.memory_size = memory_size
        self.device = device
        self.memory = []
        self.gradients = []
        
    def add_to_memory(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor
    ):
        """Add samples to memory"""
        # Compute gradients
        self.model.eval()
        self.model.zero_grad()
        
        outputs = self.model(inputs.to(self.device))
        loss = F.cross_entropy(outputs, labels.to(self.device))
        loss.backward()
        
        # Store gradients
        grads = [p.grad.clone() for p in self.model.parameters() if p.grad is not None]
        
        # Add to memory
        for i in range(inputs.size(0)):
            if len(self.memory) < self.memory_size:
                self.memory.append({
                    'input': inputs[i],
                    'label': labels[i],
                    'grads': [g[i] for g in grads]
                })
            else:
                # Replace random sample
                idx = np.random.randint(self.memory_size)
                self.memory[idx] = {
                    'input': inputs[i],
                    'label': labels[i],
                    'grads': [g[i] for g in grads]
                }
                
    def compute_gems_loss(self) -> torch.Tensor:
        """Compute GEMS loss"""
        if not self.memory:
            return torch.tensor(0.0).to(self.device)
            
        # Get current gradients
        current_grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        
        # Compute reference gradients
        ref_grad = [torch.zeros_like(g) for g in current_grads]
        for sample in self.memory:
            for i, g in enumerate(sample['grads']):
                ref_grad[i] += g / len(self.memory)
                
        # Compute constraint
        dot_products = []
        for g, rg in zip(current_grads, ref_grad):
            dot_products.append((g * rg).sum())
            
        total_dot = torch.stack(dot_products).sum()
        
        # If current gradient is aligned with reference, no penalty
        if total_dot >= 0:
            return torch.tensor(0.0).to(self.device)
        else:
            # Project gradient
            penalty = -total_dot
            return penalty
