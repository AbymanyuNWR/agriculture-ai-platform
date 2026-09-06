import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, Optional
import os

class DistributedManager:
    """Manager for distributed training"""
    
    def __init__(self):
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.distributed = False
        
    def setup(self, backend: str = 'nccl'):
        """Setup distributed training"""
        if 'RANK' in os.environ:
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
            
            dist.init_process_group(backend=backend)
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
            self.distributed = True
            
            print(f"Distributed training initialized: rank {self.rank}/{self.world_size}")
        else:
            print("Single GPU training")
            
    def cleanup(self):
        """Cleanup distributed training"""
        if self.distributed:
            dist.destroy_process_group()
            
    def is_main_process(self) -> bool:
        """Check if current process is main"""
        return self.rank == 0
    
    def barrier(self):
        """Synchronize all processes"""
        if self.distributed:
            dist.barrier()
            
    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        """Broadcast tensor from src to all processes"""
        if self.distributed:
            dist.broadcast(tensor, src)
            
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM):
        """All reduce tensor"""
        if self.distributed:
            dist.all_reduce(tensor, op)
            
    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sum tensor across all processes"""
        if self.distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor
    
    def get_sampler(self, dataset, shuffle: bool = True) -> DistributedSampler:
        """Get distributed sampler"""
        if self.distributed:
            return DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle
            )
        return None


class DistributedTrainer:
    """Distributed training wrapper"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        
    def setup_ddp(self):
        """Setup DDP"""
        self.model = DDP(self.model, device_ids=[self.device])
        
    def train_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            _, predicted = outputs.max(1)
            accuracy = (predicted == labels).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
    
    def validate_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Single validation step"""
        self.model.eval()
        
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            
            _, predicted = outputs.max(1)
            accuracy = (predicted == labels).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }


class DataParallelTrainer:
    """Data parallel training wrapper"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device_ids: list = None
    ):
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        
        self.model = nn.DataParallel(model, device_ids=device_ids)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
    def train_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Forward pass
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            _, predicted = outputs.max(1)
            accuracy = (predicted == labels).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }


class ModelEMA:
    """Exponential Moving Average for model parameters"""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        """Update EMA"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                
    def apply_shadow(self):
        """Apply shadow to model"""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
                
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class GradientAccumulation:
    """Gradient accumulation wrapper"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        accumulation_steps: int = 4
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        
    def train_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Single training step with gradient accumulation"""
        self.model.train()
        
        # Forward pass
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels) / self.accumulation_steps
        
        # Backward pass
        loss.backward()
        
        self.current_step += 1
        
        # Update weights
        if self.current_step % self.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.current_step = 0
        
        # Calculate metrics
        with torch.no_grad():
            _, predicted = outputs.max(1)
            accuracy = (predicted == labels).float().mean()
        
        return {
            'loss': loss.item() * self.accumulation_steps,
            'accuracy': accuracy.item()
        }


def setup_distributed(rank: int, world_size: int):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str
):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device
) -> Tuple[int, float]:
    """Load training checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
