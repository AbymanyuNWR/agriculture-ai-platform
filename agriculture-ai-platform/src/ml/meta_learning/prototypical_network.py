import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

class PrototypicalNetwork(nn.Module):
    """Prototypical Network for few-shot learning"""
    
    def __init__(
        self,
        encoder: nn.Module,
        distance_metric: str = "euclidean"
    ):
        super().__init__()
        self.encoder = encoder
        self.distance_metric = distance_metric
        
    def compute_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        num_classes: int
    ) -> torch.Tensor:
        """Compute class prototypes"""
        prototypes = []
        
        for c in range(num_classes):
            mask = support_labels == c
            class_embeddings = support_embeddings[mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
            
        return torch.stack(prototypes)
        
    def compute_distance(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """Compute distance to prototypes"""
        if self.distance_metric == "euclidean":
            # Euclidean distance
            distances = torch.cdist(query_embeddings, prototypes)
            return -distances  # Negative for compatibility with cross-entropy
        elif self.distance_metric == "cosine":
            # Cosine similarity
            query_norm = F.normalize(query_embeddings, dim=1)
            proto_norm = F.normalize(prototypes, dim=1)
            return torch.mm(query_norm, proto_norm.t())
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
            
    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        num_classes: int
    ) -> torch.Tensor:
        """Forward pass"""
        # Encode support set
        support_embeddings = self.encoder(support_images)
        
        # Compute prototypes
        prototypes = self.compute_prototypes(
            support_embeddings, support_labels, num_classes
        )
        
        # Encode query set
        query_embeddings = self.encoder(query_images)
        
        # Compute distances
        distances = self.compute_distance(query_embeddings, prototypes)
        
        return distances


class MatchingNetwork(nn.Module):
    """Matching Network"""
    
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.encoder = encoder
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        num_classes: int
    ) -> torch.Tensor:
        """Forward pass"""
        # Encode
        support_embeddings = self.encoder(support_images)
        query_embeddings = self.encoder(query_images)
        
        # Attention
        attention_weights = F.softmax(
            torch.mm(query_embeddings, support_embeddings.t()),
            dim=1
        )
        
        # Weighted sum
        outputs = torch.mm(attention_weights, F.one_hot(
            support_labels, num_classes
        ).float())
        
        return outputs


class RelationNetwork(nn.Module):
    """Relation Network"""
    
    def __init__(
        self,
        encoder: nn.Module,
        relation_module: nn.Module
    ):
        super().__init__()
        self.encoder = encoder
        self.relation_module = relation_module
        
    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        num_classes: int
    ) -> torch.Tensor:
        """Forward pass"""
        # Encode
        support_embeddings = self.encoder(support_images)
        query_embeddings = self.encoder(query_images)
        
        # Compute relations
        relations = []
        for c in range(num_classes):
            mask = support_labels == c
            class_embeddings = support_embeddings[mask]
            
            # Pairwise relations
            for i in range(query_embeddings.size(0)):
                query = query_embeddings[i:i+1].repeat(class_embeddings.size(0), 1)
                pairs = torch.cat([query, class_embeddings], dim=1)
                rel = self.relation_module(pairs)
                relations.append(rel.mean())
                
        return torch.tensor(relations).reshape(num_classes, -1).t()


class MetaLearner(nn.Module):
    """MAML-style meta-learner"""
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        num_inner_steps: int = 5
    ):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        
    def inner_loop(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Inner loop adaptation"""
        # Clone parameters
        fast_weights = {
            name: param.clone() 
            for name, param in self.model.named_parameters()
        }
        
        for _ in range(self.num_inner_steps):
            # Forward with fast weights
            outputs = self.model.functional_forward(
                support_images, fast_weights
            )
            loss = F.cross_entropy(outputs, support_labels)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss, fast_weights.values(), create_graph=True
            )
            
            # Update fast weights
            fast_weights = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(fast_weights.items(), grads)
            }
            
        return fast_weights
        
    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass"""
        # Inner loop
        fast_weights = self.inner_loop(support_images, support_labels)
        
        # Query predictions
        outputs = self.model.functional_forward(query_images, fast_weights)
        
        return outputs


class PrototypicalNetworkTrainer:
    """Trainer for prototypical networks"""
    
    def __init__(
        self,
        model: PrototypicalNetwork,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda"
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
    def train_episode(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
        num_classes: int
    ) -> Dict[str, float]:
        """Train on a single episode"""
        self.model.train()
        
        # Move to device
        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        query_labels = query_labels.to(self.device)
        
        # Forward
        self.optimizer.zero_grad()
        logits = self.model(
            support_images, support_labels, query_images, num_classes
        )
        
        # Loss
        loss = F.cross_entropy(logits, query_labels)
        
        # Backward
        loss.backward()
        self.optimizer.step()
        
        # Accuracy
        _, predicted = logits.max(1)
        accuracy = (predicted == query_labels).float().mean()
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy.item()
        }
        
    def evaluate_episode(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
        num_classes: int
    ) -> Dict[str, float]:
        """Evaluate on a single episode"""
        self.model.eval()
        
        with torch.no_grad():
            support_images = support_images.to(self.device)
            support_labels = support_labels.to(self.device)
            query_images = query_images.to(self.device)
            query_labels = query_labels.to(self.device)
            
            logits = self.model(
                support_images, support_labels, query_images, num_classes
            )
            
            loss = F.cross_entropy(logits, query_labels)
            
            _, predicted = logits.max(1)
            accuracy = (predicted == query_labels).float().mean()
            
        return {
            "loss": loss.item(),
            "accuracy": accuracy.item()
        }


class EpisodeSampler:
    """Sampler for few-shot episodes"""
    
    def __init__(
        self,
        dataset,
        num_classes: int = 5,
        num_support: int = 5,
        num_query: int = 15
    ):
        self.dataset = dataset
        self.num_classes = num_classes
        self.num_support = num_support
        self.num_query = num_query
        
        # Group by class
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(dataset):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)
            
    def sample_episode(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample a single episode"""
        # Sample classes
        classes = list(self.class_to_indices.keys())
        selected_classes = np.random.choice(
            classes, size=self.num_classes, replace=False
        )
        
        support_indices = []
        query_indices = []
        support_labels = []
        query_labels = []
        
        for i, cls in enumerate(selected_classes):
            class_indices = self.class_to_indices[cls]
            
            # Sample support and query
            support = np.random.choice(
                class_indices, size=self.num_support, replace=False
            )
            query = np.random.choice(
                [idx for idx in class_indices if idx not in support],
                size=self.num_query,
                replace=False
            )
            
            support_indices.extend(support)
            query_indices.extend(query)
            support_labels.extend([i] * self.num_support)
            query_labels.extend([i] * self.num_query)
            
        # Load images
        support_images = torch.stack([
            self.dataset[idx][0] for idx in support_indices
        ])
        query_images = torch.stack([
            self.dataset[idx][0] for idx in query_indices
        ])
        
        return {
            "support_images": support_images,
            "support_labels": torch.tensor(support_labels),
            "query_images": query_images,
            "query_labels": torch.tensor(query_labels),
            "num_classes": self.num_classes
        }
