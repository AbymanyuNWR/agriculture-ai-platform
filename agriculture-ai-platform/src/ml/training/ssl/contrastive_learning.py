import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class SimCLR(nn.Module):
    """SimCLR: A Simple Framework for Contrastive Learning"""
    
    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 128,
        temperature: float = 0.07
    ):
        super().__init__()
        self.encoder = encoder
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            encoder_dim = encoder(dummy).shape[1]
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, projection_dim)
        )
        
        self.temperature = temperature
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for two augmented views"""
        # Encode
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        
        # Project
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        
        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        return z1, z2
    
    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """NT-Xent loss"""
        batch_size = z1.shape[0]
        
        # Concatenate representations
        z = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature
        
        # Create mask for positive pairs
        mask = torch.eye(2 * batch_size, device=z.device)
        
        # Remove diagonal
        sim = sim * (1 - mask)
        
        # Create labels (positive pairs are offset by batch_size)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(z.device)
        
        # Compute loss
        loss = F.cross_entropy(sim, labels)
        
        return loss
    
    def nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Alternative NT-Xent loss implementation"""
        batch_size = z1.shape[0]
        
        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Similarity matrix
        sim = torch.mm(z1, z2.t()) / self.temperature
        
        # Labels
        labels = torch.arange(batch_size, device=z1.device)
        
        # Loss
        loss = F.cross_entropy(sim, labels)
        
        return loss


class MoCo(nn.Module):
    """Momentum Contrast for Unsupervised Visual Representation Learning"""
    
    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 128,
        queue_size: int = 65536,
        momentum: float = 0.999,
        temperature: float = 0.07
    ):
        super().__init__()
        self.encoder = encoder
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            encoder_dim = encoder(dummy).shape[1]
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, projection_dim)
        )
        
        # Momentum encoder
        self.momentum_encoder = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, projection_dim)
        )
        
        # Initialize momentum encoder
        self._update_momentum_encoder()
        
        # Queue
        self.register_buffer('queue', F.normalize(torch.randn(queue_size, projection_dim), dim=1))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        
    @torch.no_grad()
    def _update_momentum_encoder(self):
        """Update momentum encoder"""
        for param_q, param_k in zip(self.projector.parameters(), self.momentum_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Dequeue and enqueue keys"""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Replace keys at ptr
        self.queue[ptr:ptr + batch_size] = keys
        
        # Update pointer
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Query
        q = self.projector(self.encoder(x1))
        q = F.normalize(q, dim=1)
        
        # Key
        with torch.no_grad():
            self._update_momentum_encoder()
            k = self.momentum_encoder(self.encoder(x2))
            k = F.normalize(k, dim=1)
        
        return q, k
    
    def contrastive_loss(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """MoCo contrastive loss"""
        batch_size = q.shape[0]
        
        # Positive similarity
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Negative similarity
        l_neg = torch.einsum('nc,kc->nk', [q, self.queue])
        
        # Logits
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        
        # Labels (positive is at index 0)
        labels = torch.zeros(batch_size, dtype=torch.long, device=q.device)
        
        # Loss
        loss = F.cross_entropy(logits, labels)
        
        # Update queue
        self._dequeue_and_enqueue(k)
        
        return loss


class BYOL(nn.Module):
    """Bootstrap Your Own Latent"""
    
    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 256,
        hidden_dim: int = 4096,
        momentum: float = 0.99
    ):
        super().__init__()
        self.encoder = encoder
        self.momentum = momentum
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            encoder_dim = encoder(dummy).shape[1]
        
        # Projector
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # Momentum encoder
        self.momentum_encoder = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # Initialize momentum projector
        self._update_momentum_encoder()
        
    @torch.no_grad()
    def _update_momentum_encoder(self):
        """Update momentum encoder"""
        for param_q, param_k in zip(self.projector.parameters(), self.momentum_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)
            
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Online network
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        
        # Predictor
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        # Momentum network
        with torch.no_grad():
            self._update_momentum_encoder()
            q1 = self.momentum_encoder(self.encoder(x1))
            q2 = self.momentum_encoder(self.encoder(x2))
        
        return p1, q2, p2, q1
    
    def byol_loss(self, p1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """BYOL loss"""
        return 2 - 2 * F.cosine_similarity(p1, q2, dim=-1).mean()


class DINO(nn.Module):
    """Self-distillation with no labels"""
    
    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 256,
        hidden_dim: int = 2048,
        momentum: float = 0.99,
        temperature: float = 0.04
    ):
        super().__init__()
        self.encoder = encoder
        self.momentum = momentum
        self.temperature = temperature
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            encoder_dim = encoder(dummy).shape[1]
        
        # Student projector
        self.student_projector = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # Teacher projector
        self.teacher_projector = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # Student head
        self.student_head = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 10)  # num_classes
        )
        
        # Teacher head
        self.teacher_head = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 10)
        )
        
        # Initialize teacher
        self._update_teacher()
        
    @torch.no_grad()
    def _update_teacher(self):
        """Update teacher network"""
        for param_s, param_t in zip(
            list(self.student_projector.parameters()) + list(self.student_head.parameters()),
            list(self.teacher_projector.parameters()) + list(self.teacher_head.parameters())
        ):
            param_t.data = param_t.data * self.momentum + param_s.data * (1 - self.momentum)
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Student
        z_student = self.student_projector(self.encoder(x))
        p_student = self.student_head(z_student)
        
        # Teacher
        with torch.no_grad():
            self._update_teacher()
            z_teacher = self.teacher_projector(self.encoder(x))
            p_teacher = self.teacher_head(z_teacher)
        
        return p_student, p_teacher
    
    def dino_loss(self, p_student: torch.Tensor, p_teacher: torch.Tensor) -> torch.Tensor:
        """DINO loss"""
        p_student = F.log_softmax(p_student / self.temperature, dim=1)
        p_teacher = F.softmax(p_teacher / self.temperature, dim=1)
        
        loss = -torch.sum(p_teacher * p_student, dim=1).mean()
        
        return loss


class SimSiam(nn.Module):
    """Simple Siamese Representation Learning"""
    
    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 2048,
        hidden_dim: int = 512
    ):
        super().__init__()
        self.encoder = encoder
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            encoder_dim = encoder(dummy).shape[1]
        
        # Projector
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Encoder
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        
        # Predictor
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        return p1, z2, p2, z1
    
    def simsiam_loss(self, p1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """SimSiam loss"""
        return -F.cosine_similarity(p1, z2.detach(), dim=-1).mean()
