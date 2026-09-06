import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from torch.utils.data import DataLoader, Subset

@dataclass
class UncertaintyScore:
    """Uncertainty scores for a sample"""
    sample_idx: int
    entropy: float
    margin: float
    confidence: float
    least_confidence: float
    bayesian_uncertainty: float

class UncertaintySampler:
    """Uncertainty-based active learning sampler"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_mc_samples: int = 10
    ):
        self.model = model
        self.device = device
        self.num_mc_samples = num_mc_samples
        
    def compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of predictions"""
        # Avoid log(0)
        probs = torch.clamp(probs, min=1e-10)
        entropy = -torch.sum(probs * torch.log(probs), dim=1)
        return entropy
        
    def compute_margin(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute margin between top-2 predictions"""
        sorted_probs, _ = torch.sort(probs, descending=True)
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        return 1 - margin  # Lower margin = higher uncertainty
        
    def compute_least_confidence(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute least confidence"""
        max_probs, _ = probs.max(dim=1)
        return 1 - max_probs
        
    def mc_dropout_uncertainty(
        self,
        inputs: torch.Tensor,
        num_samples: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Monte Carlo Dropout uncertainty estimation"""
        if num_samples is None:
            num_samples = self.num_mc_samples
            
        self.model.train()  # Enable dropout
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                logits = self.model(inputs)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
                
        predictions = torch.stack(predictions)  # [num_samples, batch, num_classes]
        
        # Mean prediction
        mean_preds = predictions.mean(dim=0)
        
        # Uncertainty (entropy of mean)
        entropy = self.compute_entropy(mean_preds)
        
        # Variance (epistemic uncertainty)
        variance = predictions.var(dim=0).mean(dim=1)
        
        self.model.eval()
        
        return entropy, variance
        
    def compute_uncertainty_scores(
        self,
        dataloader: DataLoader,
        num_samples: Optional[int] = None
    ) -> List[UncertaintyScore]:
        """Compute uncertainty scores for all samples"""
        self.model.eval()
        all_scores = []
        
        sample_count = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                logits = self.model(inputs)
                probs = F.softmax(logits, dim=1)
                
            # Compute uncertainties
            entropy = self.compute_entropy(probs)
            margin = self.compute_margin(probs)
            least_conf = self.compute_least_confidence(probs)
            
            for i in range(inputs.size(0)):
                score = UncertaintyScore(
                    sample_idx=sample_count + i,
                    entropy=entropy[i].item(),
                    margin=margin[i].item(),
                    confidence=1 - least_conf[i].item(),
                    least_confidence=least_conf[i].item(),
                    bayesian_uncertainty=0.0  # Computed separately if needed
                )
                all_scores.append(score)
                
            sample_count += inputs.size(0)
            
            if num_samples and sample_count >= num_samples:
                break
                
        return all_scores
        
    def select_samples(
        self,
        dataloader: DataLoader,
        n_samples: int,
        strategy: str = "entropy"
    ) -> List[int]:
        """Select samples for labeling"""
        scores = self.compute_uncertainty_scores(dataloader)
        
        if strategy == "entropy":
            # Sort by entropy (highest first)
            sorted_scores = sorted(scores, key=lambda x: x.entropy, reverse=True)
        elif strategy == "margin":
            # Sort by margin (highest first)
            sorted_scores = sorted(scores, key=lambda x: x.margin, reverse=True)
        elif strategy == "least_confidence":
            # Sort by least confidence (highest first)
            sorted_scores = sorted(scores, key=lambda x: x.least_confidence, reverse=True)
        elif strategy == "combined":
            # Combined score
            for score in scores:
                score._combined = (
                    score.entropy + 
                    score.margin + 
                    score.least_confidence
                ) / 3
            sorted_scores = sorted(scores, key=lambda x: x._combined, reverse=True)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        # Return top-n indices
        return [s.sample_idx for s in sorted_scores[:n_samples]]


class BayesianActiveLearning:
    """Bayesian active learning"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_mc_samples: int = 20
    ):
        self.model = model
        self.device = device
        self.num_mc_samples = num_samples
        
    def bayesian_uncertainty(
        self,
        inputs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute Bayesian uncertainty using MC Dropout"""
        self.model.train()  # Enable dropout
        
        # MC samples
        mc_predictions = []
        for _ in range(self.num_mc_samples):
            with torch.no_grad():
                logits = self.model(inputs)
                probs = F.softmax(logits, dim=1)
                mc_predictions.append(probs)
                
        mc_predictions = torch.stack(mc_predictions)
        
        # Mean prediction
        mean_prediction = mc_predictions.mean(dim=0)
        
        # Predictive entropy (total uncertainty)
        predictive_entropy = -torch.sum(
            mean_prediction * torch.log(mean_prediction + 1e-10),
            dim=1
        )
        
        # Expected entropy (data uncertainty)
        expected_entropy = -torch.sum(
            mc_predictions * torch.log(mc_predictions + 1e-10),
            dim=2
        ).mean(dim=0)
        
        # Mutual information (epistemic uncertainty)
        mutual_info = predictive_entropy - expected_entropy
        
        self.model.eval()
        
        return {
            "predictive_entropy": predictive_entropy,
            "expected_entropy": expected_entropy,
            "mutual_information": mutual_info,
            "mean_prediction": mean_prediction
        }
        
    def select_samples_bayesian(
        self,
        dataloader: DataLoader,
        n_samples: int,
        acquisition_function: str = "mutual_information"
    ) -> List[int]:
        """Select samples using Bayesian uncertainty"""
        all_scores = []
        sample_idx = 0
        
        for inputs, _ in dataloader:
            inputs = inputs.to(self.device)
            
            uncertainty = self.bayesian_uncertainty(inputs)
            
            for i in range(inputs.size(0)):
                if acquisition_function == "mutual_information":
                    score = uncertainty["mutual_information"][i].item()
                elif acquisition_function == "predictive_entropy":
                    score = uncertainty["predictive_entropy"][i].item()
                elif acquisition_function == "variance_ratio":
                    # Variance ratio
                    sorted_probs = torch.sort(
                        uncertainty["mean_prediction"][i], descending=True
                    )[0]
                    score = 1 - (sorted_probs[0] - sorted_probs[1])
                else:
                    raise ValueError(f"Unknown acquisition function: {acquisition_function}")
                    
                all_scores.append((sample_idx + i, score))
                
            sample_idx += inputs.size(0)
            
        # Sort by score (highest first)
        all_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [idx for idx, _ in all_scores[:n_samples]]


class QueryByCommittee:
    """Query by Committee active learning"""
    
    def __init__(
        self,
        models: List[nn.Module],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.models = models
        self.device = device
        
    def compute_disagreement(
        self,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """Compute committee disagreement"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(inputs.to(self.device))
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
                
        predictions = torch.stack(predictions)  # [num_models, batch, num_classes]
        
        # Vote entropy
        mean_preds = predictions.mean(dim=0)
        vote_entropy = -torch.sum(
            mean_preds * torch.log(mean_preds + 1e-10),
            dim=1
        )
        
        return vote_entropy
        
    def select_samples(
        self,
        dataloader: DataLoader,
        n_samples: int
    ) -> List[int]:
        """Select samples using committee disagreement"""
        all_scores = []
        sample_idx = 0
        
        for inputs, _ in dataloader:
            disagreement = self.compute_disagreement(inputs)
            
            for i in range(inputs.size(0)):
                all_scores.append((sample_idx + i, disagreement[i].item()))
                
            sample_idx += inputs.size(0)
            
        all_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [idx for idx, _ in all_scores[:n_samples]]


class CoreSetSelection:
    """Core-set based active learning"""
    
    def __init__(self, embedding_model: nn.Module, device: str = "cpu"):
        self.embedding_model = embedding_model
        self.device = device
        
    def get_embeddings(
        self,
        dataloader: DataLoader
    ) -> torch.Tensor:
        """Get embeddings for all samples"""
        self.embedding_model.eval()
        embeddings = []
        
        for inputs, _ in dataloader:
            inputs = inputs.to(self.device)
            with torch.no_grad():
                emb = self.embedding_model(inputs)
                embeddings.append(emb)
                
        return torch.cat(embeddings, dim=0)
        
    def greedy_core_set(
        self,
        embeddings: torch.Tensor,
        n_samples: int,
        labeled_indices: List[int] = None
    ) -> List[int]:
        """Greedy core-set selection"""
        n_total = embeddings.size(0)
        
        if labeled_indices is None:
            labeled_indices = []
            
        # Initialize with random sample
        if not labeled_indices:
            labeled_indices = [np.random.randint(n_total)]
            
        # Greedy selection
        for _ in range(n_samples):
            # Compute distances to nearest labeled point
            distances = torch.cdist(embeddings, embeddings[labeled_indices])
            min_distances = distances.min(dim=1)[0]
            
            # Select farthest point
            next_idx = min_distances.argmax().item()
            labeled_indices.append(next_idx)
            
        return labeled_indices


class DiversitySampling:
    """Diversity-based sampling"""
    
    def __init__(self, embedding_model: nn.Module, device: str = "cpu"):
        self.embedding_model = embedding_model
        self.device = device
        
    def max_margin_sampling(
        self,
        embeddings: torch.Tensor,
        n_samples: int
    ) -> List[int]:
        """Maximum margin sampling"""
        n_total = embeddings.size(0)
        selected = [np.random.randint(n_total)]
        
        for _ in range(n_samples - 1):
            # Compute distances to selected points
            distances = torch.cdist(embeddings, embeddings[selected])
            
            # For each point, find distance to nearest selected
            min_distances = distances.min(dim=1)[0]
            
            # Select point with maximum minimum distance
            next_idx = min_distances.argmax().item()
            selected.append(next_idx)
            
        return selected
        
    def k_center_greedy(
        self,
        embeddings: torch.Tensor,
        n_samples: int
    ) -> List[int]:
        """K-center greedy sampling"""
        n_total = embeddings.size(0)
        selected = [np.random.randint(n_total)]
        
        # Compute pairwise distances once
        pairwise_dist = torch.cdist(embeddings, embeddings)
        
        for _ in range(n_samples - 1):
            # Distance from each point to nearest selected
            dist_to_selected = pairwise_dist[:, selected]
            min_dist = dist_to_selected.min(dim=1)[0]
            
            # Select farthest
            next_idx = min_dist.argmax().item()
            selected.append(next_idx)
            
        return selected
