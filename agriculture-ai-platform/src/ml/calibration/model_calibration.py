import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.calibration import calibration_curve

class TemperatureScaling:
    """Temperature scaling for calibration"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def fit(
        self,
        dataloader: DataLoader,
        device: str = "cuda"
    ):
        """Fit temperature parameter"""
        self.model.eval()
        
        # Collect logits and labels
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                logits = self.model(inputs)
                all_logits.append(logits.cpu())
                all_labels.append(labels)
                
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01)
        
        def eval_temperature():
            optimizer.zero_grad()
            scaled_logits = all_logits / self.temperature
            loss = F.cross_entropy(scaled_logits, all_labels)
            loss.backward()
            return loss
            
        optimizer.step(eval_temperature)
        
    def calibrate(
        self,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """Apply temperature scaling"""
        return logits / self.temperature


class PlattScaling:
    """Platt scaling for binary classification"""
    
    def __init__(self):
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))
        
    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ):
        """Fit Platt scaling parameters"""
        optimizer = torch.optim.LBFGS([self.a, self.b], lr=0.01)
        
        def eval_platt():
            optimizer.zero_grad()
            scaled = self.a * logits[:, 1] + self.b
            loss = F.binary_cross_entropy_with_logits(
                scaled, labels.float()
            )
            loss.backward()
            return loss
            
        optimizer.step(eval_platt)
        
    def calibrate(
        self,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """Apply Platt scaling"""
        scaled = self.a * logits + self.b
        return torch.sigmoid(scaled)


class IsotonicRegression:
    """Isotonic regression calibration"""
    
    def __init__(self):
        self.calibrator = None
        
    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ):
        """Fit isotonic regression"""
        from sklearn.isotonic import IsotonicRegression
        
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(probabilities, labels)
        
    def calibrate(
        self,
        probabilities: np.ndarray
    ) -> np.ndarray:
        """Apply isotonic calibration"""
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted")
            
        return self.calibrator.predict(probabilities)


class ModelCalibrator:
    """Complete model calibration pipeline"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.temperature_scaler = TemperatureScaling(model)
        
    def compute_calibration_metrics(
        self,
        dataloader: DataLoader,
        n_bins: int = 10,
        device: str = "cuda"
    ) -> Dict[str, float]:
        """Compute calibration metrics"""
        self.model.eval()
        
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                logits = self.model(inputs)
                probs = F.softmax(logits, dim=1)
                
                all_probs.append(probs.cpu())
                all_labels.append(labels)
                
        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        # Expected Calibration Error (ECE)
        ece = self._compute_ece(all_probs, all_labels, n_bins)
        
        # Maximum Calibration Error (MCE)
        mce = self._compute_mce(all_probs, all_labels, n_bins)
        
        # Brier Score
        brier = self._compute_brier_score(all_probs, all_labels)
        
        return {
            "ece": ece,
            "mce": mce,
            "brier_score": brier
        }
        
    def _compute_ece(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int
    ) -> float:
        """Compute Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find samples in bin
            confidences = np.max(probs, axis=1)
            bin_mask = (confidences >= bin_lower) & (confidences < bin_upper)
            
            if bin_mask.sum() > 0:
                bin_confidence = confidences[bin_mask].mean()
                bin_accuracy = labels[bin_mask].mean()
                
                ece += bin_mask.sum() / len(labels) * abs(bin_accuracy - bin_confidence)
                
        return ece
        
    def _compute_mce(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int
    ) -> float:
        """Compute Maximum Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        max_error = 0.0
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            confidences = np.max(probs, axis=1)
            bin_mask = (confidences >= bin_lower) & (confidences < bin_upper)
            
            if bin_mask.sum() > 0:
                bin_accuracy = labels[bin_mask].mean()
                bin_confidence = confidences[bin_mask].mean()
                
                max_error = max(max_error, abs(bin_accuracy - bin_confidence))
                
        return max_error
        
    def _compute_brier_score(
        self,
        probs: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Compute Brier Score"""
        num_classes = probs.shape[1]
        
        # One-hot encode labels
        labels_onehot = np.zeros_like(probs)
        labels_onehot[np.arange(len(labels)), labels] = 1
        
        return np.mean((probs - labels_onehot) ** 2)
        
    def plot_reliability_diagram(
        self,
        dataloader: DataLoader,
        n_bins: int = 10,
        device: str = "cuda"
    ):
        """Plot reliability diagram"""
        import matplotlib.pyplot as plt
        
        self.model.eval()
        
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                logits = self.model(inputs)
                probs = F.softmax(logits, dim=1)
                
                all_probs.append(probs.cpu())
                all_labels.append(labels)
                
        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        # Compute calibration curve
        confidences = np.max(all_probs, axis=1)
        accuracies = all_labels == np.argmax(all_probs, axis=1)
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            accuracies, confidences, n_bins=n_bins
        )
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        ax.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Reliability Diagram')
        ax.legend()
        
        return fig


class MultiClassCalibrator:
    """Multi-class calibration"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.calibrators = {}
        
    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray
    ):
        """Fit one-vs-rest calibrators"""
        for c in range(self.num_classes):
            binary_labels = (labels == c).astype(int)
            
            calibrator = IsotonicRegression()
            calibrator.fit(probs[:, c], binary_labels)
            
            self.calibrators[c] = calibrator
            
    def calibrate(
        self,
        probs: np.ndarray
    ) -> np.ndarray:
        """Calibrate multi-class probabilities"""
        calibrated = np.zeros_like(probs)
        
        for c in range(self.num_classes):
            calibrated[:, c] = self.calibrators[c].calibrate(probs[:, c])
            
        # Normalize
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / row_sums
        
        return calibrated


class BayesianCalibration:
    """Bayesian calibration"""
    
    def __init__(self, model: nn.Module, num_samples: int = 100):
        self.model = model
        self.num_samples = num_samples
        
    def predict_with_uncertainty(
        self,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with uncertainty estimation"""
        self.model.train()
        
        predictions = []
        for _ in range(self.num_samples):
            with torch.no_grad():
                logits = self.model(inputs)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
                
        predictions = torch.stack(predictions)
        
        # Mean prediction
        mean_pred = predictions.mean(dim=0)
        
        # Uncertainty (entropy of mean)
        entropy = -torch.sum(
            mean_pred * torch.log(mean_pred + 1e-10),
            dim=1
        )
        
        # Variance
        variance = predictions.var(dim=0).mean(dim=1)
        
        self.model.eval()
        
        return mean_pred, variance


class CalibrationManager:
    """Manage multiple calibration methods"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.calibrators = {
            "temperature": TemperatureScaling(model),
            "isotonic": IsotonicRegression(),
            "platt": PlattScaling()
        }
        
    def calibrate_model(
        self,
        method: str = "temperature",
        **kwargs
    ):
        """Calibrate model using specified method"""
        if method == "temperature":
            self.calibrators["temperature"].fit(**kwargs)
        elif method == "isotonic":
            self.calibrators["isotonic"].fit(**kwargs)
        elif method == "platt":
            self.calibrators["platt"].fit(**kwargs)
            
    def get_calibrated_predictions(
        self,
        logits: torch.Tensor,
        method: str = "temperature"
    ) -> torch.Tensor:
        """Get calibrated predictions"""
        if method == "temperature":
            return F.softmax(
                self.calibrators["temperature"].calibrate(logits),
                dim=1
            )
        elif method == "isotonic":
            probs = F.softmax(logits, dim=1).numpy()
            return torch.tensor(
                self.calibrators["isotonic"].calibrate(probs)
            )
        else:
            return F.softmax(logits, dim=1)
