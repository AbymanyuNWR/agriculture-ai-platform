import numpy as np
from typing import Dict, List

class PredictionPostprocessor:
    def __init__(self):
        self.class_names = [
            "healthy",
            "blast",
            "brown_spot",
            "leaf_blight",
            "bacterial_blight",
            "tungro",
            "grassy_stunt",
            "ragged_stunt",
            "rice_grassy_virus",
            "rice_tungro_bacilliform"
        ]
        
        self.severity_thresholds = {
            "high": 0.85,
            "medium": 0.65,
            "low": 0.0
        }
    
    def process(
        self,
        probabilities: np.ndarray,
        crop_type: str = None,
        confidence_threshold: float = 0.5
    ) -> Dict:
        # Get top prediction
        top_idx = np.argmax(probabilities)
        top_prob = probabilities[0][top_idx]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(probabilities[0])[-3:][::-1]
        top_3 = [
            {
                "class": self.class_names[idx],
                "confidence": float(probabilities[0][idx])
            }
            for idx in top_3_idx
        ]
        
        # Determine severity
        severity = self.determine_severity(top_prob)
        
        return {
            "diagnosis": self.class_names[top_idx],
            "confidence": float(top_prob),
            "severity": severity,
            "top_3_predictions": top_3,
            "crop_type": crop_type,
            "is_healthy": self.class_names[top_idx] == "healthy"
        }
    
    def determine_severity(self, confidence: float) -> str:
        for level, threshold in self.severity_thresholds.items():
            if confidence >= threshold:
                return level
        return "low"
    
    def calibrate_confidence(
        self,
        probabilities: np.ndarray,
        temperature: float = 1.5
    ) -> np.ndarray:
        # Temperature scaling for calibration
        scaled_logits = np.log(probabilities + 1e-10) / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        calibrated = exp_logits / np.sum(exp_logits)
        
        return calibrated
