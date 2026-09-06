import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

@dataclass
class DriftAlert:
    """Drift detection alert"""
    feature_name: str
    drift_type: str
    severity: str
    p_value: float
    detected_at: str
    details: Dict[str, any]

class DataDriftDetector:
    """Detect data drift in features"""
    
    def __init__(self, reference_data: Optional[np.ndarray] = None):
        self.reference_data = reference_data
        self.reference_stats = None
        self.drift_history: List[DriftAlert] = []
        
        if reference_data is not None:
            self._compute_reference_stats()
            
    def _compute_reference_stats(self):
        """Compute reference statistics"""
        self.reference_stats = {
            "mean": np.mean(self.reference_data, axis=0),
            "std": np.std(self.reference_data, axis=0),
            "min": np.min(self.reference_data, axis=0),
            "max": np.max(self.reference_data, axis=0),
            "percentiles": {
                "25": np.percentile(self.reference_data, 25, axis=0),
                "50": np.percentile(self.reference_data, 50, axis=0),
                "75": np.percentile(self.reference_data, 75, axis=0)
            }
        }
        
    def set_reference(self, data: np.ndarray):
        """Set reference data"""
        self.reference_data = data
        self._compute_reference_stats()
        
    def detect_ks_drift(
        self,
        current_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.05
    ) -> List[DriftAlert]:
        """Detect drift using Kolmogorov-Smirnov test"""
        if self.reference_data is None:
            raise ValueError("Reference data not set")
            
        alerts = []
        
        for i in range(current_data.shape[1]):
            feature_name = feature_names[i] if feature_names else f"feature_{i}"
            
            # KS test
            ks_stat, p_value = stats.ks_2samp(
                self.reference_data[:, i],
                current_data[:, i]
            )
            
            if p_value < threshold:
                severity = "high" if p_value < 0.01 else "medium"
                
                alert = DriftAlert(
                    feature_name=feature_name,
                    drift_type="distribution",
                    severity=severity,
                    p_value=p_value,
                    detected_at=str(np.datetime64('now')),
                    details={
                        "ks_statistic": ks_stat,
                        "reference_mean": float(self.reference_stats["mean"][i]),
                        "current_mean": float(np.mean(current_data[:, i]))
                    }
                )
                alerts.append(alert)
                self.drift_history.append(alert)
                
        return alerts
        
    def detect_psi(
        self,
        current_data: np.ndarray,
        n_bins: int = 10,
        threshold: float = 0.25
    ) -> List[DriftAlert]:
        """Detect drift using Population Stability Index"""
        alerts = []
        
        for i in range(current_data.shape[1]):
            # Create bins from reference
            percentiles = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.percentile(self.reference_data[:, i], percentiles)
            
            # Compute histograms
            ref_hist, _ = np.histogram(self.reference_data[:, i], bins=bin_edges)
            cur_hist, _ = np.histogram(current_data[:, i], bins=bin_edges)
            
            # Normalize
            ref_hist = ref_hist / ref_hist.sum() + 1e-10
            cur_hist = cur_hist / cur_hist.sum() + 1e-10
            
            # Compute PSI
            psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))
            
            if psi > threshold:
                severity = "high" if psi > 0.5 else "medium"
                
                alert = DriftAlert(
                    feature_name=f"feature_{i}",
                    drift_type="population",
                    severity=severity,
                    p_value=psi,
                    detected_at=str(np.datetime64('now')),
                    details={
                        "psi": psi,
                        "threshold": threshold
                    }
                )
                alerts.append(alert)
                self.drift_history.append(alert)
                
        return alerts
        
    def detect_mmd_drift(
        self,
        current_data: np.ndarray,
        kernel: str = "rbf",
        threshold: float = 0.1
    ) -> DriftAlert:
        """Detect drift using Maximum Mean Discrepancy"""
        from sklearn.metrics.pairwise import rbf_kernel
        
        # Compute MMD
        if kernel == "rbf":
            K_rr = rbf_kernel(self.reference_data)
            K_cc = rbf_kernel(current_data)
            K_rc = rbf_kernel(self.reference_data, current_data)
            
            mmd = np.mean(K_rr) + np.mean(K_cc) - 2 * np.mean(K_rc)
        else:
            # Linear kernel
            mmd = (
                np.mean(self.reference_data @ self.reference_data.T) +
                np.mean(current_data @ current_data.T) -
                2 * np.mean(self.reference_data @ current_data.T)
            )
            
        drift_detected = mmd > threshold
        
        return DriftAlert(
            feature_name="all_features",
            drift_type="distribution",
            severity="high" if mmd > threshold * 2 else "medium",
            p_value=mmd,
            detected_at=str(np.datetime64('now')),
            details={"mmd": mmd, "threshold": threshold}
        )


class ConceptDriftDetector:
    """Detect concept drift"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.predictions = []
        self.labels = []
        
    def add_observation(
        self,
        prediction: int,
        label: int
    ):
        """Add prediction observation"""
        self.predictions.append(prediction)
        self.labels.append(label)
        
    def detect_ddm(
        self,
        warning_level: float = 2.0,
        drift_level: float = 3.0
    ) -> Optional[DriftAlert]:
        """Detect concept drift using DDM"""
        if len(self.predictions) < self.window_size:
            return None
            
        # Compute error rate
        errors = np.array(self.predictions) != np.array(self.labels)
        error_rate = np.mean(errors[-self.window_size:])
        
        # Compute std
        std = np.sqrt(error_rate * (1 - error_rate) / self.window_size)
        
        # DDM detector
        min_error = np.min(errors)
        min_std = np.sqrt(min_error * (1 - min_error) / self.window_size)
        
        if error_rate + std > min_error + drift_level * min_std:
            return DriftAlert(
                feature_name="predictions",
                drift_type="concept",
                severity="high",
                p_value=error_rate,
                detected_at=str(np.datetime64('now')),
                details={
                    "error_rate": error_rate,
                    "threshold": min_error + drift_level * min_std
                }
            )
        elif error_rate + std > min_error + warning_level * min_std:
            return DriftAlert(
                feature_name="predictions",
                drift_type="concept",
                severity="medium",
                p_value=error_rate,
                detected_at=str(np.datetime64('now')),
                details={
                    "error_rate": error_rate,
                    "threshold": min_error + warning_level * min_std
                }
            )
            
        return None
        
    def detect_eddm(
        self,
        warning_level: float = 0.95,
        drift_level: float = 0.90
    ) -> Optional[DriftAlert]:
        """Detect concept drift using EDDM"""
        if len(self.predictions) < self.window_size:
            return None
            
        # Compute errors
        errors = np.array(self.predictions) != np.array(self.labels)
        
        # Compute distances between errors
        error_indices = np.where(errors)[0]
        if len(error_indices) < 2:
            return None
            
        distances = np.diff(error_indices)
        
        if len(distances) < 10:
            return None
            
        # Compute mean and std
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # Check for drift
        if len(distances) > 30:
            recent_mean = np.mean(distances[-30:])
            
            if recent_mean < mean_dist * drift_level:
                return DriftAlert(
                    feature_name="errors",
                    drift_type="concept",
                    severity="high",
                    p_value=recent_mean / mean_dist,
                    detected_at=str(np.datetime64('now')),
                    details={
                        "mean_distance": mean_dist,
                        "recent_mean": recent_mean
                    }
                )
                
        return None


class PerformanceMonitor:
    """Monitor model performance"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = defaultdict(list)
        
    def add_metric(
        self,
        metric_name: str,
        value: float
    ):
        """Add metric value"""
        self.metrics_history[metric_name].append(value)
        
    def compute_rolling_metrics(
        self,
        metric_name: str
    ) -> Dict[str, float]:
        """Compute rolling metrics"""
        values = self.metrics_history[metric_name]
        
        if not values:
            return {}
            
        recent = values[-self.window_size:]
        
        return {
            "mean": np.mean(recent),
            "std": np.std(recent),
            "min": np.min(recent),
            "max": np.max(recent),
            "trend": np.polyfit(range(len(recent)), recent, 1)[0]
        }
        
    def detect_anomaly(
        self,
        metric_name: str,
        threshold: float = 2.0
    ) -> bool:
        """Detect metric anomaly"""
        values = self.metrics_history[metric_name]
        
        if len(values) < self.window_size:
            return False
            
        recent = values[-self.window_size:]
        mean = np.mean(recent[:-1])
        std = np.std(recent[:-1])
        
        current = recent[-1]
        
        if std > 0:
            z_score = abs(current - mean) / std
            return z_score > threshold
            
        return False


class AlertManager:
    """Manage drift alerts"""
    
    def __init__(self):
        self.alerts: List[DriftAlert] = []
        self.alert_callbacks = []
        
    def add_alert(self, alert: DriftAlert):
        """Add new alert"""
        self.alerts.append(alert)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            callback(alert)
            
    def register_callback(self, callback):
        """Register alert callback"""
        self.alert_callbacks.append(callback)
        
    def get_alerts(
        self,
        severity: Optional[str] = None,
        drift_type: Optional[str] = None
    ) -> List[DriftAlert]:
        """Get alerts with filters"""
        filtered = self.alerts
        
        if severity:
            filtered = [a for a in filtered if a.severity == severity]
            
        if drift_type:
            filtered = [a for a in filtered if a.drift_type == drift_type]
            
        return filtered
        
    def get_summary(self) -> Dict[str, int]:
        """Get alert summary"""
        return {
            "total": len(self.alerts),
            "high": len([a for a in self.alerts if a.severity == "high"]),
            "medium": len([a for a in self.alerts if a.severity == "medium"]),
            "low": len([a for a in self.alerts if a.severity == "low"])
        }
        
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts.clear()
