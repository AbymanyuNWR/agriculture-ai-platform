import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib
from collections import defaultdict
import pickle

@dataclass
class Feature:
    """Feature definition"""
    name: str
    dtype: str
    description: str
    default_value: Any = None
    is_required: bool = True
    tags: List[str] = field(default_factory=list)

@dataclass
class FeatureValue:
    """Feature value with metadata"""
    feature_name: str
    value: Any
    entity_id: str
    timestamp: datetime
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

class FeatureStore:
    """Feature store for ML features"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.features: Dict[str, Feature] = {}
        self.feature_values: Dict[str, List[FeatureValue]] = defaultdict(list)
        self.entity_features: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.storage_path = storage_path
        
    def register_feature(self, feature: Feature):
        """Register a new feature"""
        self.features[feature.name] = feature
        
    def ingest_feature(
        self,
        feature_name: str,
        entity_id: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Ingest a feature value"""
        if feature_name not in self.features:
            raise ValueError(f"Feature {feature_name} not registered")
            
        feature_value = FeatureValue(
            feature_name=feature_name,
            value=value,
            entity_id=entity_id,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.feature_values[feature_name].append(feature_value)
        self.entity_features[entity_id][feature_name] = value
        
    def get_feature(
        self,
        entity_id: str,
        feature_name: str,
        as_of: Optional[datetime] = None
    ) -> Any:
        """Get feature value for entity"""
        if feature_name not in self.features:
            raise ValueError(f"Feature {feature_name} not registered")
            
        # Get latest value
        values = self.feature_values.get(feature_name, [])
        entity_values = [v for v in values if v.entity_id == entity_id]
        
        if not entity_values:
            return self.features[feature_name].default_value
            
        if as_of:
            entity_values = [v for v in entity_values if v.timestamp <= as_of]
            
        if not entity_values:
            return self.features[feature_name].default_value
            
        return entity_values[-1].value
        
    def get_entity_features(
        self,
        entity_id: str,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get all features for an entity"""
        if feature_names is None:
            feature_names = list(self.features.keys())
            
        features = {}
        for name in feature_names:
            features[name] = self.get_feature(entity_id, name)
            
        return features
        
    def get_batch_features(
        self,
        entity_ids: List[str],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get features for multiple entities"""
        return {
            entity_id: self.get_entity_features(entity_id, feature_names)
            for entity_id in entity_ids
        }
        
    def create_feature_vector(
        self,
        entity_id: str,
        feature_names: Optional[List[str]] = None
    ) -> torch.Tensor:
        """Create feature vector for ML model"""
        features = self.get_entity_features(entity_id, feature_names)
        
        # Convert to tensor
        values = []
        for name, value in features.items():
            if isinstance(value, (int, float)):
                values.append(value)
            elif isinstance(value, list):
                values.extend(value)
            elif isinstance(value, np.ndarray):
                values.extend(value.flatten().tolist())
                
        return torch.tensor(values, dtype=torch.float32)
        
    def create_batch_tensors(
        self,
        entity_ids: List[str],
        feature_names: Optional[List[str]] = None
    ) -> torch.Tensor:
        """Create batch of feature tensors"""
        vectors = [
            self.create_feature_vector(eid, feature_names)
            for eid in entity_ids
        ]
        return torch.stack(vectors)
        
    def compute_feature_statistics(
        self,
        feature_name: str
    ) -> Dict[str, float]:
        """Compute feature statistics"""
        values = [v.value for v in self.feature_values.get(feature_name, [])]
        
        if not values:
            return {}
            
        values = np.array(values, dtype=float)
        
        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values))
        }
        
    def save(self, path: str):
        """Save feature store to disk"""
        data = {
            "features": self.features,
            "feature_values": dict(self.feature_values),
            "entity_features": dict(self.entity_features)
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
            
    def load(self, path: str):
        """Load feature store from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        self.features = data["features"]
        self.feature_values = defaultdict(list, data["feature_values"])
        self.entity_features = defaultdict(dict, data["entity_features"])


class OnlineFeatureStore(FeatureStore):
    """Online feature store with Redis-like interface"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ttl: Dict[str, int] = {}  # Time-to-live in seconds
        
    def set_with_ttl(
        self,
        feature_name: str,
        entity_id: str,
        value: Any,
        ttl: int = 3600
    ):
        """Set feature with TTL"""
        self.ingest_feature(feature_name, entity_id, value)
        key = f"{entity_id}:{feature_name}"
        self.ttl[key] = ttl
        
    def get_realtime_features(
        self,
        entity_id: str,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Get real-time features"""
        features = {}
        for name in feature_names:
            value = self.get_feature(entity_id, name)
            if value is not None:
                features[name] = value
        return features


class FeaturePipeline:
    """Feature engineering pipeline"""
    
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self.transformations: Dict[str, callable] = {}
        
    def register_transformation(
        self,
        feature_name: str,
        transformation: callable
    ):
        """Register feature transformation"""
        self.transformations[feature_name] = transformation
        
    def compute_features(
        self,
        entity_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute features from raw data"""
        computed = {}
        
        for feature_name, transform in self.transformations.items():
            try:
                value = transform(input_data)
                computed[feature_name] = value
                
                # Store in feature store
                self.feature_store.ingest_feature(
                    feature_name, entity_id, value
                )
            except Exception as e:
                print(f"Error computing feature {feature_name}: {e}")
                
        return computed
        
    def compute_batch_features(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Compute features for batch of entities"""
        return [
            self.compute_features(entity["id"], entity)
            for entity in entities
        ]


class FeatureMonitor:
    """Monitor feature quality"""
    
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self.baseline_statistics: Dict[str, Dict[str, float]] = {}
        
    def set_baseline(self, feature_name: str):
        """Set baseline statistics"""
        stats = self.feature_store.compute_feature_statistics(feature_name)
        self.baseline_statistics[feature_name] = stats
        
    def detect_drift(
        self,
        feature_name: str,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Detect feature drift"""
        if feature_name not in self.baseline_statistics:
            return {"drifted": False, "reason": "No baseline"}
            
        current_stats = self.feature_store.compute_feature_statistics(feature_name)
        baseline = self.baseline_statistics[feature_name]
        
        # Check for drift
        drift_detected = False
        drift_metrics = {}
        
        for metric in ["mean", "std"]:
            if metric in current_stats and metric in baseline:
                baseline_val = baseline[metric]
                current_val = current_stats[metric]
                
                if baseline_val != 0:
                    relative_change = abs(current_val - baseline_val) / baseline_val
                    drift_metrics[metric] = relative_change
                    
                    if relative_change > threshold:
                        drift_detected = True
                        
        return {
            "drifted": drift_detected,
            "metrics": drift_metrics
        }
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate feature quality report"""
        report = {}
        
        for feature_name in self.feature_store.features:
            stats = self.feature_store.compute_feature_statistics(feature_name)
            drift = self.detect_drift(feature_name)
            
            report[feature_name] = {
                "statistics": stats,
                "drift": drift
            }
            
        return report
