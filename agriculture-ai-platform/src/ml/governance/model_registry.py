import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import hashlib
from pathlib import Path

class ModelStage(Enum):
    """Model deployment stages"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

@dataclass
class ModelMetadata:
    """Model metadata"""
    model_id: str
    name: str
    version: str
    stage: ModelStage
    description: str
    created_at: datetime
    updated_at: datetime
    author: str
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)

class ModelRegistry:
    """Model registry for version control"""
    
    def __init__(self, registry_path: str = "model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.models: Dict[str, Dict[str, ModelMetadata]] = {}
        
    def register_model(
        self,
        model: nn.Module,
        name: str,
        version: str,
        description: str,
        author: str,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None
    ) -> ModelMetadata:
        """Register a new model"""
        model_id = f"{name}_v{version}"
        
        # Save model
        model_path = self.registry_path / f"{model_id}.pth"
        torch.save(model.state_dict(), model_path)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            stage=ModelStage.DEVELOPMENT,
            description=description,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            author=author,
            tags=tags or [],
            metrics=metrics or {},
            artifacts={"model_path": str(model_path)}
        )
        
        # Store metadata
        if name not in self.models:
            self.models[name] = {}
        self.models[name][version] = metadata
        
        # Save metadata
        self._save_metadata(metadata)
        
        return metadata
        
    def _save_metadata(self, metadata: ModelMetadata):
        """Save metadata to disk"""
        metadata_path = self.registry_path / f"{metadata.model_id}_metadata.json"
        
        data = {
            "model_id": metadata.model_id,
            "name": metadata.name,
            "version": metadata.version,
            "stage": metadata.stage.value,
            "description": metadata.description,
            "created_at": metadata.created_at.isoformat(),
            "updated_at": metadata.updated_at.isoformat(),
            "author": metadata.author,
            "tags": metadata.tags,
            "metrics": metadata.metrics,
            "parameters": metadata.parameters,
            "artifacts": metadata.artifacts
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def promote_model(
        self,
        name: str,
        version: str,
        new_stage: ModelStage
    ):
        """Promote model to new stage"""
        if name in self.models and version in self.models[name]:
            metadata = self.models[name][version]
            metadata.stage = new_stage
            metadata.updated_at = datetime.now()
            self._save_metadata(metadata)
            
    def get_model(
        self,
        name: str,
        version: str
    ) -> Optional[nn.Module]:
        """Load model from registry"""
        if name not in self.models or version not in self.models[name]:
            return None
            
        metadata = self.models[name][version]
        model_path = metadata.artifacts.get("model_path")
        
        if model_path and Path(model_path).exists():
            # Load model
            state_dict = torch.load(model_path)
            return state_dict
            
        return None
        
    def get_production_model(
        self,
        name: str
    ) -> Optional[nn.Module]:
        """Get production model"""
        if name not in self.models:
            return None
            
        for version, metadata in self.models[name].items():
            if metadata.stage == ModelStage.PRODUCTION:
                return self.get_model(name, version)
                
        return None
        
    def compare_versions(
        self,
        name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Compare two model versions"""
        if name not in self.models:
            return {}
            
        meta1 = self.models[name].get(version1)
        meta2 = self.models[name].get(version2)
        
        if not meta1 or not meta2:
            return {}
            
        return {
            "version1": {
                "stage": meta1.stage.value,
                "metrics": meta1.metrics,
                "created_at": meta1.created_at.isoformat()
            },
            "version2": {
                "stage": meta2.stage.value,
                "metrics": meta2.metrics,
                "created_at": meta2.created_at.isoformat()
            },
            "differences": {
                metric: meta2.metrics.get(metric, 0) - meta1.metrics.get(metric, 0)
                for metric in set(meta1.metrics.keys()) | set(meta2.metrics.keys())
            }
        }
        
    def list_models(
        self,
        stage: Optional[ModelStage] = None
    ) -> List[ModelMetadata]:
        """List all models"""
        all_models = []
        
        for name, versions in self.models.items():
            for version, metadata in versions.items():
                if stage is None or metadata.stage == stage:
                    all_models.append(metadata)
                    
        return all_models
        
    def delete_model(self, name: str, version: str):
        """Delete model from registry"""
        if name in self.models and version in self.models[name]:
            metadata = self.models[name][version]
            
            # Delete model file
            model_path = metadata.artifacts.get("model_path")
            if model_path and Path(model_path).exists():
                Path(model_path).unlink()
                
            # Delete metadata
            metadata_path = self.registry_path / f"{metadata.model_id}_metadata.json"
            if metadata_path.exists():
                metadata_path.unlink()
                
            # Remove from registry
            del self.models[name][version]


class ModelAuditor:
    """Audit model changes"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.audit_log: List[Dict[str, Any]] = []
        
    def log_action(
        self,
        action: str,
        model_name: str,
        version: str,
        user: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log an action"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "model_name": model_name,
            "version": version,
            "user": user,
            "details": details or {}
        }
        
        self.audit_log.append(entry)
        
    def get_audit_trail(
        self,
        model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get audit trail"""
        if model_name:
            return [e for e in self.audit_log if e["model_name"] == model_name]
        return self.audit_log
        
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report"""
        return {
            "total_actions": len(self.audit_log),
            "unique_models": len(set(e["model_name"] for e in self.audit_log)),
            "unique_users": len(set(e["user"] for e in self.audit_log)),
            "recent_actions": self.audit_log[-10:] if self.audit_log else []
        }


class ABTestManager:
    """A/B testing for models"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.experiments: Dict[str, Dict[str, Any]] = {}
        
    def create_experiment(
        self,
        name: str,
        model_a_name: str,
        model_a_version: str,
        model_b_name: str,
        model_b_version: str,
        traffic_split: float = 0.5
    ) -> Dict[str, Any]:
        """Create A/B test experiment"""
        experiment = {
            "name": name,
            "model_a": {
                "name": model_a_name,
                "version": model_a_version
            },
            "model_b": {
                "name": model_b_name,
                "version": model_b_version
            },
            "traffic_split": traffic_split,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "metrics_a": {"predictions": 0, "correct": 0},
            "metrics_b": {"predictions": 0, "correct": 0}
        }
        
        self.experiments[name] = experiment
        return experiment
        
    def route_request(
        self,
        experiment_name: str,
        request_id: str
    ) -> str:
        """Route request to model A or B"""
        experiment = self.experiments.get(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment {experiment_name} not found")
            
        # Deterministic routing based on request_id
        hash_value = hashlib.md5(request_id.encode()).hexdigest()
        route = "a" if (int(hash_value, 16) % 100) / 100 < experiment["traffic_split"] else "b"
        
        return route
        
    def log_prediction(
        self,
        experiment_name: str,
        model: str,
        correct: bool
    ):
        """Log prediction result"""
        experiment = self.experiments.get(experiment_name)
        if not experiment:
            return
            
        if model == "a":
            experiment["metrics_a"]["predictions"] += 1
            if correct:
                experiment["metrics_a"]["correct"] += 1
        else:
            experiment["metrics_b"]["predictions"] += 1
            if correct:
                experiment["metrics_b"]["correct"] += 1
                
    def get_results(
        self,
        experiment_name: str
    ) -> Dict[str, Any]:
        """Get experiment results"""
        experiment = self.experiments.get(experiment_name)
        if not experiment:
            return {}
            
        metrics_a = experiment["metrics_a"]
        metrics_b = experiment["metrics_b"]
        
        accuracy_a = (
            metrics_a["correct"] / metrics_a["predictions"]
            if metrics_a["predictions"] > 0
            else 0
        )
        accuracy_b = (
            metrics_b["correct"] / metrics_b["predictions"]
            if metrics_b["predictions"] > 0
            else 0
        )
        
        # Statistical significance (simplified)
        import scipy.stats as stats
        
        if metrics_a["predictions"] > 0 and metrics_b["predictions"] > 0:
            # Chi-square test
            contingency_table = [
                [metrics_a["correct"], metrics_a["predictions"] - metrics_a["correct"]],
                [metrics_b["correct"], metrics_b["predictions"] - metrics_b["correct"]]
            ]
            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
            significant = p_value < 0.05
        else:
            p_value = 1.0
            significant = False
            
        return {
            "experiment": experiment_name,
            "model_a": {
                "accuracy": accuracy_a,
                "predictions": metrics_a["predictions"]
            },
            "model_b": {
                "accuracy": accuracy_b,
                "predictions": metrics_b["predictions"]
            },
            "winner": "a" if accuracy_a > accuracy_b else "b",
            "p_value": p_value,
            "significant": significant
        }
        
    def conclude_experiment(
        self,
        experiment_name: str
    ) -> Dict[str, Any]:
        """Conclude experiment and return winner"""
        results = self.get_results(experiment_name)
        
        if results:
            self.experiments[experiment_name]["status"] = "concluded"
            
            # Promote winner to production
            winner = results["winner"]
            if winner == "a":
                model_info = self.experiments[experiment_name]["model_a"]
            else:
                model_info = self.experiments[experiment_name]["model_b"]
                
            self.registry.promote_model(
                model_info["name"],
                model_info["version"],
                ModelStage.PRODUCTION
            )
            
        return results


class ModelGovernance:
    """Model governance and compliance"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.auditor = ModelAuditor(registry)
        self.ab_test_manager = ABTestManager(registry)
        
    def approve_model(
        self,
        name: str,
        version: str,
        approver: str
    ):
        """Approve model for production"""
        self.registry.promote_model(name, version, ModelStage.PRODUCTION)
        self.auditor.log_action(
            "approve",
            name,
            version,
            approver
        )
        
    def rollback_model(
        self,
        name: str,
        version: str,
        reason: str,
        user: str
    ):
        """Rollback model to previous version"""
        self.registry.promote_model(name, version, ModelStage.ARCHIVED)
        self.auditor.log_action(
            "rollback",
            name,
            version,
            user,
            {"reason": reason}
        )
        
    def generate_governance_report(self) -> Dict[str, Any]:
        """Generate governance report"""
        models = self.registry.list_models()
        
        production_models = [m for m in models if m.stage == ModelStage.PRODUCTION]
        staging_models = [m for m in models if m.stage == ModelStage.STAGING]
        
        return {
            "total_models": len(models),
            "production_models": len(production_models),
            "staging_models": len(staging_models),
            "audit_summary": self.auditor.generate_compliance_report(),
            "active_experiments": len([
                e for e in self.ab_test_manager.experiments.values()
                if e["status"] == "active"
            ])
        }
