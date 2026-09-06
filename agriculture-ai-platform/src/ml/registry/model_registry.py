import mlflow
import mlflow.pytorch
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime
import torch

from src.api.app.config import settings

class ModelRegistry:
    def __init__(self, tracking_uri: str = None):
        self.tracking_uri = tracking_uri or settings.MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
    def register_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        version: str,
        metrics: Dict = None,
        tags: Dict = None,
        artifact_path: str = "model"
    ) -> str:
        """Register a new model version"""
        with mlflow.start_run(run_name=f"{model_name}_v{version}"):
            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("version", version)
            mlflow.log_param("registration_time", datetime.now().isoformat())
            
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            
            # Log metrics
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
            
            # Log model
            mlflow.pytorch.log_model(
                model,
                artifact_path,
                registered_model_name=model_name
            )
            
            run_id = mlflow.active_run().info.run_id
            
            print(f"Model registered: {model_name} v{version}")
            print(f"Run ID: {run_id}")
            
            return run_id
    
    def get_model_versions(self, model_name: str) -> List[Dict]:
        """Get all versions of a model"""
        client = mlflow.tracking.MlflowClient()
        
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            return [
                {
                    'version': v.version,
                    'run_id': v.run_id,
                    'status': v.status,
                    'creation_timestamp': v.creation_timestamp
                }
                for v in versions
            ]
        except Exception as e:
            print(f"Error getting model versions: {e}")
            return []
    
    def get_latest_version(self, model_name: str, stage: str = "Production") -> Optional[Dict]:
        """Get the latest version of a model in a specific stage"""
        client = mlflow.tracking.MlflowClient()
        
        try:
            versions = client.search_model_versions(
                f"name='{model_name}' AND stage='{stage}'"
            )
            
            if versions:
                latest = max(versions, key=lambda v: v.creation_timestamp)
                return {
                    'version': latest.version,
                    'run_id': latest.run_id,
                    'status': latest.status
                }
        except Exception as e:
            print(f"Error getting latest version: {e}")
        
        return None
    
    def load_model(self, model_name: str, version: str = None) -> torch.nn.Module:
        """Load a registered model"""
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/latest"
        
        try:
            model = mlflow.pytorch.load_model(model_uri)
            print(f"Model loaded: {model_name} v{version or 'latest'}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def transition_stage(self, model_name: str, version: str, new_stage: str):
        """Transition model to a new stage"""
        client = mlflow.tracking.MlflowClient()
        
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=new_stage
            )
            print(f"Model {model_name} v{version} transitioned to {new_stage}")
        except Exception as e:
            print(f"Error transitioning model: {e}")
    
    def delete_model(self, model_name: str, version: str):
        """Delete a model version"""
        client = mlflow.tracking.MlflowClient()
        
        try:
            client.delete_model_version(name=model_name, version=version)
            print(f"Model {model_name} v{version} deleted")
        except Exception as e:
            print(f"Error deleting model: {e}")
    
    def compare_versions(self, model_name: str, version1: str, version2: str) -> Dict:
        """Compare two model versions"""
        client = mlflow.tracking.MlflowClient()
        
        try:
            run1 = client.get_run(mlflow.get_model_version(model_name, version1).run_id)
            run2 = client.get_run(mlflow.get_model_version(model_name, version2).run_id)
            
            comparison = {
                'version1': {
                    'version': version1,
                    'metrics': run1.data.metrics,
                    'params': run1.data.params
                },
                'version2': {
                    'version': version2,
                    'metrics': run2.data.metrics,
                    'params': run2.data.params
                },
                'metric_comparison': {}
            }
            
            # Compare metrics
            all_metrics = set(run1.data.metrics.keys()) | set(run2.data.metrics.keys())
            for metric in all_metrics:
                val1 = run1.data.metrics.get(metric)
                val2 = run2.data.metrics.get(metric)
                
                if val1 is not None and val2 is not None:
                    comparison['metric_comparison'][metric] = {
                        'version1': val1,
                        'version2': val2,
                        'difference': val2 - val1,
                        'percent_change': ((val2 - val1) / val1 * 100) if val1 != 0 else 0
                    }
            
            return comparison
            
        except Exception as e:
            print(f"Error comparing versions: {e}")
            return {}
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get detailed information about a model"""
        client = mlflow.tracking.MlflowClient()
        
        try:
            model = client.get_registered_model(model_name)
            versions = self.get_model_versions(model_name)
            
            return {
                'name': model.name,
                'description': model.description,
                'tags': model.tags,
                'versions': versions,
                'latest_versions': {
                    'Production': self.get_latest_version(model_name, 'Production'),
                    'Staging': self.get_latest_version(model_name, 'Staging')
                }
            }
        except Exception as e:
            print(f"Error getting model info: {e}")
            return {}
