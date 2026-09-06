#!/usr/bin/env python3
"""
Register trained model in MLflow model registry
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mlflow
import torch
import json

from src.ml.models.classification.disease_classifier import DiseaseClassifier
from src.api.app.config import settings

def register_model(model_path, model_name, version, metrics_path=None):
    """Register model in MLflow"""
    
    # Set tracking URI
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = DiseaseClassifier(num_classes=10)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load metrics if provided
    metrics = {}
    if metrics_path and Path(metrics_path).exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"{model_name}_v{version}"):
        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("version", version)
        mlflow.log_param("architecture", "ResNet-50")
        mlflow.log_param("num_classes", 10)
        mlflow.log_param("training_date", datetime.now().isoformat())
        
        # Log metrics
        if metrics:
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
        
        # Log model
        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name=model_name
        )
        
        # Log model architecture
        mlflow.log_text(str(model), "model_architecture.txt")
        
        print(f"Model registered successfully!")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Model name: {model_name}")
        print(f"Version: {version}")

def main():
    parser = argparse.ArgumentParser(description='Register model in MLflow')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--model-name', type=str, default='disease_classifier',
                        help='Name for the model')
    parser.add_argument('--version', type=str, default='1.0.0',
                        help='Model version')
    parser.add_argument('--metrics-path', type=str, default=None,
                        help='Path to metrics JSON file')
    
    args = parser.parse_args()
    
    register_model(
        model_path=args.model_path,
        model_name=args.model_name,
        version=args.version,
        metrics_path=args.metrics_path
    )

if __name__ == '__main__':
    main()
