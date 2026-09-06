#!/usr/bin/env python3
"""
Monitor model performance and detect degradation
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prometheus_client import Gauge, start_http_server
import mlflow
import numpy as np

class ModelPerformanceMonitor:
    def __init__(self):
        self.accuracy_gauge = Gauge('model_accuracy', 'Current model accuracy')
        self.latency_gauge = Gauge('model_latency', 'Model inference latency')
        self.drift_score_gauge = Gauge('data_drift_score', 'Data drift detection score')
        
    def check_accuracy(self, threshold=0.85):
        """Check if model accuracy is below threshold"""
        # Get latest metrics from MLflow
        client = mlflow.tracking.MlflowClient()
        
        experiment = client.get_experiment_by_name("agriculture_ai")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if runs:
                accuracy = runs[0].data.metrics.get("accuracy", 0)
                self.accuracy_gauge.set(accuracy)
                
                if accuracy < threshold:
                    print(f"⚠️ Model accuracy ({accuracy:.4f}) is below threshold ({threshold})")
                    return False
        
        return True
    
    def check_latency(self, threshold_ms=500):
        """Check if model latency is above threshold"""
        # Simulate latency check
        latency_ms = np.random.normal(100, 20)
        self.latency_gauge.set(latency_ms)
        
        if latency_ms > threshold_ms:
            print(f"⚠️ Model latency ({latency_ms:.2f}ms) is above threshold ({threshold_ms}ms)")
            return False
        
        return True
    
    def check_data_drift(self, threshold=0.1):
        """Check for data drift"""
        # Simulate drift detection
        drift_score = np.random.uniform(0, 0.2)
        self.drift_score_gauge.set(drift_score)
        
        if drift_score > threshold:
            print(f"⚠️ Data drift detected (score: {drift_score:.4f})")
            return False
        
        return True
    
    def run_checks(self):
        """Run all performance checks"""
        print("Running model performance checks...")
        
        checks = [
            ("Accuracy", self.check_accuracy),
            ("Latency", self.check_latency),
            ("Data Drift", self.check_data_drift),
        ]
        
        all_passed = True
        
        for name, check_func in checks:
            passed = check_func()
            status = "✓" if passed else "✗"
            print(f"{status} {name} check {'passed' if passed else 'failed'}")
            if not passed:
                all_passed = False
        
        return all_passed

def main():
    # Start Prometheus metrics server
    start_http_server(8001)
    
    monitor = ModelPerformanceMonitor()
    
    # Run checks
    passed = monitor.run_checks()
    
    if not passed:
        print("\n⚠️ Some checks failed. Consider retraining the model.")
        sys.exit(1)
    else:
        print("\n✓ All checks passed.")
        sys.exit(0)

if __name__ == '__main__':
    main()
