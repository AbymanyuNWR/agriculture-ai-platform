#!/usr/bin/env python3
"""
Monitor data drift and trigger retraining if needed
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scipy import stats
import pandas as pd

class DataDriftMonitor:
    def __init__(self, baseline_data_path=None):
        self.baseline_data = self.load_baseline(baseline_data_path)
        
    def load_baseline(self, path):
        """Load baseline data for comparison"""
        if path and Path(path).exists():
            return pd.read_csv(path)
        
        # Generate synthetic baseline
        return pd.DataFrame({
            'temperature': np.random.normal(28, 5, 1000),
            'humidity': np.random.normal(75, 10, 1000),
            'soil_moisture': np.random.normal(60, 15, 1000),
            'nitrogen': np.random.normal(50, 10, 1000),
            'phosphorus': np.random.normal(30, 8, 1000),
            'potassium': np.random.normal(40, 12, 1000),
        })
    
    def detect_drift(self, current_data, threshold=0.05):
        """Detect drift using Kolmogorov-Smirnov test"""
        drift_results = {}
        
        for column in current_data.columns:
            if column in self.baseline_data.columns:
                # KS test
                stat, p_value = stats.ks_2samp(
                    self.baseline_data[column],
                    current_data[column]
                )
                
                drift_detected = p_value < threshold
                drift_results[column] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'drift_detected': drift_detected
                }
        
        return drift_results
    
    def calculate_drift_score(self, drift_results):
        """Calculate overall drift score"""
        if not drift_results:
            return 0
        
        drift_count = sum(1 for r in drift_results.values() if r['drift_detected'])
        return drift_count / len(drift_results)
    
    def generate_report(self, drift_results, output_path=None):
        """Generate drift report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'drift_results': drift_results,
            'summary': {
                'total_features': len(drift_results),
                'drifted_features': sum(1 for r in drift_results.values() if r['drift_detected']),
                'drift_score': self.calculate_drift_score(drift_results)
            }
        }
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report

def main():
    # Generate synthetic current data (simulating drift)
    current_data = pd.DataFrame({
        'temperature': np.random.normal(30, 6, 100),  # Slightly different
        'humidity': np.random.normal(80, 12, 100),   # Slightly different
        'soil_moisture': np.random.normal(55, 18, 100),
        'nitrogen': np.random.normal(45, 12, 100),
        'phosphorus': np.random.normal(32, 9, 100),
        'potassium': np.random.normal(38, 14, 100),
    })
    
    monitor = DataDriftMonitor()
    
    print("Checking for data drift...")
    drift_results = monitor.detect_drift(current_data)
    
    # Print results
    print("\nDrift Detection Results:")
    print("-" * 50)
    
    for feature, result in drift_results.items():
        status = "⚠️ DRIFT" if result['drift_detected'] else "✓ OK"
        print(f"{feature:20} | p-value: {result['p_value']:.4f} | {status}")
    
    # Calculate overall score
    drift_score = monitor.calculate_drift_score(drift_results)
    print(f"\nOverall drift score: {drift_score:.4f}")
    
    if drift_score > 0.3:
        print("\n⚠️ Significant drift detected. Consider retraining the model.")
        sys.exit(1)
    else:
        print("\n✓ No significant drift detected.")
        sys.exit(0)

if __name__ == '__main__':
    main()
