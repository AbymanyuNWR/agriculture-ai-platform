import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from scipy import stats
import json

from src.ml.config import MonitoringConfig

class PerformanceMonitor:
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        self.baseline_metrics = None
        self.metrics_history = []
        
    def load_baseline(self, metrics_path: str):
        """Load baseline metrics"""
        with open(metrics_path) as f:
            self.baseline_metrics = json.load(f)
        
        print(f"Baseline metrics loaded: {metrics_path}")
    
    def check_performance(self, current_metrics: Dict) -> Dict:
        """Check if current performance meets baseline"""
        if self.baseline_metrics is None:
            return {'status': 'no_baseline', 'message': 'No baseline metrics loaded'}
        
        results = {
            'status': 'ok',
            'degraded_metrics': [],
            'improved_metrics': [],
            'details': {}
        }
        
        for metric_name, baseline_value in self.baseline_metrics.items():
            if metric_name in current_metrics:
                current_value = current_metrics[metric_name]
                threshold = self.config.alert_threshold
                
                # Check if metric has degraded
                if isinstance(baseline_value, (int, float)) and isinstance(current_value, (int, float)):
                    degradation = (baseline_value - current_value) / baseline_value
                    
                    results['details'][metric_name] = {
                        'baseline': baseline_value,
                        'current': current_value,
                        'degradation': degradation
                    }
                    
                    if degradation > threshold:
                        results['status'] = 'degraded'
                        results['degraded_metrics'].append({
                            'metric': metric_name,
                            'baseline': baseline_value,
                            'current': current_value,
                            'degradation': degradation
                        })
                    elif degradation < -threshold:
                        results['improved_metrics'].append({
                            'metric': metric_name,
                            'baseline': baseline_value,
                            'current': current_value,
                            'improvement': -degradation
                        })
        
        # Store in history
        self.metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': current_metrics,
            'status': results['status']
        })
        
        return results
    
    def detect_anomalies(self, metrics_series: List[Dict], window_size: int = 10) -> Dict:
        """Detect anomalies in metrics over time"""
        if len(metrics_series) < window_size:
            return {'anomalies': [], 'message': 'Not enough data points'}
        
        anomalies = []
        
        # Extract metric values
        metric_names = set()
        for m in metrics_series:
            metric_names.update(m.keys())
        
        for metric_name in metric_names:
            values = [m.get(metric_name, 0) for m in metrics_series]
            
            # Calculate rolling statistics
            values_series = pd.Series(values)
            rolling_mean = values_series.rolling(window=window_size).mean()
            rolling_std = values_series.rolling(window=window_size).std()
            
            # Detect anomalies using z-score
            for i in range(window_size, len(values)):
                if rolling_std.iloc[i] > 0:
                    z_score = (values[i] - rolling_mean.iloc[i-1]) / rolling_std.iloc[i-1]
                    
                    if abs(z_score) > 3:  # 3 sigma rule
                        anomalies.append({
                            'metric': metric_name,
                            'index': i,
                            'value': values[i],
                            'expected_mean': rolling_mean.iloc[i-1],
                            'z_score': z_score,
                            'timestamp': datetime.now().isoformat()
                        })
        
        return {'anomalies': anomalies}
    
    def calculate_trend(self, metrics_series: List[Dict], metric_name: str) -> Dict:
        """Calculate trend for a specific metric"""
        values = [m.get(metric_name, 0) for m in metrics_series]
        
        if len(values) < 2:
            return {'trend': 'insufficient_data'}
        
        # Linear regression
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Determine trend
        if p_value > 0.05:
            trend = 'stable'
        elif slope > 0:
            trend = 'improving'
        else:
            trend = 'degrading'
        
        return {
            'trend': trend,
            'slope': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'current_value': values[-1],
            'predicted_next': slope * len(values) + intercept
        }
    
    def generate_health_report(self) -> Dict:
        """Generate system health report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_checks': len(self.metrics_history),
            'status_counts': {},
            'recent_metrics': {},
            'trends': {}
        }
        
        # Count statuses
        for entry in self.metrics_history:
            status = entry['status']
            report['status_counts'][status] = report['status_counts'].get(status, 0) + 1
        
        # Get recent metrics (last 10)
        if self.metrics_history:
            recent = self.metrics_history[-10:]
            report['recent_metrics'] = recent[-1]['metrics'] if recent else {}
            
            # Calculate trends for each metric
            all_metrics = [entry['metrics'] for entry in self.metrics_history]
            metric_names = set()
            for m in all_metrics:
                metric_names.update(m.keys())
            
            for metric_name in metric_names:
                trend_data = self.calculate_trend(all_metrics, metric_name)
                report['trends'][metric_name] = trend_data
        
        return report
    
    def save_report(self, report: Dict, output_path: str):
        """Save health report"""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Health report saved to {output_path}")
    
    def should_alert(self, check_result: Dict) -> bool:
        """Determine if alert should be sent"""
        if check_result['status'] == 'degraded':
            # Alert if accuracy drops more than 5%
            for metric in check_result.get('degraded_metrics', []):
                if metric['metric'] == 'accuracy' and metric['degradation'] > 0.05:
                    return True
        
        return False
    
    def format_alert_message(self, check_result: Dict) -> str:
        """Format alert message"""
        if check_result['status'] != 'degraded':
            return ""
        
        message_parts = ["⚠️ Model Performance Degradation Detected\n"]
        
        for metric in check_result.get('degraded_metrics', []):
            message_parts.append(
                f"- {metric['metric']}: {metric['baseline']:.4f} → {metric['current']:.4f} "
                f"({metric['degradation']*100:.1f}% degradation)"
            )
        
        message_parts.append("\nPlease check model performance and consider retraining.")
        
        return "\n".join(message_parts)
