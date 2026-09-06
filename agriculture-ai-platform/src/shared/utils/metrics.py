from prometheus_client import Counter, Histogram, Gauge, Summary
from functools import wraps
import time

# Metrics definitions
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

MODEL_PREDICTIONS = Counter(
    'model_predictions_total',
    'Total model predictions',
    ['model_name', 'prediction_class']
)

MODEL_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Model prediction latency',
    ['model_name']
)

DATA_DRIFT = Gauge(
    'data_drift_score',
    'Data drift detection score',
    ['feature_name']
)

ACTIVE_USERS = Gauge(
    'active_users',
    'Number of active users'
)

def track_metrics(endpoint_name):
    """Decorator to track API metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                status_code = 200
            except Exception as e:
                status_code = 500
                raise
            finally:
                duration = time.time() - start_time
                
                REQUEST_COUNT.labels(
                    method="POST",
                    endpoint=endpoint_name,
                    status_code=status_code
                ).inc()
                
                REQUEST_LATENCY.labels(
                    method="POST",
                    endpoint=endpoint_name
                ).observe(duration)
            
            return result
        return wrapper
    return decorator

def track_model_prediction(model_name):
    """Decorator to track model predictions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            duration = time.time() - start_time
            
            MODEL_PREDICTIONS.labels(
                model_name=model_name,
                prediction_class=result.get("diagnosis", "unknown")
            ).inc()
            
            MODEL_LATENCY.labels(
                model_name=model_name
            ).observe(duration)
            
            return result
        return wrapper
    return decorator
