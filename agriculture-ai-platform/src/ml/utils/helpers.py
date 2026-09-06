import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import hashlib
import json
from datetime import datetime
import random

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')

def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str
):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str
) -> Tuple[int, float]:
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def calculate_model_size(model: torch.nn.Module) -> float:
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def create_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """Create experiment directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    (experiment_dir / "plots").mkdir(exist_ok=True)
    
    return experiment_dir

def log_experiment(
    experiment_dir: Path,
    config: dict,
    metrics: dict,
    notes: str = ""
):
    """Log experiment details"""
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'metrics': metrics,
        'notes': notes
    }
    
    with open(experiment_dir / "experiment_log.json", 'w') as f:
        json.dump(log_data, f, indent=2)

def calculate_metrics_from_predictions(
    predictions: np.ndarray,
    labels: np.ndarray
) -> dict:
    """Calculate metrics from predictions and labels"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='macro', zero_division=0),
        'recall': recall_score(labels, predictions, average='macro', zero_division=0),
        'f1_score': f1_score(labels, predictions, average='macro', zero_division=0)
    }

def get_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Calculate class weights for imbalanced dataset"""
    from collections import Counter
    
    counter = Counter(labels)
    total = len(labels)
    
    weights = []
    for i in range(max(labels) + 1):
        weight = total / (len(counter) * counter[i])
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)

def format_time(seconds: float) -> str:
    """Format seconds to human readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def get_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def ensure_dir(path: str) -> Path:
    """Ensure directory exists"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_json(path: str) -> dict:
    """Load JSON file"""
    with open(path) as f:
        return json.load(f)

def save_json(data: dict, path: str):
    """Save data to JSON file"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def print_system_info():
    """Print system information"""
    import platform
    
    print("=" * 60)
    print("System Information")
    print("=" * 60)
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("=" * 60)
