#!/usr/bin/env python3
"""
Generate system report
"""

import sys
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def generate_report():
    report = {
        "timestamp": datetime.now().isoformat(),
        "system": {},
        "ml": {},
        "database": {},
        "usage": {}
    }
    
    # System info
    import platform
    report["system"] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor()
    }
    
    # ML info
    try:
        import torch
        report["ml"] = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    except ImportError:
        report["ml"] = {"error": "PyTorch not installed"}
    
    # Database stats
    try:
        from sqlalchemy import create_engine, text
        from src.api.app.config import settings
        
        engine = create_engine(settings.DATABASE_URL.replace("+asyncpg", ""))
        with engine.connect() as conn:
            # Count users
            result = conn.execute(text("SELECT COUNT(*) FROM users"))
            user_count = result.scalar()
            
            # Count diagnoses
            result = conn.execute(text("SELECT COUNT(*) FROM diagnoses"))
            diagnosis_count = result.scalar()
            
            report["database"] = {
                "users": user_count,
                "diagnoses": diagnosis_count,
                "status": "connected"
            }
    except Exception as e:
        report["database"] = {"status": "error", "message": str(e)}
    
    # Model info
    try:
        from src.ml.models.classification.disease_classifier import DiseaseClassifier
        
        model = DiseaseClassifier(num_classes=10)
        param_count = sum(p.numel() for p in model.parameters())
        
        report["ml"]["model"] = {
            "name": "DiseaseClassifier",
            "parameters": param_count,
            "size_mb": param_count * 4 / (1024 * 1024)
        }
    except Exception as e:
        report["ml"]["model"] = {"error": str(e)}
    
    # Save report
    output_path = Path("reports") / f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Report generated: {output_path}")
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    generate_report()
