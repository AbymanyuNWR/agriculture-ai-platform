import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
from scipy import stats
import json

class DataValidator:
    def __init__(self, baseline_stats: Optional[Dict] = None):
        self.baseline_stats = baseline_stats or {}
        
    def validate_dataset(self, dataset_path: str) -> Dict:
        """Validate entire dataset"""
        dataset_path = Path(dataset_path)
        
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check if path exists
        if not dataset_path.exists():
            results['valid'] = False
            results['errors'].append(f"Dataset path not found: {dataset_path}")
            return results
        
        # Find images
        if dataset_path.is_dir():
            image_files = list(dataset_path.rglob("*.jpg")) + \
                         list(dataset_path.rglob("*.jpeg")) + \
                         list(dataset_path.rglob("*.png"))
        else:
            results['valid'] = False
            results['errors'].append(f"Path is not a directory: {dataset_path}")
            return results
        
        # Validate images
        image_results = self.validate_images(image_files)
        results['statistics']['total_images'] = len(image_files)
        results['statistics']['valid_images'] = image_results['valid_count']
        results['statistics']['invalid_images'] = image_results['invalid_count']
        
        if image_results['invalid_count'] > 0:
            results['warnings'].append(
                f"{image_results['invalid_count']} invalid images found"
            )
        
        # Check class distribution
        class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        if class_dirs:
            class_distribution = {}
            for class_dir in class_dirs:
                class_count = len(list(class_dir.glob("*.jpg"))) + \
                            len(list(class_dir.glob("*.jpeg"))) + \
                            len(list(class_dir.glob("*.png")))
                class_distribution[class_dir.name] = class_count
            
            results['statistics']['class_distribution'] = class_distribution
            results['statistics']['num_classes'] = len(class_distribution)
            
            # Check class balance
            counts = list(class_distribution.values())
            if max(counts) / min(counts) > 10:
                results['warnings'].append("Dataset is highly imbalanced")
        
        return results
    
    def validate_images(self, image_paths: List[Path]) -> Dict:
        """Validate a list of images"""
        results = {
            'valid_count': 0,
            'invalid_count': 0,
            'errors': [],
            'statistics': {
                'sizes': [],
                'formats': [],
                'modes': []
            }
        }
        
        for img_path in image_paths:
            try:
                with Image.open(img_path) as img:
                    img.verify()
                    
                # Get image info
                with Image.open(img_path) as img:
                    results['statistics']['sizes'].append((img.width, img.height))
                    results['statistics']['formats'].append(img.format)
                    results['statistics']['modes'].append(img.mode)
                
                results['valid_count'] += 1
                
            except Exception as e:
                results['invalid_count'] += 1
                results['errors'].append(f"{img_path}: {str(e)}")
        
        # Calculate size statistics
        if results['statistics']['sizes']:
            widths = [s[0] for s in results['statistics']['sizes']]
            heights = [s[1] for s in results['statistics']['sizes']]
            
            results['statistics']['size_stats'] = {
                'width': {
                    'mean': np.mean(widths),
                    'std': np.std(widths),
                    'min': min(widths),
                    'max': max(widths)
                },
                'height': {
                    'mean': np.mean(heights),
                    'std': np.std(heights),
                    'min': min(heights),
                    'max': max(heights)
                }
            }
        
        return results
    
    def validate_metadata(self, metadata_path: str) -> Dict:
        """Validate metadata CSV"""
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            df = pd.read_csv(metadata_path)
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Failed to load metadata: {str(e)}")
            return results
        
        # Check required columns
        required_columns = ['image_path', 'label']
        for col in required_columns:
            if col not in df.columns:
                results['valid'] = False
                results['errors'].append(f"Missing required column: {col}")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            results['warnings'].append(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            results['warnings'].append(f"Found {duplicates} duplicate rows")
        
        # Statistics
        results['statistics'] = {
            'total_rows': len(df),
            'num_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'label_distribution': df['label'].value_counts().to_dict() if 'label' in df.columns else {}
        }
        
        return results
    
    def check_data_drift(
        self, 
        current_data: pd.DataFrame, 
        baseline_data: pd.DataFrame,
        threshold: float = 0.05
    ) -> Dict:
        """Check for data drift between current and baseline data"""
        results = {
            'drift_detected': False,
            'features': {},
            'overall_drift_score': 0
        }
        
        drifted_features = []
        
        for column in current_data.columns:
            if column in baseline_data.columns:
                # KS test for numerical columns
                if current_data[column].dtype in ['int64', 'float64']:
                    stat, p_value = stats.ks_2samp(
                        baseline_data[column].dropna(),
                        current_data[column].dropna()
                    )
                    
                    drift_detected = p_value < threshold
                    results['features'][column] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'drift_detected': drift_detected
                    }
                    
                    if drift_detected:
                        drifted_features.append(column)
        
        results['drift_detected'] = len(drifted_features) > 0
        results['drifted_features'] = drifted_features
        results['overall_drift_score'] = len(drifted_features) / len(current_data.columns) if current_data.columns.any() else 0
        
        return results
    
    def generate_validation_report(self, results: Dict, output_path: str):
        """Generate validation report"""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'results': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Validation report saved to {output_path}")
