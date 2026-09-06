import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json

class DataProcessor:
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_metadata(self, metadata_path: str) -> pd.DataFrame:
        """Load metadata from CSV"""
        return pd.read_csv(metadata_path)
    
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """Validate data quality"""
        validation_results = {
            'total_samples': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'class_distribution': df['label'].value_counts().to_dict() if 'label' in df.columns else {},
            'valid': True,
            'errors': []
        }
        
        # Check for required columns
        required_columns = ['image_path', 'label']
        for col in required_columns:
            if col not in df.columns:
                validation_results['valid'] = False
                validation_results['errors'].append(f"Missing required column: {col}")
        
        # Check class balance
        if 'label' in df.columns:
            class_counts = df['label'].value_counts()
            min_class_count = class_counts.min()
            if min_class_count < 10:
                validation_results['errors'].append(f"Class with only {min_class_count} samples")
        
        return validation_results
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Remove rows with missing values in required columns
        df = df.dropna(subset=['image_path', 'label'])
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        train_size: float = 0.8, 
        val_size: float = 0.1, 
        test_size: float = 0.1,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
        
        # First split: train + val vs test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=42,
            stratify=df['label'] if stratify else None
        )
        
        # Second split: train vs val
        relative_val_size = val_size / (train_size + val_size)
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=relative_val_size, 
            random_state=42,
            stratify=train_val_df['label'] if stratify else None
        )
        
        return train_df, val_df, test_df
    
    def save_splits(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame
    ):
        """Save data splits to CSV"""
        train_df.to_csv(self.output_dir / "train.csv", index=False)
        val_df.to_csv(self.output_dir / "val.csv", index=False)
        test_df.to_csv(self.output_dir / "test.csv", index=False)
        
        print(f"Train: {len(train_df)} samples")
        print(f"Val: {len(val_df)} samples")
        print(f"Test: {len(test_df)} samples")
    
    def create_data_splits(self, metadata_path: str) -> Dict:
        """Create complete data splits"""
        # Load data
        df = self.load_metadata(metadata_path)
        
        # Validate
        validation = self.validate_data(df)
        if not validation['valid']:
            raise ValueError(f"Data validation failed: {validation['errors']}")
        
        # Clean
        df = self.clean_data(df)
        
        # Split
        train_df, val_df, test_df = self.split_data(df)
        
        # Save
        self.save_splits(train_df, val_df, test_df)
        
        # Create summary
        summary = {
            'total_samples': len(df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'num_classes': df['label'].nunique(),
            'classes': df['label'].unique().tolist(),
            'class_distribution': {
                'train': train_df['label'].value_counts().to_dict(),
                'val': val_df['label'].value_counts().to_dict(),
                'test': test_df['label'].value_counts().to_dict()
            }
        }
        
        # Save summary
        with open(self.output_dir / "data_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def encode_labels(self, labels: pd.Series) -> Tuple[np.ndarray, Dict]:
        """Encode labels"""
        encoded = self.label_encoder.fit_transform(labels)
        label_map = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        return encoded, label_map
    
    def decode_labels(self, encoded_labels: np.ndarray) -> np.ndarray:
        """Decode labels"""
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def get_class_weights(self, labels: pd.Series) -> Dict:
        """Calculate class weights for imbalanced data"""
        class_counts = labels.value_counts()
        total_samples = len(labels)
        
        weights = {}
        for class_name, count in class_counts.items():
            weights[class_name] = total_samples / (len(class_counts) * count)
        
        return weights
    
    def augment_minority_classes(
        self, 
        df: pd.DataFrame, 
        target_count: int = None
    ) -> pd.DataFrame:
        """Augment minority classes to balance dataset"""
        if target_count is None:
            target_count = df['label'].value_counts().max()
        
        augmented_dfs = []
        
        for class_name in df['label'].unique():
            class_df = df[df['label'] == class_name]
            
            if len(class_df) < target_count:
                # Oversample
                oversampled = class_df.sample(
                    target_count, 
                    replace=True, 
                    random_state=42
                )
                augmented_dfs.append(oversampled)
            else:
                augmented_dfs.append(class_df)
        
        return pd.concat(augmented_dfs, ignore_index=True)
