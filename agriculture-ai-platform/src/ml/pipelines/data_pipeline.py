import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from src.ml.data.collectors.image_collector import ImageCollector
from src.ml.data.processors.image_processor import ImageProcessor
from src.ml.data.processors.data_processor import DataProcessor
from src.ml.data.validators.data_validator import DataValidator
from src.ml.features.engineering.feature_engineer import FeatureEngineer

class DataPipeline:
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.collector = ImageCollector(output_dir / "raw")
        self.processor = ImageProcessor()
        self.data_processor = DataProcessor(output_dir / "processed")
        self.validator = DataValidator()
        self.feature_engineer = FeatureEngineer()
        
    def run_collection(self, source_dirs: Dict[str, str]) -> Dict:
        """Run data collection pipeline"""
        print("=" * 60)
        print("Data Collection Pipeline")
        print("=" * 60)
        
        results = {}
        
        for class_name, source_dir in source_dirs.items():
            print(f"\nCollecting {class_name} from {source_dir}...")
            count = self.collector.collect_from_directory(source_dir, class_name)
            results[class_name] = count
            print(f"Collected {count} images for {class_name}")
        
        # Save metadata
        self.collector.save_metadata()
        
        # Remove duplicates
        duplicates_removed = self.collector.remove_duplicates()
        print(f"\nRemoved {duplicates_removed} duplicate images")
        
        # Get statistics
        stats = self.collector.get_statistics()
        print(f"\nCollection Statistics:")
        print(f"Total images: {stats['total_images']}")
        print(f"Total size: {stats['total_size_mb']:.2f} MB")
        
        return {
            'collection_results': results,
            'statistics': stats,
            'duplicates_removed': duplicates_removed
        }
    
    def run_processing(self, metadata_path: str = None) -> Dict:
        """Run data processing pipeline"""
        print("\n" + "=" * 60)
        print("Data Processing Pipeline")
        print("=" * 60)
        
        if metadata_path is None:
            metadata_path = self.output_dir / "raw" / "metadata.csv"
        
        # Load metadata
        df = pd.read_csv(metadata_path)
        print(f"Loaded {len(df)} samples")
        
        # Validate data
        print("\nValidating data...")
        validation_results = self.validator.validate_metadata(metadata_path)
        print(f"Validation: {'Passed' if validation_results['valid'] else 'Failed'}")
        
        if not validation_results['valid']:
            raise ValueError(f"Data validation failed: {validation_results['errors']}")
        
        # Process images
        print("\nProcessing images...")
        processed_count = 0
        for idx, row in df.iterrows():
            image_path = self.output_dir / "raw" / row['image_path']
            if image_path.exists():
                try:
                    # Process image
                    processed_images = self.processor.process_image(str(image_path))
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
        
        print(f"Processed {processed_count} images")
        
        # Create data splits
        print("\nCreating data splits...")
        data_summary = self.data_processor.create_data_splits(str(metadata_path))
        
        print(f"Data splits created:")
        print(f"  Train: {data_summary['train_samples']}")
        print(f"  Val: {data_summary['val_samples']}")
        print(f"  Test: {data_summary['test_samples']}")
        
        return data_summary
    
    def run_feature_extraction(self) -> Dict:
        """Run feature extraction pipeline"""
        print("\n" + "=" * 60)
        print("Feature Extraction Pipeline")
        print("=" * 60)
        
        # Load data
        train_df = pd.read_csv(self.output_dir / "processed" / "train.csv")
        
        # Extract features
        print("Extracting features...")
        image_paths = [self.output_dir / "raw" / p for p in train_df['image_path']]
        labels = train_df['label'].values
        
        features, encoded_labels, valid_paths = self.feature_engineer.process_dataset(
            [str(p) for p in image_paths],
            labels
        )
        
        print(f"Extracted {features.shape[1]} features from {features.shape[0]} images")
        
        # Save features
        self.feature_engineer.save_features(
            features,
            encoded_labels,
            str(self.output_dir / "features" / "train_features.npz")
        )
        
        # Dimensionality reduction
        if features.shape[1] > 50:
            print("\nApplying PCA for dimensionality reduction...")
            features_reduced = self.feature_engineer.reduce_dimensions(features, n_components=50)
            
            self.feature_engineer.save_features(
                features_reduced,
                encoded_labels,
                str(self.output_dir / "features" / "train_features_pca.npz")
            )
            
            print(f"Reduced to {features_reduced.shape[1]} features")
        
        return {
            'num_features': features.shape[1],
            'num_samples': features.shape[0],
            'num_classes': len(np.unique(encoded_labels))
        }
    
    def run_validation(self) -> Dict:
        """Run data validation pipeline"""
        print("\n" + "=" * 60)
        print("Data Validation Pipeline")
        print("=" * 60)
        
        # Validate dataset
        results = self.validator.validate_dataset(str(self.output_dir / "raw"))
        
        print(f"Validation Results:")
        print(f"  Valid: {results['valid']}")
        print(f"  Total images: {results['statistics'].get('total_images', 0)}")
        print(f"  Valid images: {results['statistics'].get('valid_images', 0)}")
        print(f"  Invalid images: {results['statistics'].get('invalid_images', 0)}")
        
        if results['warnings']:
            print("\nWarnings:")
            for warning in results['warnings']:
                print(f"  - {warning}")
        
        return results
    
    def run_full_pipeline(self, source_dirs: Dict[str, str] = None) -> Dict:
        """Run complete data pipeline"""
        print("=" * 60)
        print("Agriculture AI - Complete Data Pipeline")
        print("=" * 60)
        
        results = {}
        
        # Step 1: Collection
        if source_dirs:
            print("\n[Step 1] Data Collection")
            results['collection'] = self.run_collection(source_dirs)
        
        # Step 2: Validation
        print("\n[Step 2] Data Validation")
        results['validation'] = self.run_validation()
        
        # Step 3: Processing
        print("\n[Step 3] Data Processing")
        results['processing'] = self.run_processing()
        
        # Step 4: Feature Extraction
        print("\n[Step 4] Feature Extraction")
        results['features'] = self.run_feature_extraction()
        
        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        
        return results


def main():
    """Main entry point for data pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Data Pipeline')
    parser.add_argument('--source-dirs', type=str, nargs='+',
                        help='Source directories for each class (format: class_name:path)')
    parser.add_argument('--skip-collection', action='store_true',
                        help='Skip data collection step')
    parser.add_argument('--skip-features', action='store_true',
                        help='Skip feature extraction step')
    
    args = parser.parse_args()
    
    pipeline = DataPipeline()
    
    if args.source_dirs:
        source_dirs = {}
        for item in args.source_dirs:
            class_name, path = item.split(':')
            source_dirs[class_name] = path
    else:
        source_dirs = None
    
    results = pipeline.run_full_pipeline(source_dirs)
    
    print("\nPipeline Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
