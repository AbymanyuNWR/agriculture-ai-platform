import sys
from pathlib import Path
from typing import Dict, Optional
import json

from src.ml.config import TrainingConfig, DataConfig, ModelConfig
from src.ml.data.processors.data_processor import DataProcessor
from src.ml.data.processors.image_processor import ImageProcessor
from src.ml.training.trainers.model_trainer import ModelTrainer
from src.ml.evaluation.metrics.model_metrics import ModelMetrics

class TrainingPipeline:
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.data_processor = DataProcessor()
        self.image_processor = ImageProcessor()
        self.trainer = ModelTrainer(self.config)
        self.metrics_calculator = ModelMetrics()
        
    def run(self, metadata_path: str = None) -> Dict:
        """Run complete training pipeline"""
        print("=" * 60)
        print("Agriculture AI - Training Pipeline")
        print("=" * 60)
        
        # Step 1: Data Preparation
        print("\n[Step 1] Preparing data...")
        data_summary = self.prepare_data(metadata_path)
        print(f"Data prepared: {data_summary['total_samples']} samples")
        print(f"Classes: {data_summary['num_classes']}")
        
        # Step 2: Create Data Loaders
        print("\n[Step 2] Creating data loaders...")
        train_loader, val_loader, test_loader = self.create_data_loaders()
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Step 3: Train Model
        print("\n[Step 3] Training model...")
        training_results = self.trainer.train(train_loader, val_loader, data_summary['num_classes'])
        print(f"Training complete. Best val acc: {training_results['best_val_acc']:.2f}%")
        
        # Step 4: Evaluate Model
        print("\n[Step 4] Evaluating model...")
        test_metrics = self.trainer.evaluate(test_loader)
        print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1-score: {test_metrics['f1_score']:.4f}")
        
        # Step 5: Generate Report
        print("\n[Step 5] Generating report...")
        report = self.generate_report(training_results, test_metrics, data_summary)
        
        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        
        return {
            'data_summary': data_summary,
            'training_results': training_results,
            'test_metrics': test_metrics,
            'report': report
        }
    
    def prepare_data(self, metadata_path: str = None) -> Dict:
        """Prepare data for training"""
        if metadata_path is None:
            metadata_path = "data/raw/metadata.csv"
        
        # Create data splits
        data_summary = self.data_processor.create_data_splits(metadata_path)
        
        return data_summary
    
    def create_data_loaders(self):
        """Create PyTorch data loaders"""
        import torch
        from torch.utils.data import Dataset, DataLoader
        from PIL import Image
        import pandas as pd
        
        class AgricultureDataset(Dataset):
            def __init__(self, csv_path, image_dir, transform=None):
                self.data = pd.read_csv(csv_path)
                self.image_dir = Path(image_dir)
                self.transform = transform
                self.label_map = {label: idx for idx, label in enumerate(self.data['label'].unique())}
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                row = self.data.iloc[idx]
                image_path = self.image_dir / row['image_path']
                
                image = Image.open(image_path).convert('RGB')
                label = self.label_map[row['label']]
                
                if self.transform:
                    image = self.transform(image)
                
                return image, label
        
        # Create transforms
        train_transform = self.image_processor.create_train_transform()
        val_transform = self.image_processor.create_val_transform()
        
        # Create datasets
        train_dataset = AgricultureDataset(
            "data/processed/train.csv",
            "data/raw",
            train_transform
        )
        
        val_dataset = AgricultureDataset(
            "data/processed/val.csv",
            "data/raw",
            val_transform
        )
        
        test_dataset = AgricultureDataset(
            "data/processed/test.csv",
            "data/raw",
            val_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def generate_report(self, training_results: Dict, test_metrics: Dict, data_summary: Dict) -> Dict:
        """Generate training report"""
        report = {
            'config': self.config.__dict__,
            'data_summary': data_summary,
            'training_results': {
                'best_val_acc': training_results['best_val_acc'],
                'final_train_loss': training_results['train_losses'][-1],
                'final_val_loss': training_results['val_losses'][-1]
            },
            'test_metrics': test_metrics
        }
        
        # Save report
        output_path = Path("reports")
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / "training_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {output_path / 'training_report.json'}")
        
        return report


def main():
    """Main entry point for training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Disease Classifier')
    parser.add_argument('--dataset-version', type=str, default='v1',
                        help='Dataset version')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (cuda:0 or cpu)')
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        dataset_version=args.dataset_version,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    pipeline = TrainingPipeline(config)
    results = pipeline.run()
    
    print(f"\nFinal Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"Final Test F1-Score: {results['test_metrics']['f1_score']:.4f}")


if __name__ == '__main__':
    main()
