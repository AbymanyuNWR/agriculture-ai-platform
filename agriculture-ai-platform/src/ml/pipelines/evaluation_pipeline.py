import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json

from src.ml.config import EvaluationConfig
from src.ml.models.classification.disease_classifier import DiseaseClassifier
from src.ml.evaluation.metrics.model_metrics import ModelMetrics

class EvaluationPipeline:
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.metrics_calculator = ModelMetrics()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self, model_path: str, num_classes: int = 10) -> DiseaseClassifier:
        """Load trained model"""
        model = DiseaseClassifier(num_classes=num_classes)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def evaluate_model(self, model: DiseaseClassifier, test_loader) -> Dict:
        """Evaluate model on test set"""
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                
                outputs, _ = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(all_labels, all_preds, all_probs)
        
        return {
            'metrics': metrics,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
    
    def generate_report(self, results: Dict, output_dir: str = "reports") -> str:
        """Generate evaluation report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate confusion matrix
        self.metrics_calculator.plot_confusion_matrix(
            results['labels'],
            results['predictions'],
            output_path / "confusion_matrix.png"
        )
        
        # Generate ROC curve
        self.metrics_calculator.plot_roc_curve(
            results['labels'],
            results['probabilities'],
            output_path / "roc_curve.png"
        )
        
        # Generate precision-recall curve
        self.metrics_calculator.plot_precision_recall_curve(
            results['labels'],
            results['probabilities'],
            output_path / "precision_recall_curve.png"
        )
        
        # Generate metrics comparison
        self.metrics_calculator.plot_metrics_comparison(
            results['metrics'],
            output_path / "metrics_comparison.png"
        )
        
        # Generate per-class metrics
        self.metrics_calculator.plot_per_class_metrics(
            results['metrics'],
            output_path / "per_class_metrics.png"
        )
        
        # Generate classification report
        report = self.metrics_calculator.generate_classification_report(
            results['labels'],
            results['predictions'],
            output_path / "classification_report.txt"
        )
        
        # Save metrics
        with open(output_path / "metrics.json", 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        
        print(f"Report generated in {output_path}")
        
        return report
    
    def run(self, model_path: str, test_loader, output_dir: str = "reports") -> Dict:
        """Run complete evaluation pipeline"""
        print("=" * 60)
        print("Evaluation Pipeline")
        print("=" * 60)
        
        # Load model
        print("\nLoading model...")
        model = self.load_model(model_path)
        print(f"Model loaded from {model_path}")
        
        # Evaluate
        print("\nEvaluating model...")
        results = self.evaluate_model(model, test_loader)
        
        # Print metrics
        print("\nEvaluation Results:")
        print(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"  Precision: {results['metrics']['precision']:.4f}")
        print(f"  Recall: {results['metrics']['recall']:.4f}")
        print(f"  F1-Score: {results['metrics']['f1_score']:.4f}")
        
        # Generate report
        print("\nGenerating report...")
        report = self.generate_report(results, output_dir)
        
        print("\n" + "=" * 60)
        print("Evaluation Complete!")
        print("=" * 60)
        
        return {
            'metrics': results['metrics'],
            'report': report
        }


def main():
    """Main entry point for evaluation pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Disease Classifier')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--test-data', type=str, default='data/processed/test.csv',
                        help='Path to test data CSV')
    parser.add_argument('--output-dir', type=str, default='reports',
                        help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Create test loader
    from torch.utils.data import DataLoader
    from PIL import Image
    import pandas as pd
    
    class TestDataset:
        def __init__(self, csv_path, image_dir):
            self.data = pd.read_csv(csv_path)
            self.image_dir = Path(image_dir)
            self.label_map = {label: idx for idx, label in enumerate(self.data['label'].unique())}
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            image_path = self.image_dir / row['image_path']
            
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224, 224))
            image = np.array(image).transpose(2, 0, 1) / 255.0
            
            label = self.label_map[row['label']]
            
            return torch.tensor(image, dtype=torch.float32), label
    
    test_dataset = TestDataset(args.test_data, 'data/raw')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    pipeline = EvaluationPipeline()
    results = pipeline.run(args.model_path, test_loader, args.output_dir)
    
    print(f"\nFinal Metrics:")
    for key, value in results['metrics'].items():
        if not isinstance(value, dict):
            print(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    main()
