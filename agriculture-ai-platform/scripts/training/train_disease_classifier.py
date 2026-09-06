#!/usr/bin/env python3
"""
Training script for Disease Classifier
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ml.pipelines.training_pipeline import MLPipeline
from src.ml.config import TrainingConfig

def main():
    parser = argparse.ArgumentParser(description='Train Disease Classifier')
    parser.add_argument('--dataset-version', type=str, default='v1',
                        help='Dataset version to use')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to train on (cuda:0 or cpu)')
    parser.add_argument('--model-name', type=str, default='disease_classifier',
                        help='Name for the model')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        model_name=args.model_name,
        dataset_version=args.dataset_version,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # Create pipeline
    pipeline = MLPipeline(config)
    
    # Run training
    print("Starting training...")
    results = pipeline.run()
    
    print(f"Training complete!")
    print(f"Best accuracy: {results['best_accuracy']:.4f}")
    print(f"Model saved to: {results['model_path']}")

if __name__ == '__main__':
    main()
