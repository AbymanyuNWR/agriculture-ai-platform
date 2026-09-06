import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import mlflow
import mlflow.pytorch
from pathlib import Path
import json
from datetime import datetime

from src.ml.config import TrainingConfig, ModelConfig
from src.ml.models.classification.disease_classifier import DiseaseClassifier

class ModelTrainer:
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def setup_model(self, num_classes: int = 10) -> nn.Module:
        """Initialize model"""
        self.model = DiseaseClassifier(num_classes=num_classes)
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        if self.config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        
        # Setup scheduler
        if self.config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        elif self.config.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        
        return self.model
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs, _ = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int = 10
    ) -> Dict:
        """Complete training loop"""
        print(f"Training on {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        # Setup model
        self.setup_model(num_classes)
        
        # MLflow tracking
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("agriculture_ai")
        
        best_val_acc = 0.0
        patience_counter = 0
        
        with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params({
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'optimizer': self.config.optimizer,
                'scheduler': self.config.scheduler,
                'model': 'disease_classifier'
            })
            
            for epoch in range(self.config.epochs):
                print(f"\nEpoch {epoch+1}/{self.config.epochs}")
                print("-" * 40)
                
                # Train
                train_loss, train_acc = self.train_epoch(train_loader)
                self.train_losses.append(train_loss)
                self.train_accs.append(train_acc)
                
                # Validate
                val_loss, val_acc = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                
                # Update scheduler
                if self.scheduler:
                    self.scheduler.step()
                
                # Log metrics
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }, step=epoch)
                
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    self.save_model('best_model.pth')
                    mlflow.pytorch.log_model(self.model, "model")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= self.config.early_stopping:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Log final metrics
            mlflow.log_metrics({
                'best_val_acc': best_val_acc,
                'final_train_loss': self.train_losses[-1],
                'final_val_loss': self.val_losses[-1]
            })
        
        return {
            'best_val_acc': best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }, path)
    
    def load_model(self, path: str, num_classes: int = 10):
        """Load model checkpoint"""
        self.setup_model(num_classes)
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return self.model
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate model on test set"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                
                outputs, _ = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        import numpy as np
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='macro'),
            'recall': recall_score(all_labels, all_preds, average='macro'),
            'f1_score': f1_score(all_labels, all_preds, average='macro')
        }
        
        return metrics
