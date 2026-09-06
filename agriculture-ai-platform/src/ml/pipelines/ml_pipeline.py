import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
from datetime import datetime

from ..models.advanced.model_factory import ModelFactory
from ..training.trainers.model_trainer import ModelTrainer
from ..evaluation.metrics.classification_metrics import ClassificationEvaluator
from ..evaluation.metrics.detection_metrics import DetectionEvaluator
from ..evaluation.metrics.segmentation_metrics import SegmentationEvaluator
from .inference_pipeline import InferencePipeline, ImagePreprocessor, PredictionPostprocessor
from ..optimization.quantization import ModelQuantizer
from ..governance.model_registry import ModelRegistry, ModelStage

class MLPipelineOrchestrator:
    """Orchestrate complete ML workflow"""
    
    def __init__(
        self,
        project_name: str,
        base_dir: str = "ml_projects"
    ):
        self.project_name = project_name
        self.base_dir = Path(base_dir) / project_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.model_factory = ModelFactory()
        self.model_registry = ModelRegistry(str(self.base_dir / "registry"))
        
        # State
        self.current_model = None
        self.current_config = None
        
    def create_experiment(
        self,
        experiment_name: str,
        model_type: str = "efficientnet",
        num_classes: int = 10,
        input_size: Tuple[int, int] = (224, 224),
        **kwargs
    ) -> Dict[str, Any]:
        """Create new experiment"""
        experiment_dir = self.base_dir / "experiments" / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model
        model = self.model_factory.create_model(
            model_type=model_type,
            num_classes=num_classes,
            input_size=input_size,
            **kwargs
        )
        
        # Save config
        config = {
            "experiment_name": experiment_name,
            "model_type": model_type,
            "num_classes": num_classes,
            "input_size": input_size,
            "created_at": datetime.now().isoformat(),
            "kwargs": kwargs
        }
        
        config_path = experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        self.current_model = model
        self.current_config = config
        
        return {
            "experiment_dir": str(experiment_dir),
            "config": config,
            "model": model
        }
        
    def train_model(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ) -> Dict[str, Any]:
        """Train model"""
        if self.current_model is None:
            raise ValueError("No model created. Run create_experiment first.")
            
        # Create trainer
        trainer = ModelTrainer(
            model=self.current_model,
            device=device,
            **kwargs
        )
        
        # Train
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate
        )
        
        # Save model
        model_path = self.base_dir / "experiments" / self.current_config["experiment_name"] / "model.pth"
        torch.save(self.current_model.state_dict(), model_path)
        
        # Register in registry
        self.model_registry.register_model(
            model=self.current_model,
            name=self.project_name,
            version=self.current_config["experiment_name"],
            description=f"Experiment: {self.current_config['experiment_name']}",
            author="system",
            metrics=results.get("metrics", {})
        )
        
        results["model_path"] = str(model_path)
        
        return results
        
    def evaluate_model(
        self,
        test_loader,
        task_type: str = "classification",
        num_classes: int = 10,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate model"""
        if self.current_model is None:
            raise ValueError("No model available.")
            
        # Create evaluator
        if task_type == "classification":
            evaluator = ClassificationEvaluator(num_classes, class_names)
        elif task_type == "detection":
            evaluator = DetectionEvaluator(num_classes, class_names)
        elif task_type == "segmentation":
            evaluator = SegmentationEvaluator(num_classes, class_names)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
            
        # Evaluate
        self.current_model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.current_model(inputs)
                
                if task_type == "classification":
                    _, preds = outputs.max(1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())
                    all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
                    
        # Compute metrics
        if task_type == "classification":
            metrics = evaluator.compute_metrics(
                np.array(all_labels),
                np.array(all_preds),
                np.array(all_probs)
            )
        else:
            metrics = {}
            
        return {"metrics": metrics}
        
    def optimize_model(
        self,
        optimization_type: str = "quantization",
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize model for deployment"""
        if self.current_model is None:
            raise ValueError("No model available.")
            
        results = {}
        
        if optimization_type == "quantization":
            quantizer = ModelQuantizer(self.current_model)
            
            if kwargs.get("method") == "dynamic":
                quantized = quantizer.dynamic_quantization()
            elif kwargs.get("method") == "static":
                quantized = quantizer.static_quantization()
            elif kwargs.get("method") == "qat":
                quantized = quantizer.quantize_aware_training(
                    kwargs.get("train_loader")
                )
            else:
                quantized = quantizer.dynamic_quantization()
                
            results["quantized_model"] = quantized
            results["size_comparison"] = quantizer.compare_sizes()
            
        elif optimization_type == "pruning":
            import torch.nn.utils.prune as prune
            
            amount = kwargs.get("amount", 0.3)
            for name, module in self.current_model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    prune.l1_unstructured(module, name='weight', amount=amount)
                    
            results["pruned_model"] = self.current_model
            
        elif optimization_type == "distillation":
            from ..optimization.quantization import ModelDistiller
            
            teacher = kwargs.get("teacher_model")
            if teacher is None:
                raise ValueError("Teacher model required for distillation")
                
            distiller = ModelDistiller(
                teacher_model=teacher,
                student_model=self.current_model,
                temperature=kwargs.get("temperature", 4.0),
                alpha=kwargs.get("alpha", 0.5)
            )
            
            # Train student
            train_loader = kwargs.get("train_loader")
            num_epochs = kwargs.get("num_epochs", 10)
            
            optimizer = torch.optim.Adam(self.current_model.parameters())
            
            for epoch in range(num_epochs):
                for inputs, labels in train_loader:
                    result = distiller.train_step(inputs, labels, optimizer)
                    
            results["distilled_model"] = self.current_model
            
        return results
        
    def deploy_model(
        self,
        deployment_target: str = "onnx",
        output_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Deploy model"""
        if self.current_model is None:
            raise ValueError("No model available.")
            
        if output_path is None:
            output_path = str(self.base_dir / "deploy" / f"model.{deployment_target}")
            
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        results = {"output_path": output_path}
        
        if deployment_target == "onnx":
            from ..optimization.quantization import ONNXExporter
            exporter = ONNXExporter(self.current_model, kwargs.get("input_size", (224, 224)))
            exporter.export(output_path)
            
        elif deployment_target == "torchscript":
            self.current_model.eval()
            dummy_input = torch.randn(1, 3, *kwargs.get("input_size", (224, 224)))
            traced = torch.jit.trace(self.current_model, dummy_input)
            traced.save(output_path)
            
        elif deployment_target == "quantized":
            from ..optimization.quantization import ModelQuantizer
            quantizer = ModelQuantizer(self.current_model)
            quantizer.quantize_for_mobile(output_path)
            
        # Promote to production in registry
        self.model_registry.promote_model(
            self.project_name,
            self.current_config["experiment_name"],
            ModelStage.PRODUCTION
        )
        
        return results
        
    def create_inference_pipeline(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> InferencePipeline:
        """Create inference pipeline"""
        if model_path and Path(model_path).exists():
            state_dict = torch.load(model_path)
            self.current_model.load_state_dict(state_dict)
            
        pipeline = InferencePipeline(
            model=self.current_model,
            device=device
        )
        
        return pipeline
        
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get experiment summary"""
        return {
            "project_name": self.project_name,
            "config": self.current_config,
            "model_info": {
                "parameters": sum(p.numel() for p in self.current_model.parameters()) if self.current_model else 0,
                "device": next(self.current_model.parameters()).device if self.current_model else "none"
            }
        }


class FullMLPipeline:
    """Complete ML pipeline from data to deployment"""
    
    def __init__(self, project_name: str, base_dir: str = "ml_projects"):
        self.orchestrator = MLPipelineOrchestrator(project_name, base_dir)
        
    def run_complete_pipeline(
        self,
        train_loader,
        val_loader,
        test_loader,
        model_type: str = "efficientnet",
        num_classes: int = 10,
        num_epochs: int = 100,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run complete ML pipeline"""
        results = {}
        
        # Step 1: Create experiment
        print("Step 1: Creating experiment...")
        experiment = self.orchestrator.create_experiment(
            experiment_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_type=model_type,
            num_classes=num_classes
        )
        results["experiment"] = experiment
        
        # Step 2: Train model
        print("Step 2: Training model...")
        train_results = self.orchestrator.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs
        )
        results["training"] = train_results
        
        # Step 3: Evaluate model
        print("Step 3: Evaluating model...")
        eval_results = self.orchestrator.evaluate_model(
            test_loader=test_loader,
            task_type="classification",
            num_classes=num_classes,
            class_names=class_names
        )
        results["evaluation"] = eval_results
        
        # Step 4: Optimize model
        print("Step 4: Optimizing model...")
        opt_results = self.orchestrator.optimize_model(
            optimization_type="quantization",
            method="dynamic"
        )
        results["optimization"] = opt_results
        
        # Step 5: Deploy model
        print("Step 5: Deploying model...")
        deploy_results = self.orchestrator.deploy_model(
            deployment_target="onnx"
        )
        results["deployment"] = deploy_results
        
        # Step 6: Create inference pipeline
        print("Step 6: Creating inference pipeline...")
        inference_pipeline = self.orchestrator.create_inference_pipeline()
        results["inference_pipeline"] = inference_pipeline
        
        print("Pipeline complete!")
        
        return results
