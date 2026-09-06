import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import json

class EdgeDeployer:
    """Deploy models to edge devices"""
    
    def __init__(
        self,
        model: nn.Module,
        input_size: Tuple[int, int] = (224, 224),
        device: str = "cpu"
    ):
        self.model = model
        self.input_size = input_size
        self.device = device
        
    def export_to_onnx(
        self,
        output_path: str,
        opset_version: int = 11,
        dynamic_axes: bool = False
    ) -> str:
        """Export model to ONNX"""
        self.model.eval()
        dummy_input = torch.randn(1, 3, *self.input_size)
        
        dynamic_axes_config = None
        if dynamic_axes:
            dynamic_axes_config = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
            
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes_config
        )
        
        return output_path
        
    def export_to_torchscript(
        self,
        output_path: str,
        tracing: bool = True
    ) -> str:
        """Export to TorchScript"""
        self.model.eval()
        dummy_input = torch.randn(1, 3, *self.input_size)
        
        if tracing:
            traced_model = torch.jit.trace(self.model, dummy_input)
        else:
            scripted_model = torch.jit.script(self.model)
            
        traced_model.save(output_path)
        return output_path
        
    def quantize_for_mobile(
        self,
        output_path: str
    ) -> str:
        """Quantize model for mobile deployment"""
        self.model.eval()
        
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        # Export to TorchScript
        dummy_input = torch.randn(1, 3, *self.input_size)
        traced = torch.jit.trace(quantized_model, dummy_input)
        traced.save(output_path)
        
        return output_path


class MobileNetDeployer(EdgeDeployer):
    """MobileNet-specific deployment"""
    
    def optimize_for_mobile(
        self,
        output_path: str,
        optimize_for_gpu: bool = False
    ) -> str:
        """Optimize for mobile deployment"""
        # Use MobileNet backbone if available
        try:
            from torchvision.models import mobilenet_v3_small
            mobile_model = mobilenet_v3_small(pretrained=True)
            
            # Replace classifier
            num_features = mobile_model.classifier[3].in_features
            mobile_model.classifier[3] = torch.nn.Linear(
                num_features, self.model.num_classes
            )
            
            # Quantize
            mobile_model.eval()
            quantized = torch.quantization.quantize_dynamic(
                mobile_model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            
            # Export
            dummy_input = torch.randn(1, 3, *self.input_size)
            traced = torch.jit.trace(quantized, dummy_input)
            traced.save(output_path)
            
            return output_path
            
        except Exception as e:
            print(f"Error optimizing for mobile: {e}")
            return self.export_to_torchscript(output_path)


class TensorRTOptimizer:
    """TensorRT optimization for NVIDIA GPUs"""
    
    def __init__(self, model: nn.Module, input_size: Tuple[int, int] = (224, 224)):
        self.model = model
        self.input_size = input_size
        
    def export_for_tensorrt(
        self,
        output_path: str,
        fp16: bool = True,
        max_batch_size: int = 32
    ) -> str:
        """Export for TensorRT optimization"""
        # First export to ONNX
        onnx_path = output_path.replace('.engine', '.onnx')
        
        deployer = EdgeDeployer(self.model, self.input_size)
        deployer.export_to_onnx(onnx_path, dynamic_axes=True)
        
        # Build TensorRT engine (requires trtexec)
        import subprocess
        
        cmd = [
            'trtexec',
            f'--onnx={onnx_path}',
            f'--saveEngine={output_path}',
            f'--maxBatchSize={max_batch_size}'
        ]
        
        if fp16:
            cmd.append('--fp16')
            
        subprocess.run(cmd, check=True)
        
        return output_path


class OpenVINOExporter:
    """OpenVINO export for Intel hardware"""
    
    def __init__(self, model: nn.Module, input_size: Tuple[int, int] = (224, 224)):
        self.model = model
        self.input_size = input_size
        
    def export_for_openvino(
        self,
        output_dir: str
    ) -> str:
        """Export for OpenVINO"""
        # Export to ONNX first
        onnx_path = f"{output_dir}/model.onnx"
        
        deployer = EdgeDeployer(self.model, self.input_size)
        deployer.export_to_onnx(onnx_path)
        
        # Convert to OpenVINO IR (requires mo.py)
        import subprocess
        
        cmd = [
            'mo.py',
            '--input_model', onnx_path,
            '--output_dir', output_dir,
            '--data_type', 'FP16'
        ]
        
        subprocess.run(cmd, check=True)
        
        return output_dir


class EdgeOptimizer:
    """Optimize models for edge deployment"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        
    def prune_model(
        self,
        amount: float = 0.3
    ) -> nn.Module:
        """Prune model for faster inference"""
        import torch.nn.utils.prune as prune
        
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=amount)
                
        return self.model
        
    def channel_pruning(
        self,
        amount: float = 0.3
    ) -> nn.Module:
        """Channel pruning"""
        import torch.nn.utils.prune as prune
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.ln_structured(
                    module, 
                    name='weight', 
                    amount=amount, 
                    n=2, 
                    dim=0
                )
                
        return self.model
        
    def knowledge_distillation(
        self,
        teacher_model: nn.Module,
        train_loader,
        temperature: float = 4.0,
        alpha: float = 0.7,
        num_epochs: int = 10
    ) -> nn.Module:
        """Knowledge distillation"""
        teacher_model.eval()
        self.model.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                # Teacher predictions
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                    
                # Student predictions
                student_outputs = self.model(inputs)
                
                # Distillation loss
                soft_teacher = torch.softmax(teacher_outputs / temperature, dim=1)
                soft_student = torch.log_softmax(student_outputs / temperature, dim=1)
                distill_loss = torch.nn.functional.kl_div(
                    soft_student, soft_teacher, reduction='batchmean'
                ) * (temperature ** 2)
                
                # Hard loss
                hard_loss = criterion(student_outputs, labels)
                
                # Combined loss
                loss = alpha * distill_loss + (1 - alpha) * hard_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        return self.model


class EdgeModelManager:
    """Manage multiple edge models"""
    
    def __init__(self, models_dir: str = "edge_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.models: Dict[str, nn.Module] = {}
        
    def add_model(
        self,
        name: str,
        model: nn.Module,
        optimize: bool = True
    ):
        """Add model for deployment"""
        if optimize:
            optimizer = EdgeOptimizer(model)
            model = optimizer.prune_model(amount=0.3)
            
        self.models[name] = model
        
    def deploy_model(
        self,
        name: str,
        format: str = "onnx"
    ) -> str:
        """Deploy model"""
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
            
        model = self.models[name]
        output_path = str(self.models_dir / f"{name}.{format}")
        
        if format == "onnx":
            deployer = EdgeDeployer(model)
            deployer.export_to_onnx(output_path)
        elif format == "torchscript":
            deployer = EdgeDeployer(model)
            deployer.export_to_torchscript(output_path)
        elif format == "quantized":
            deployer = EdgeDeployer(model)
            deployer.quantize_for_mobile(output_path)
            
        return output_path
        
    def get_deployment_info(self) -> Dict[str, Any]:
        """Get deployment information"""
        info = {}
        
        for name, model in self.models.items():
            # Count parameters
            params = sum(p.numel() for p in model.parameters())
            
            # Estimate size
            torch.save(model.state_dict(), '/tmp/temp.pth')
            import os
            size_mb = os.path.getsize('/tmp/temp.pth') / (1024 * 1024)
            os.remove('/tmp/temp.pth')
            
            info[name] = {
                "parameters": params,
                "size_mb": size_mb,
                "formats_available": ["onnx", "torchscript", "quantized"]
            }
            
        return info
