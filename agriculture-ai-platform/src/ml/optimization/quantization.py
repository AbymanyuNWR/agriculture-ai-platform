import torch
import torch.nn as nn
import torch.quantization as quantization
from typing import Dict, Optional, Tuple
import numpy as np

class ModelQuantizer:
    """Model quantization for inference optimization"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.quantized_model = None
        
    def dynamic_quantization(
        self,
        qconfig: str = 'qint8'
    ) -> nn.Module:
        """Dynamic quantization"""
        self.model.eval()
        
        # Select quantization config
        if qconfig == 'qint8':
            qconfig = quantization.get_default_qconfig('fbgemm')
        elif qconfig == 'float16':
            qconfig = quantization.float16_dynamic_qconfig
        
        # Prepare model
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            qconfig
        )
        
        return self.quantized_model
    
    def static_quantization(
        self,
        calibration_loader=None,
        qconfig: str = 'qint8'
    ) -> nn.Module:
        """Static quantization with calibration"""
        self.model.eval()
        
        # Set quantization config
        if qconfig == 'qint8':
            self.model.qconfig = quantization.get_default_qconfig('fbgemm')
        
        # Prepare model
        model_prep = quantization.prepare(self.model)
        
        # Calibration
        if calibration_loader is not None:
            with torch.no_grad():
                for inputs, _ in calibration_loader:
                    model_prep(inputs)
        
        # Convert
        self.quantized_model = quantization.convert(model_prep)
        
        return self.quantized_model
    
    def quantize_aware_training(
        self,
        train_loader,
        num_epochs: int = 1
    ) -> nn.Module:
        """Quantization-aware training"""
        self.model.train()
        
        # Set quantization config
        self.model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
        
        # Prepare QAT
        model_prep = quantization.prepare_qat(self.model)
        
        # Training loop
        optimizer = torch.optim.SGD(model_prep.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model_prep(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Convert
        model_prep.eval()
        self.quantized_model = quantization.convert(model_prep)
        
        return self.quantized_model
    
    def get_model_size(self, model: nn.Module = None) -> Dict[str, float]:
        """Get model size"""
        if model is None:
            model = self.model
        
        torch.save(model.state_dict(), '/tmp/temp_model.pth')
        import os
        size_mb = os.path.getsize('/tmp/temp_model.pth') / (1024 * 1024)
        os.remove('/tmp/temp_model.pth')
        
        return {'size_mb': size_mb}
    
    def compare_sizes(self) -> Dict[str, float]:
        """Compare original and quantized model sizes"""
        original_size = self.get_model_size(self.model)
        
        if self.quantized_model is not None:
            quantized_size = self.get_model_size(self.quantized_model)
            compression_ratio = original_size['size_mb'] / quantized_size['size_mb']
        else:
            quantized_size = {'size_mb': 0}
            compression_ratio = 0
        
        return {
            'original_size_mb': original_size['size_mb'],
            'quantized_size_mb': quantized_size['size_mb'],
            'compression_ratio': compression_ratio
        }


class FP16Quantizer:
    """FP16 quantization"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        
    def convert_to_fp16(self) -> nn.Module:
        """Convert model to FP16"""
        self.model.eval()
        self.model = self.model.half()
        return self.model
    
    def convert_to_mixed_precision(self) -> nn.Module:
        """Convert model to mixed precision"""
        self.model.eval()
        
        # Keep batch norm in FP32
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                module.float()
        
        return self.model


class ONNXExporter:
    """Export model to ONNX format"""
    
    def __init__(self, model: nn.Module, input_size: Tuple[int, int] = (224, 224)):
        self.model = model
        self.input_size = input_size
        
    def export(
        self,
        output_path: str,
        opset_version: int = 11,
        dynamic_axes: bool = True
    ):
        """Export model to ONNX"""
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, *self.input_size)
        
        # Dynamic axes
        if dynamic_axes:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        else:
            dynamic_axes = None
        
        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        
        print(f"Model exported to {output_path}")
        
    def validate_onnx(self, onnx_path: str) -> bool:
        """Validate ONNX model"""
        try:
            import onnx
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            print("ONNX model is valid")
            return True
        except Exception as e:
            print(f"ONNX validation failed: {e}")
            return False


class TensorRTExporter:
    """Export model to TensorRT"""
    
    def __init__(self, model: nn.Module, input_size: Tuple[int, int] = (224, 224)):
        self.model = model
        self.input_size = input_size
        
    def export_to_tensorrt(
        self,
        output_path: str,
        fp16: bool = True,
        max_batch_size: int = 32
    ):
        """Export to TensorRT via ONNX"""
        # First export to ONNX
        onnx_exporter = ONNXExporter(self.model, self.input_size)
        onnx_path = output_path.replace('.engine', '.onnx')
        onnx_exporter.export(onnx_path)
        
        # Convert to TensorRT (requires trtexec)
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
        
        print(f"TensorRT engine saved to {output_path}")


class TFLiteExporter:
    """Export model to TFLite"""
    
    def __init__(self, model: nn.Module, input_size: Tuple[int, int] = (224, 224)):
        self.model = model
        self.input_size = input_size
        
    def export_to_tflite(
        self,
        output_path: str,
        quantize: bool = True
    ):
        """Export to TFLite"""
        # First export to ONNX
        onnx_exporter = ONNXExporter(self.model, self.input_size)
        onnx_path = output_path.replace('.tflite', '.onnx')
        onnx_exporter.export(onnx_path, dynamic_axes=False)
        
        # Convert using onnx2tf
        import subprocess
        
        cmd = [
            'onnx2tf',
            '-i', onnx_path,
            '-o', output_path.replace('.tflite', '_tflite'),
            '-oiqt',  # OpenVINO IR quantized
        ]
        
        if quantize:
            cmd.append('--quantize')
        
        subprocess.run(cmd, check=True)
        
        print(f"TFLite model saved to {output_path}")


class ModelDistiller:
    """Knowledge distillation"""
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute distillation loss"""
        # Soft targets
        soft_student = torch.nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = torch.nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        
        # KL divergence loss
        distill_loss = torch.nn.functional.kl_div(
            soft_student, soft_teacher,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard target loss
        hard_loss = torch.nn.functional.cross_entropy(student_logits, labels)
        
        # Combined loss
        loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
        
        return loss
    
    def train_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Single training step"""
        self.teacher_model.eval()
        self.student_model.train()
        
        # Get teacher predictions
        with torch.no_grad():
            teacher_logits = self.teacher_model(inputs)
        
        # Get student predictions
        student_logits = self.student_model(inputs)
        
        # Compute loss
        loss = self.distillation_loss(student_logits, teacher_logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            _, student_pred = student_logits.max(1)
            accuracy = (student_pred == labels).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }


class FeatureDistiller:
    """Feature-level distillation"""
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        teacher_features_dim: int,
        student_features_dim: int
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        
        # Projection layer to match dimensions
        self.projector = nn.Linear(student_features_dim, teacher_features_dim)
        
    def feature_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute feature matching loss"""
        # Project student features
        projected = self.projector(student_features)
        
        # L2 loss
        loss = torch.nn.functional.mse_loss(projected, teacher_features)
        
        return loss
