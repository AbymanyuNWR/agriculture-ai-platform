import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
        
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate CAM for input image"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Pool gradients
        weights = gradients.mean(dim=[1, 2], keepdim=True)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=0)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam = cv2.resize(cam.cpu().numpy(), (input_tensor.shape[2], input_tensor.shape[3]))
        
        return cam
    
    def generate_cam_batch(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> List[np.ndarray]:
        """Generate CAM for batch of images"""
        batch_size = input_tensor.shape[0]
        cams = []
        
        for i in range(batch_size):
            cam = self.generate_cam(input_tensor[i:i+1], target_class)
            cams.append(cam)
        
        return cams
    
    def overlay_cam(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Overlay CAM on image"""
        # Convert image to uint8 if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize cam to match image size
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # Apply colormap
        cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = cv2.addWeighted(image, 1 - alpha, cam_colored, alpha, 0)
        
        return overlay
    
    def save_cam(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        output_path: str,
        alpha: float = 0.5
    ):
        """Save CAM visualization"""
        overlay = self.overlay_cam(image, cam, alpha)
        
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title('GradCAM')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ implementation"""
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate Grad-CAM++"""
        self.model.eval()
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Grad-CAM++ weights
        gradients_power_2 = gradients ** 2
        gradients_power_3 = gradients ** 3
        
        sum_activations = activations.sum(dim=[1, 2], keepdim=True)
        
        numerator = gradients_power_2
        denominator = 2 * gradients_power_2 + sum_activations * gradients_power_3 + 1e-8
        
        weights = numerator / denominator
        
        cam = (weights * activations).sum(dim=0)
        
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        cam = cv2.resize(cam.cpu().numpy(), (input_tensor.shape[2], input_tensor.shape[3]))
        
        return cam


class GuidedBackpropagation:
    """Guided Backpropagation"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradients = None
        self.inputs = None
        
        self._register_hooks()
        
    def _register_hooks(self):
        """Register hooks for guided backpropagation"""
        def forward_hook(module, input, output):
            self.inputs = input[0].detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Register hooks for ReLU layers
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                
    def generate_guided_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate guided gradients"""
        self.model.eval()
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and inputs
        gradients = self.gradients
        inputs = self.inputs
        
        # Guided backpropagation
        positive_gradients = (gradients > 0).float()
        guided_gradients = inputs * positive_gradients
        
        return guided_gradients[0].cpu().numpy()


class IntegratedGradients:
    """Integrated Gradients"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        
    def generate_integrated_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        n_steps: int = 50,
        n_samples: int = 10
    ) -> np.ndarray:
        """Generate integrated gradients"""
        self.model.eval()
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Create baseline (black image)
        baseline = torch.zeros_like(input_tensor)
        
        # Generate interpolated images
        interpolated_images = []
        for i in range(n_steps + 1):
            alpha = i / n_steps
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated_images.append(interpolated)
        
        # Calculate gradients for each interpolated image
        all_gradients = []
        for interp_img in interpolated_images:
            interp_img.requires_grad_(True)
            
            output = self.model(interp_img)
            
            self.model.zero_grad()
            
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1
            output.backward(gradient=one_hot, retain_graph=True)
            
            all_gradients.append(interp_img.grad.detach())
        
        # Average gradients
        avg_gradients = torch.stack(all_gradients).mean(dim=0)
        
        # Calculate integrated gradients
        integrated_gradients = (input_tensor - baseline) * avg_gradients
        
        return integrated_gradients[0].cpu().numpy()


class AttentionMap:
    """Attention Map extraction for attention-based models"""
    
    def __init__(self, model: nn.Module, attention_layer: nn.Module):
        self.model = model
        self.attention_layer = attention_layer
        
        self.attentions = None
        
        self._register_hooks()
        
    def _register_hooks(self):
        """Register hook for attention extraction"""
        def hook(module, input, output):
            # For multi-head attention, we need to handle the attention weights
            if hasattr(module, 'attention_weights'):
                self.attentions = module.attention_weights.detach()
            else:
                # Try to get attention from output
                self.attentions = output.detach()
        
        self.attention_layer.register_forward_hook(hook)
        
    def generate_attention_map(
        self,
        input_tensor: torch.Tensor,
        head_index: Optional[int] = None
    ) -> np.ndarray:
        """Generate attention map"""
        self.model.eval()
        
        with torch.no_grad():
            self.model(input_tensor)
        
        if self.attentions is None:
            raise ValueError("No attention weights captured")
        
        # Handle different attention formats
        if len(self.attentions.shape) == 4:
            # Batch, heads, seq_len, seq_len
            if head_index is not None:
                attention = self.attentions[0, head_index]
            else:
                # Average over heads
                attention = self.attentions[0].mean(dim=0)
        else:
            attention = self.attentions[0]
        
        # Remove CLS token attention if present
        if attention.shape[0] > 1:
            attention = attention[1:, 1:]
        
        # Reshape to spatial dimensions
        seq_len = attention.shape[0]
        spatial_size = int(np.sqrt(seq_len))
        
        if spatial_size * spatial_size == seq_len:
            attention = attention.reshape(spatial_size, spatial_size)
        
        # Normalize
        attention = attention - attention.min()
        attention = attention / (attention.max() + 1e-8)
        
        return attention.cpu().numpy()
