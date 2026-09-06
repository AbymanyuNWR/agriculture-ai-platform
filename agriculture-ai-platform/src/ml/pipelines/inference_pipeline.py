import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
import json

@dataclass
class InferenceResult:
    """Inference result"""
    prediction: Any
    confidence: float
    probabilities: Dict[str, float]
    bounding_boxes: Optional[List[Dict[str, float]]] = None
    segmentation_mask: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

class ImagePreprocessor:
    """Image preprocessing pipeline"""
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ):
        self.input_size = input_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        
    def preprocess(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """Preprocess image"""
        # Convert to tensor
        if isinstance(image, Image.Image):
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1)
            
        # Normalize
        image = image.float() / 255.0
        image = (image - self.mean.to(image.device)) / self.std.to(image.device)
        
        return image
        
    def batch_preprocess(
        self,
        images: List[Union[Image.Image, np.ndarray]]
    ) -> torch.Tensor:
        """Preprocess batch of images"""
        processed = [self.preprocess(img) for img in images]
        return torch.stack(processed)


class PredictionPostprocessor:
    """Postprocess predictions"""
    
    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
        top_k: int = 5
    ):
        self.class_names = class_names or []
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        
    def postprocess_classification(
        self,
        logits: torch.Tensor
    ) -> List[InferenceResult]:
        """Postprocess classification predictions"""
        probs = F.softmax(logits, dim=1)
        
        results = []
        for i in range(logits.shape[0]):
            sample_probs = probs[i]
            
            # Get top-k predictions
            top_k_probs, top_k_indices = torch.topk(sample_probs, min(self.top_k, len(sample_probs)))
            
            prob_dict = {}
            for prob, idx in zip(top_k_probs, top_k_indices):
                class_name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
                prob_dict[class_name] = prob.item()
                
            # Get prediction
            pred_idx = torch.argmax(sample_probs).item()
            pred_class = self.class_names[pred_idx] if pred_idx < len(self.class_names) else f"class_{pred_idx}"
            confidence = sample_probs[pred_idx].item()
            
            result = InferenceResult(
                prediction=pred_class,
                confidence=confidence,
                probabilities=prob_dict,
                metadata={"raw_logits": logits[i].cpu().numpy().tolist()}
            )
            results.append(result)
            
        return results
        
    def postprocess_detection(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        image_size: Tuple[int, int] = (640, 640)
    ) -> List[InferenceResult]:
        """Postprocess detection predictions"""
        results = []
        
        # Filter by confidence
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        # Group by image (assuming batch size 1 for simplicity)
        box_dicts = []
        for box, score, label in zip(boxes, scores, labels):
            box_dict = {
                "x1": box[0].item() / image_size[1],
                "y1": box[1].item() / image_size[0],
                "x2": box[2].item() / image_size[1],
                "y2": box[3].item() / image_size[0],
                "confidence": score.item(),
                "class_id": label.item(),
                "class_name": self.class_names[label.item()] if label.item() < len(self.class_names) else f"class_{label.item()}"
            }
            box_dicts.append(box_dict)
            
        # Sort by confidence
        box_dicts.sort(key=lambda x: x["confidence"], reverse=True)
        
        result = InferenceResult(
            prediction=box_dicts[0]["class_name"] if box_dicts else "no_detection",
            confidence=box_dicts[0]["confidence"] if box_dicts else 0.0,
            probabilities={},
            bounding_boxes=box_dicts
        )
        
        return [result]
        
    def postprocess_segmentation(
        self,
        logits: torch.Tensor,
        original_size: Tuple[int, int] = None
    ) -> List[InferenceResult]:
        """Postprocess segmentation predictions"""
        # Get class predictions
        predictions = torch.argmax(logits, dim=1)
        
        results = []
        for i in range(logits.shape[0]):
            pred_mask = predictions[i].cpu().numpy()
            
            # Compute per-class probabilities
            probs = F.softmax(logits[i], dim=0)
            class_probs = {}
            for c in range(logits.shape[1]):
                class_name = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
                class_probs[class_name] = probs[c].mean().item()
                
            # Resize mask if needed
            if original_size is not None:
                from torchvision.transforms import functional as TF
                mask_tensor = torch.from_numpy(pred_mask).unsqueeze(0).float()
                mask_tensor = F.interpolate(
                    mask_tensor.unsqueeze(0),
                    size=original_size,
                    mode='nearest'
                ).squeeze()
                pred_mask = mask_tensor.numpy().astype(int)
                
            result = InferenceResult(
                prediction=pred_mask,
                confidence=max(class_probs.values()) if class_probs else 0.0,
                probabilities=class_probs,
                segmentation_mask=pred_mask
            )
            results.append(result)
            
        return results


class InferencePipeline:
    """Complete inference pipeline"""
    
    def __init__(
        self,
        model: nn.Module,
        preprocessor: Optional[ImagePreprocessor] = None,
        postprocessor: Optional[PredictionPostprocessor] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_type: str = "classification"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.model_type = model_type
        
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.postprocessor = postprocessor or PredictionPostprocessor()
        
    def predict(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor, str],
        return_probabilities: bool = True
    ) -> InferenceResult:
        """Single image prediction"""
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            
        # Preprocess
        input_tensor = self.preprocessor.preprocess(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            
        # Postprocess
        if self.model_type == "classification":
            results = self.postprocessor.postprocess_classification(logits)
        elif self.model_type == "detection":
            # Detection models return different outputs
            results = self.postprocessor.postprocess_detection(
                logits['boxes'], logits['scores'], logits['labels']
            )
        elif self.model_type == "segmentation":
            original_size = (image.size[1], image.size[0]) if isinstance(image, Image.Image) else None
            results = self.postprocessor.postprocess_segmentation(logits, original_size)
        else:
            results = self.postprocessor.postprocess_classification(logits)
            
        return results[0]
        
    def predict_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, str]],
        batch_size: int = 32
    ) -> List[InferenceResult]:
        """Batch prediction"""
        all_results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            # Load images if paths
            loaded_batch = []
            for img in batch:
                if isinstance(img, str):
                    loaded_batch.append(Image.open(img).convert('RGB'))
                else:
                    loaded_batch.append(img)
                    
            # Preprocess
            input_tensors = torch.stack([
                self.preprocessor.preprocess(img) for img in loaded_batch
            ]).to(self.device)
            
            # Inference
            with torch.no_grad():
                logits = self.model(input_tensors)
                
            # Postprocess
            if self.model_type == "classification":
                batch_results = self.postprocessor.postprocess_classification(logits)
            else:
                batch_results = self.postprocessor.postprocess_classification(logits)
                
            all_results.extend(batch_results)
            
        return all_results
        
    def predict_with_uncertainty(
        self,
        image: Union[Image.Image, np.ndarray],
        num_samples: int = 10
    ) -> Tuple[InferenceResult, float]:
        """Prediction with uncertainty estimation (MC Dropout)"""
        self.model.train()  # Enable dropout
        
        input_tensor = self.preprocessor.preprocess(image).unsqueeze(0).to(self.device)
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
                
        predictions = torch.stack(predictions)
        
        # Mean prediction
        mean_probs = predictions.mean(dim=0)
        
        # Uncertainty (entropy of mean)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
        
        self.model.eval()
        
        # Get result
        results = self.postprocessor.postprocess_classification(mean_probs)
        result = results[0]
        result.metadata = {"uncertainty": entropy.item()}
        
        return result, entropy.item()
        
    def explain_prediction(
        self,
        image: Union[Image.Image, np.ndarray],
        target_class: Optional[int] = None
    ):
        """Explain prediction using GradCAM"""
        from ..evaluation.explainability.gradcam import GradCAM
        
        input_tensor = self.preprocessor.preprocess(image).unsqueeze(0).to(self.device)
        
        # Get model's last conv layer
        if hasattr(self.model, 'features'):
            target_layer = self.model.features[-1]
        elif hasattr(self.model, 'layer4'):
            target_layer = self.model.layer4
        else:
            raise ValueError("Cannot find target layer for GradCAM")
            
        gradcam = GradCAM(self.model, target_layer)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            
        if target_class is None:
            target_class = torch.argmax(output).item()
            
        # Generate heatmap
        heatmap = gradcam.generate(input_tensor, target_class)
        
        return {
            "prediction": target_class,
            "confidence": F.softmax(output, dim=1)[0, target_class].item(),
            "heatmap": heatmap
        }


class RealtimeInference:
    """Real-time inference with buffering"""
    
    def __init__(
        self,
        pipeline: InferencePipeline,
        buffer_size: int = 10
    ):
        self.pipeline = pipeline
        self.buffer_size = buffer_size
        self.buffer = []
        
    def process_frame(
        self,
        frame: np.ndarray
    ) -> Optional[InferenceResult]:
        """Process a single frame"""
        self.buffer.append(frame)
        
        if len(self.buffer) >= self.buffer_size:
            # Process buffer
            result = self.pipeline.predict_batch(self.buffer)
            self.buffer = []
            return result[-1] if result else None
            
        return None
        
    def get_streaming_results(
        self,
        video_source: str = 0
    ):
        """Stream results from video source"""
        import cv2
        
        cap = cv2.VideoCapture(video_source)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.process_frame(frame_rgb)
            
            if result:
                yield result
                
        cap.release()


class ModelEnsembleInference:
    """Ensemble inference"""
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        postprocessor: Optional[PredictionPostprocessor] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.models = [m.to(device).eval() for m in models]
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.postprocessor = postprocessor or PredictionPostprocessor()
        self.device = device
        
    def predict(
        self,
        image: Union[Image.Image, np.ndarray]
    ) -> InferenceResult:
        """Ensemble prediction"""
        input_tensor = self.preprocessor.preprocess(image).unsqueeze(0).to(self.device)
        
        all_logits = []
        
        for model, weight in zip(self.models, self.weights):
            with torch.no_grad():
                logits = model(input_tensor)
                all_logits.append(logits * weight)
                
        # Average logits
        ensemble_logits = torch.stack(all_logits).sum(dim=0)
        
        # Postprocess
        results = self.postprocessor.postprocess_classification(ensemble_logits)
        
        return results[0]


class InferenceCache:
    """Cache inference results"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        
    def get_hash(self, image: np.ndarray) -> str:
        """Get image hash"""
        import hashlib
        return hashlib.md5(image.tobytes()).hexdigest()
        
    def get(self, image: np.ndarray) -> Optional[InferenceResult]:
        """Get cached result"""
        img_hash = self.get_hash(image)
        return self.cache.get(img_hash)
        
    def set(self, image: np.ndarray, result: InferenceResult):
        """Cache result"""
        if len(self.cache) >= self.max_size:
            # Remove oldest
            self.cache.pop(next(iter(self.cache)))
            
        img_hash = self.get_hash(image)
        self.cache[img_hash] = result
        
    def clear(self):
        """Clear cache"""
        self.cache.clear()
