import pytest
import torch
import numpy as np
from PIL import Image

from src.ml.models.classification.disease_classifier import DiseaseClassifier
from src.ml.serving.preprocessors.image_preprocessor import ImagePreprocessor
from src.ml.serving.postprocessors.prediction_postprocessor import PredictionPostprocessor

class TestDiseaseClassifier:
    def setup_method(self):
        self.model = DiseaseClassifier(num_classes=10)
        self.model.eval()
    
    def test_model_initialization(self):
        assert self.model is not None
        
    def test_model_forward(self):
        dummy_input = torch.randn(1, 3, 224, 224)
        logits, confidence = self.model(dummy_input)
        
        assert logits.shape == (1, 10)
        assert confidence.shape == (1, 1)
    
    def test_model_predict_with_confidence(self):
        dummy_input = torch.randn(1, 3, 224, 224)
        result = self.model.predict_with_confidence(dummy_input)
        
        assert "predicted_class" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert 0 <= result["confidence"] <= 1
        assert len(result["probabilities"]) == 10

class TestImagePreprocessor:
    def setup_method(self):
        self.preprocessor = ImagePreprocessor()
    
    def test_process_image(self):
        # Create dummy image
        image = Image.new('RGB', (224, 224), color='red')
        
        processed = self.preprocessor.process(image)
        
        assert processed.shape == (3, 224, 224)
        assert processed.dtype == np.float32
    
    def test_process_batch(self):
        images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='green'),
        ]
        
        processed = self.preprocessor.process_batch(images)
        
        assert processed.shape == (2, 3, 224, 224)

class TestPredictionPostprocessor:
    def setup_method(self):
        self.postprocessor = PredictionPostprocessor()
    
    def test_process_prediction(self):
        probabilities = np.random.rand(1, 10)
        probabilities = probabilities / probabilities.sum()  # Normalize
        
        result = self.postprocessor.process(probabilities)
        
        assert "diagnosis" in result
        assert "confidence" in result
        assert "severity" in result
        assert "top_3_predictions" in result
        assert 0 <= result["confidence"] <= 1
    
    def test_determine_severity(self):
        assert self.postprocessor.determine_severity(0.95) == "high"
        assert self.postprocessor.determine_severity(0.75) == "medium"
        assert self.postprocessor.determine_severity(0.5) == "low"
