import pytest
import uuid
from datetime import datetime

from src.api.app.services.diagnosis_service import DiagnosisService
from src.api.app.services.knowledge_service import KnowledgeService

class TestDiagnosisService:
    def setup_method(self):
        self.service = DiagnosisService()
    
    @pytest.mark.asyncio
    async def test_predict_diagnosis(self):
        # Create dummy image
        from PIL import Image
        import io
        
        image = Image.new('RGB', (224, 224), color='red')
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        result = await self.service.predict(
            image_bytes=image_bytes,
            crop_type="padi",
            user_id="test_user"
        )
        
        assert result.id is not None
        assert result.diagnosis is not None
        assert 0 <= result.confidence <= 1
        assert result.severity in ["low", "medium", "high"]
    
    @pytest.mark.asyncio
    async def test_get_history(self):
        history = await self.service.get_history(
            user_id="test_user",
            page=1,
            limit=10
        )
        
        assert "diagnoses" in history
        assert "total" in history

class TestKnowledgeService:
    def setup_method(self):
        self.service = KnowledgeService()
    
    def test_get_disease_info(self):
        info = self.service.get_disease_info("Blast", "Padi")
        
        assert "symptoms" in info
        assert "causes" in info
        assert "treatments" in info
    
    def test_get_default_info(self):
        info = self.service.get_default_info("Unknown Disease")
        
        assert "symptoms" in info
        assert "causes" in info
