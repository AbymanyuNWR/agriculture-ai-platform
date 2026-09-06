import uuid
from typing import Optional
from datetime import datetime

from src.api.app.schemas.diagnosis import DiagnosisResponse
from src.api.app.ml.pipelines.inference_pipeline import InferencePipeline
from src.api.app.services.knowledge_service import KnowledgeService

class DiagnosisService:
    def __init__(self):
        self.inference_pipeline = InferencePipeline()
        self.knowledge_service = KnowledgeService()
    
    async def predict(
        self,
        image_bytes: bytes,
        crop_type: Optional[str] = None,
        location: Optional[dict] = None,
        user_id: str = None
    ) -> DiagnosisResponse:
        # Run ML inference
        prediction = await self.inference_pipeline.predict(
            image_bytes=image_bytes,
            crop_type=crop_type
        )
        
        # Get knowledge enrichment
        knowledge = await self.knowledge_service.get_disease_info(
            disease_name=prediction["diagnosis"],
            crop_type=crop_type
        )
        
        # Generate recommendations
        recommendations = await self.generate_recommendations(
            prediction=prediction,
            knowledge=knowledge,
            location=location
        )
        
        # Create response
        response = DiagnosisResponse(
            id=uuid.uuid4(),
            crop_type=crop_type or prediction.get("crop_type", "unknown"),
            diagnosis=prediction["diagnosis"],
            confidence=prediction["confidence"],
            severity=self.determine_severity(prediction["confidence"]),
            symptoms=knowledge.get("symptoms", []),
            causes=knowledge.get("causes", []),
            recommendations=recommendations,
            treatment_options=knowledge.get("treatments", []),
            prevention_tips=knowledge.get("prevention", []),
            environmental_factors=knowledge.get("environmental_factors", {}),
            created_at=datetime.now()
        )
        
        # Save to database
        await self.save_diagnosis(response, user_id)
        
        return response
    
    def determine_severity(self, confidence: float) -> str:
        if confidence > 0.9:
            return "high"
        elif confidence > 0.7:
            return "medium"
        else:
            return "low"
    
    async def generate_recommendations(
        self,
        prediction: dict,
        knowledge: dict,
        location: Optional[dict]
    ) -> list:
        recommendations = []
        
        # Add immediate actions
        if prediction["confidence"] > 0.8:
            recommendations.append("Segera lakukan treatment sesuai diagnosis")
        
        # Add prevention tips
        recommendations.extend(knowledge.get("prevention", []))
        
        # Add environmental recommendations
        if location:
            weather_recommendations = await self.get_weather_recommendations(location)
            recommendations.extend(weather_recommendations)
        
        return recommendations
    
    async def get_history(
        self,
        user_id: str,
        page: int = 1,
        limit: int = 10
    ) -> dict:
        # Query database for user's diagnosis history
        # This is a placeholder - implement actual database query
        return {
            "diagnoses": [],
            "total": 0,
            "page": page,
            "limit": limit
        }
    
    async def get_by_id(
        self,
        diagnosis_id: uuid.UUID,
        user_id: str
    ) -> Optional[DiagnosisResponse]:
        # Query database for specific diagnosis
        # This is a placeholder - implement actual database query
        return None
    
    async def save_diagnosis(self, diagnosis: DiagnosisResponse, user_id: str):
        # Save to database
        pass
