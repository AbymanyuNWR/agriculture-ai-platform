from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import uuid

class DiagnosisRequest(BaseModel):
    crop_type: Optional[str] = None
    location: Optional[dict] = None

class DiagnosisResponse(BaseModel):
    id: uuid.UUID
    crop_type: str
    diagnosis: str
    confidence: float
    severity: str
    symptoms: List[str]
    causes: List[str]
    recommendations: List[str]
    treatment_options: List[dict]
    prevention_tips: List[str]
    environmental_factors: dict
    created_at: datetime
    
    class Config:
        from_attributes = True

class DiagnosisHistory(BaseModel):
    diagnoses: List[DiagnosisResponse]
    total: int
    page: int
    limit: int
