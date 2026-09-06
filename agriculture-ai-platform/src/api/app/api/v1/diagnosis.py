from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import Optional
import uuid

from src.api.app.schemas.diagnosis import DiagnosisRequest, DiagnosisResponse
from src.api.app.services.diagnosis_service import DiagnosisService
from src.api.app.core.security import get_current_user

router = APIRouter()

@router.post("/predict", response_model=DiagnosisResponse)
async def predict_disease(
    image: UploadFile = File(...),
    crop_type: Optional[str] = None,
    location: Optional[dict] = None,
    current_user: dict = Depends(get_current_user)
):
    diagnosis_service = DiagnosisService()
    
    # Validate image
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image
    image_bytes = await image.read()
    
    # Run diagnosis
    result = await diagnosis_service.predict(
        image_bytes=image_bytes,
        crop_type=crop_type,
        location=location,
        user_id=current_user["user_id"]
    )
    
    return result

@router.get("/history")
async def get_diagnosis_history(
    page: int = 1,
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    diagnosis_service = DiagnosisService()
    history = await diagnosis_service.get_history(
        user_id=current_user["user_id"],
        page=page,
        limit=limit
    )
    return history

@router.get("/{diagnosis_id}")
async def get_diagnosis(
    diagnosis_id: uuid.UUID,
    current_user: dict = Depends(get_current_user)
):
    diagnosis_service = DiagnosisService()
    diagnosis = await diagnosis_service.get_by_id(
        diagnosis_id=diagnosis_id,
        user_id=current_user["user_id"]
    )
    if not diagnosis:
        raise HTTPException(status_code=404, detail="Diagnosis not found")
    return diagnosis
