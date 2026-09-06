from fastapi import APIRouter
from src.api.app.api.v1 import diagnosis, crops, users, sensors, knowledge

api_router = APIRouter()

api_router.include_router(
    diagnosis.router,
    prefix="/diagnosis",
    tags=["Diagnosis"]
)

api_router.include_router(
    crops.router,
    prefix="/crops",
    tags=["Crops"]
)

api_router.include_router(
    users.router,
    prefix="/users",
    tags=["Users"]
)

api_router.include_router(
    sensors.router,
    prefix="/sensors",
    tags=["Sensors"]
)

api_router.include_router(
    knowledge.router,
    prefix="/knowledge",
    tags=["Knowledge"]
)
