from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.api.app.config import settings
from src.api.app.api.v1.router import api_router
from src.api.app.core.database import init_db
from src.api.app.core.middleware import MonitoringMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    yield
    # Shutdown

app = FastAPI(
    title="Agriculture AI Platform",
    description="AI-powered crop disease detection and smart farming platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monitoring
app.add_middleware(MonitoringMiddleware)

# API Routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {
        "message": "Agriculture AI Platform",
        "version": "1.0.0",
        "docs": "/docs"
    }
