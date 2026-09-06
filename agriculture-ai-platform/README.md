# Agriculture AI Platform

Platform AI untuk deteksi penyakit tanaman dan smart farming.

## Fitur Utama

- **Disease Detection**: Deteksi penyakit tanaman menggunakan Computer Vision
- **Pest Detection**: Identifikasi hama pada tanaman
- **Smart Irrigation**: Sistem irigasi cerdas berbasis sensor
- **Knowledge Graph**: Pengetahuan tentang penyakit dan treatment
- **Digital Twin**: Simulasi kondisi pertanian
- **LLM Integration**: Konsultasi dengan AI dalam bahasa natural

## Arsitektur

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│    (Mobile App / Web Dashboard / SMS Gateway)           │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    API Gateway                          │
│              (FastAPI + Rate Limiting)                  │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  ML Service  │  │  Knowledge   │  │   Weather    │
│  (PyTorch)   │  │   Graph      │  │    API       │
└──────────────┘  └──────────────┘  └──────────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    Data Layer                            │
│  (PostgreSQL + Redis + Neo4j + MinIO)                   │
└─────────────────────────────────────────────────────────┘
```

## Tech Stack

- **Backend**: FastAPI, SQLAlchemy, PostgreSQL
- **ML**: PyTorch, torchvision, scikit-learn
- **Knowledge Graph**: Neo4j
- **MLOps**: MLflow, Prefect
- **Monitoring**: Prometheus, Grafana
- **Infrastructure**: Docker, Kubernetes, Terraform

## Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- PostgreSQL
- Redis
- Neo4j

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/agriculture-ai-platform.git
cd agriculture-ai-platform

# Install dependencies
poetry install

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Start services
docker-compose up -d

# Run migrations
alembic upgrade head

# Start development server
uvicorn src.api.app.main:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development

### Code Style

```bash
# Format code
poetry run black .

# Lint code
poetry run ruff .

# Type check
poetry run mypy .
```

### Testing

```bash
# Run unit tests
poetry run pytest tests/unit/ -v

# Run integration tests
poetry run pytest tests/integration/ -v

# Run with coverage
poetry run pytest tests/ --cov=src --cov-report=xml
```

## Deployment

### Kubernetes

```bash
# Deploy to staging
kubectl apply -k infrastructure/kubernetes/overlays/staging/

# Deploy to production
kubectl apply -k infrastructure/kubernetes/overlays/production/
```

### Docker

```bash
# Build images
docker build -t agriculture-api -f infrastructure/docker/Dockerfile.api .
docker build -t agriculture-ml -f infrastructure/docker/Dockerfile.ml .

# Run with docker-compose
docker-compose -f infrastructure/docker/docker-compose.yml up -d
```

## Project Structure

```
agriculture-ai-platform/
├── src/
│   ├── api/           # FastAPI application
│   ├── ml/            # Machine Learning models
│   ├── web/           # Frontend application
│   └── shared/        # Shared utilities
├── infrastructure/    # Infrastructure configs
├── mlops/             # MLOps pipelines
├── tests/             # Test suites
├── notebooks/         # Jupyter notebooks
└── docs/              # Documentation
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- Email: team@agriculture-ai.com
- Slack: #agriculture-ai
