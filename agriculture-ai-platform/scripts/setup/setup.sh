#!/bin/bash

# Agriculture AI Platform Setup Script

set -e

echo "========================================="
echo "Agriculture AI Platform Setup"
echo "========================================="

# Check prerequisites
echo ""
echo "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed"
    exit 1
fi

# Check Poetry
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
fi

echo "All prerequisites found!"

# Setup environment
echo ""
echo "Setting up environment..."

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file from .env.example"
    echo "Please edit .env with your configuration"
fi

# Install dependencies
echo ""
echo "Installing Python dependencies..."
poetry install

# Start services
echo ""
echo "Starting Docker services..."
docker-compose -f infrastructure/docker/docker-compose.yml up -d

# Wait for services
echo ""
echo "Waiting for services to start..."
sleep 10

# Initialize database
echo ""
echo "Initializing database..."
poetry run alembic upgrade head

# Seed knowledge graph
echo ""
echo "Seeding knowledge graph..."
poetry run python scripts/utilities/seed_knowledge_graph.py

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env with your configuration"
echo "2. Run: poetry run uvicorn src.api.app.main:app --reload"
echo "3. Open http://localhost:8000/docs for API documentation"
echo ""
echo "Services:"
echo "- API: http://localhost:8000"
echo "- MLflow: http://localhost:5000"
echo "- Grafana: http://localhost:3001"
echo "- Neo4j: http://localhost:7474"
echo ""
