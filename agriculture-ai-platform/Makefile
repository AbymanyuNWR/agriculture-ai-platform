.PHONY: help install dev test lint format run deploy clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	poetry install

dev: ## Start development server
	poetry run uvicorn src.api.app.main:app --reload --host 0.0.0.0 --port 8000

test: ## Run tests
	poetry run pytest tests/ -v --cov=src --cov-report=html

test-unit: ## Run unit tests
	poetry run pytest tests/unit/ -v

test-integration: ## Run integration tests
	poetry run pytest tests/integration/ -v

lint: ## Run linting
	poetry run black --check .
	poetry run ruff .
	poetry run mypy .

format: ## Format code
	poetry run black .
	poetry run ruff --fix .

run: ## Run production server
	poetry run uvicorn src.api.app.main:app --host 0.0.0.0 --port 8000 --workers 4

train: ## Train model
	poetry run python scripts/training/train_disease_classifier.py

evaluate: ## Evaluate model
	poetry run python -m src.ml.evaluation.evaluate_model

deploy-staging: ## Deploy to staging
	./scripts/deployment/deploy.sh staging

deploy-production: ## Deploy to production
	./scripts/deployment/deploy.sh production

seed-db: ## Seed knowledge graph
	poetry run python scripts/utilities/seed_knowledge_graph.py

health-check: ## Run health check
	poetry run python scripts/utilities/health_check.py

report: ## Generate system report
	poetry run python scripts/utilities/generate_report.py

docker-build: ## Build Docker images
	docker build -t agriculture-api -f infrastructure/docker/Dockerfile.api .
	docker build -t agriculture-ml -f infrastructure/docker/Dockerfile.ml .

docker-up: ## Start Docker services
	docker-compose -f infrastructure/docker/docker-compose.yml up -d

docker-down: ## Stop Docker services
	docker-compose -f infrastructure/docker/docker-compose.yml down

docker-logs: ## View Docker logs
	docker-compose -f infrastructure/docker/docker-compose.yml logs -f

clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf coverage.xml
	rm -rf reports

setup: ## Setup development environment
	./scripts/setup/setup.sh

db-migrate: ## Run database migrations
	poetry run alembic upgrade head

db-revision: ## Create new migration
	poetry run alembic revision --autogenerate -m "$(msg)"
