PYTHON ?= ./venv/bin/python
PIP ?= ./venv/bin/pip
UVICORN ?= ./venv/bin/uvicorn
STREAMLIT ?= ./venv/bin/streamlit
COMPOSE ?= docker compose

.PHONY: help install data etl evaluate evaluate-smoke train train-eval api ui test check clean docker-build docker-setup docker-up docker-down docker-logs

help:
	@echo "Movie recommender workflow"
	@echo ""
	@echo "Setup and data:"
	@echo "  make install         Install Python dependencies"
	@echo "  make data            Download MovieLens 100k and run ETL"
	@echo "  make etl             Rebuild cleaned dataset from data/ml-100k"
	@echo ""
	@echo "Research:"
	@echo "  make evaluate        Run full temporal model comparison"
	@echo "  make evaluate-smoke  Run a fast model-comparison smoke test"
	@echo ""
	@echo "Production artifact:"
	@echo "  make train           Train SVD artifact for API"
	@echo "  make train-eval      Validate SVD, then train final artifact"
	@echo ""
	@echo "Serving interface:"
	@echo "  make api             Start FastAPI server"
	@echo "  make ui              Start Streamlit app"
	@echo ""
	@echo "Container workflow:"
	@echo "  make docker-build    Build Docker image"
	@echo "  make docker-setup    Download data and train model in Docker"
	@echo "  make docker-up       Start API and UI with Docker Compose"
	@echo "  make docker-down     Stop Docker Compose services"
	@echo "  make docker-logs     Follow Docker Compose logs"
	@echo ""
	@echo "Quality:"
	@echo "  make test            Run tests"
	@echo "  make check           Run tests and syntax checks"
	@echo "  make clean           Remove Python caches"

install:
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) download_data.py
	$(PYTHON) src/etl.py

etl:
	$(PYTHON) src/etl.py

evaluate:
	$(PYTHON) -m src.compare_models

evaluate-smoke:
	$(PYTHON) -m src.compare_models --sample-rows 500 --n-splits 2 --no-mlflow --output /tmp/model_comparison_smoke.csv

train:
	$(PYTHON) -m src.train_svd --no-eval

train-eval:
	$(PYTHON) -m src.train_svd

api:
	$(UVICORN) src.api:app --reload

ui:
	$(STREAMLIT) run src/streamlit_app.py

docker-build:
	$(COMPOSE) build

docker-setup:
	$(COMPOSE) run --rm setup

docker-up:
	$(COMPOSE) up api ui

docker-down:
	$(COMPOSE) down

docker-logs:
	$(COMPOSE) logs -f

test:
	$(PYTHON) -m pytest tests

check: test
	$(PYTHON) -m py_compile src/evaluation.py src/compare_models.py src/recommender_service.py src/api.py src/train_svd.py src/streamlit_app.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
