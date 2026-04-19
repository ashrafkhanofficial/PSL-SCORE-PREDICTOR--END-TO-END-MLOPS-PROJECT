# Makefile for PSL Score Predictor

.PHONY: help install install-dev test lint format clean data train predict

help:
	@echo "Available commands:"
	@echo "  make install          - Install project dependencies"
	@echo "  make install-dev      - Install project and development dependencies"
	@echo "  make test             - Run tests"
	@echo "  make lint             - Run code linters"
	@echo "  make format           - Format code with black"
	@echo "  make clean            - Remove build artifacts and cache"
	@echo "  make data             - Generate dataset"
	@echo "  make train            - Train the model"
	@echo "  make predict          - Make predictions"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest

lint:
	flake8 src tests
	mypy src

format:
	black src tests config

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build dist *.egg-info
	rm -rf .coverage htmlcov
	rm -rf .pytest_cache

data:
	@echo "Data generation not yet implemented"

train:
	@echo "Model training script not yet implemented"

predict:
	@echo "Prediction script not yet implemented"
