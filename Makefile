# Makefile for Solar-Suitability-Map
SHELL := /bin/bash

PY := .venv/bin/python
PIP := .venv/bin/pip

.DEFAULT_GOAL := help

.PHONY: help venv install download preprocess features train predict visualize run clean deep-clean

help: ## Show this help
	@echo "Usage: make <target>"
	@echo
	@echo "Targets:"
	@echo "  venv        Create virtual environment and install requirements"
	@echo "  install     Alias for venv"
	@echo "  download    Run scripts/download_data.py"
	@echo "  preprocess  Run scripts/preprocess.py"
	@echo "  features    Run scripts/extract_features.py"
	@echo "  train       Run scripts/train_model.py"
	@echo "  predict     Run scripts/predict_map.py"
	@echo "  visualize   Run scripts/visualize.py"
	@echo "  run         Run the full pipeline (download -> visualize)"
	@echo "  clean       Remove generated artifacts (keeps data/.gitkeep)"
	@echo "  deep-clean  Remove venv and generated data/models (DANGEROUS)"

venv: .venv/bin/python ## Create venv and install requirements

.venv/bin/python:
	@python3 -m venv .venv
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt

install: venv ## Install dependencies (alias)

download: venv ## Download data (Sentinel-2 composite, slope, GHI proxy)
	@$(PY) scripts/download_data.py $(DOWNLOAD_ARGS)

preprocess: venv ## Align/reproject rasters to common grid
	@$(PY) scripts/preprocess.py $(PREPROCESS_ARGS)

features: venv ## Compute NDVI, NDBI, slope, GHI -> features.npy
	@$(PY) scripts/extract_features.py $(FEATURES_ARGS)

train: venv ## Train Random Forest and save model
	@$(PY) scripts/train_model.py $(TRAIN_ARGS)

predict: venv ## Predict suitability map (GeoTIFF)
	@$(PY) scripts/predict_map.py $(PREDICT_ARGS)

visualize: venv ## Quick visualization
	@$(PY) scripts/visualize.py $(VISUALIZE_ARGS)

run: download preprocess features train predict visualize ## Full pipeline

clean: ## Remove generated artifacts (safe)
	@rm -f data/processed/suitability_map.tif
	@rm -f data/processed/features.npy
	@rm -f models/rf_model.joblib

deep-clean: ## Remove venv and generated data/models (DANGEROUS)
	@rm -rf .venv
	@find data -type f ! -name '.gitkeep' -delete 2>/dev/null || true
	@find models -type f ! -name '.gitkeep' -delete 2>/dev/null || true
