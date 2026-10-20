SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

.PHONY: install
install: ## Install Python dependencies and the Chromium browser used by the Web-browser agent
	pip install -r requirements.txt
	playwright install chromium --with-deps --no-shell

.PHONY: test
test: ## Run the offline unit tests (no API keys required)
	python -m pytest tests -q

.PHONY: run
run: ## Run EvoMAS on the example tasks
	python run.py --config configs/evomas.py

.PHONY: train
train: ## Train the Workflow Adapter on the example tasks
	python train.py --config configs/evomas_train.py

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-10s %s\n", $$1, $$2}'
