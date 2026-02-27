SHELL := /bin/bash

.PHONY: setup lint test smoke

setup:
	python -m pip install -U pip
	pip install -r requirements.txt
	pip install -e .

lint:
	ruff check .

test:
	pytest

smoke:
	distill-safe pipeline run --config configs/pipelines/smoke_test.yaml
