.PHONY: install test lint format clean train

# ── Install ───────────────────────────────────────────────────────
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pip install pytest pytest-cov black mypy flake8

# ── Testing ───────────────────────────────────────────────────────
test:
	PYTHONPATH=. python3 tests/test_model_quality.py

pytest:
	PYTHONPATH=. pytest tests/ -v

test-coverage:
	PYTHONPATH=. pytest tests/ -v --cov=src --cov-report=html

# ── Training ──────────────────────────────────────────────────────
train:
	PYTHONPATH=. python3 src/training/trainer.py

train-fast:
	PYTHONPATH=. python3 src/training/trainer.py --epochs 10 --batch-size 32

# ── Data Collection ───────────────────────────────────────────────
collect-data:
	PYTHONPATH=. python3 scripts/data_generation/collect_training_data.py \
		--num-programs 20 \
		--transforms-per-program 100

# ── Code Quality ──────────────────────────────────────────────────
lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E501

format:
	black src/ tests/ scripts/ --line-length=100

type-check:
	mypy src/ --ignore-missing-imports

# ── Cleanup ───────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ htmlcov/ 2>/dev/null || true

# ── Help ──────────────────────────────────────────────────────────
help:
	@echo "NS-IR Compiler — Available Commands:"
	@echo "  make install       Install package in development mode"
	@echo "  make test          Run quality test suite"
	@echo "  make pytest        Run full pytest suite"
	@echo "  make train         Train the Transformer cost model (50 epochs)"
	@echo "  make train-fast    Quick training run (10 epochs)"
	@echo "  make collect-data  Collect real hardware training data"
	@echo "  make lint          Run flake8 linter"
	@echo "  make format        Auto-format code with black"
	@echo "  make type-check    Run mypy type checker"
	@echo "  make clean         Remove build artifacts"
