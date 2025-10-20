.PHONY: help install install-dev test test-cov lint format type-check clean build docs serve-docs
.DEFAULT_GOAL := help

PYTHON := python3
PIP := pip
PYTEST := pytest
BLACK := black
FLAKE8 := flake8
MYPY := mypy
SPHINX := sphinx-build

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	$(PIP) install -e .

install-dev: ## Install the package with development dependencies
	$(PIP) install -e ".[dev]"

install-all: ## Install the package with all dependencies
	$(PIP) install -e ".[all]"

test: ## Run tests
	$(PYTEST) tests/ -v

test-cov: ## Run tests with coverage
	$(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term

lint: ## Run linting checks
	$(FLAKE8) src/ tests/

format: ## Format code with black
	$(BLACK) src/ tests/

format-check: ## Check code formatting
	$(BLACK) --check src/ tests/

type-check: ## Run type checking with mypy
	$(MYPY) src/

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build the package
	$(PYTHON) -m build

install-pre-commit: ## Install pre-commit hooks
	pre-commit install

run-pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

update-pre-commit: ## Update pre-commit hooks
	pre-commit autoupdate

benchmark: ## Run benchmark tests
	$(PYTEST) tests/ -m "benchmark" -v

integration-test: ## Run integration tests
	$(PYTEST) tests/ -m "integration" -v

unit-test: ## Run unit tests only
	$(PYTEST) tests/ -m "unit" -v

profile: ## Run performance profiling
	$(PYTHON) -m cProfile -o profile.stats -m pytest tests/
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

docs: ## Build documentation
	cd docs && $(SPHINX) -b html . _build

serve-docs: docs ## Serve documentation locally
	cd docs/_build && $(PYTHON) -m http.server 8000

check-deps: ## Check for dependency updates
	$(PIP) list --outdated

security-check: ## Run security checks
	bandit -r src/

type-coverage: ## Generate type coverage report
	$(MYPY) src/ --cobertura-xml-report type_coverage

complexity: ## Check code complexity
	xenon --max-absolute A --max-modules A --max-average A src/

all-checks: format-check lint type-check test security-check ## Run all quality checks

release-check: ## Check if ready for release
	@echo "Checking if ready for release..."
	$(MAKE) clean
	$(MAKE) all-checks
	@echo "Checking version consistency..."
	@$(PYTHON) -c "import src.hybrid_retrieval; print(f'Version: {src.hybrid_retrieval.__version__}')"
	@echo "Release check completed successfully!"

quick-test: ## Run quick tests (unit tests only)
	$(PYTEST) tests/ -m "not slow" --tb=short

debug-test: ## Run tests with debugging enabled
	$(PYTEST) tests/ --pdb --tb=short

jupyter: ## Launch Jupyter notebook
	jupyter notebook examples/ || echo "Jupyter not installed. Install with: pip install jupyter"

demo: ## Run demo script
	$(PYTHON) examples/demo.py || echo "Demo script not found. Create examples/demo.py first"

install-demo-deps: ## Install dependencies for demos
	$(PIP) install jupyter matplotlib seaborn pandas

help-advanced: ## Show advanced help
	@echo "Advanced commands:"
	@echo "  profile          - Run performance profiling"
	@echo "  type-coverage    - Generate type coverage report"
	@echo "  complexity       - Check code complexity"
	@echo "  all-checks       - Run all quality checks"
	@echo "  release-check    - Check if ready for release"
	@echo "  quick-test       - Run quick tests (unit tests only)"
	@echo "  debug-test       - Run tests with debugging enabled"
	@echo "  jupyter          - Launch Jupyter notebook"
	@echo "  demo             - Run demo script"
	@echo "  install-demo-deps - Install demo dependencies"

.DEFAULT:
	@echo "Unknown target: $@"
	@echo "Run 'make help' for available targets"

.SILENT: help help-advanced