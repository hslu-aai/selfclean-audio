.DEFAULT_GOAL := help

###########################
# HELP
###########################
include *.mk

# Development commands (all run in Docker)
_docker_exec = @docker compose exec app /bin/bash -lc "$(1)" || (echo "Container not running. Start with 'make up' first." && exit 1)

.PHONY: test
test:  ##@Development run tests with coverage
	$(call _docker_exec,tox -e tests)

.PHONY: lint
lint:  ##@Development run linting and formatting checks
	$(call _docker_exec,tox -e pre-commit)

.PHONY: typecheck
typecheck:  ##@Development run type checking
	$(call _docker_exec,tox -e type-checking)

.PHONY: docs
docs:  ##@Development build documentation
	$(call _docker_exec,tox -e docs)

.PHONY: build
build:  ##@Development build and validate distribution packages
	$(call _docker_exec,tox -e build)

.PHONY: check
check: lint typecheck test  ##@Development run all checks (lint, typecheck, test)

.PHONY: all
all:  ##@Development run all tox environments
	$(call _docker_exec,tox)

.PHONY: clean
clean:  ##@Utils clean the project
	@echo "Cleaning Python artifacts..."
	@find . -name '*.pyc' -delete
	@find . -name '*.pyo' -delete
	@find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaning build artifacts..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf selfclean.egg-info/
	@rm -rf selfclean_audio.egg-info/
	@echo "Cleaning test and coverage artifacts..."
	@rm -rf .pytest_cache/
	@rm -rf .coverage*
	@rm -rf coverage.xml
	@rm -rf report.xml
	@rm -rf cov.xml
	@rm -rf htmlcov/
	@rm -rf cov_html/
	@echo "Cleaning development artifacts..."
	@rm -rf .tox/
	@rm -rf .mypy_cache/
	@rm -rf .ruff_cache/
	@rm -rf .cache/
	@rm -rf node_modules/
	@echo "Cleaning temporary and system files..."
	@rm -rf tmp/
	@rm -rf temp/
	@rm -f .DS_Store
	@find . -name '.DS_Store' -delete
	@echo "Cleaning IDE artifacts..."
	@rm -rf .idea/
	@rm -rf .vscode/
	@rm -rf *.swp
	@rm -rf *.swo
	@echo "Clean complete!"

###########################
# DOCKER COMPOSE
###########################
.PHONY: up
up:  ##@Docker compose up (build + start app and db)
	docker compose up -d --build

.PHONY: down
down: ##@Docker compose down (stop and remove)
	docker compose down

.PHONY: logs
logs: ##@Docker tail app logs
	docker compose logs -f app

.PHONY: bash
bash: ##@Docker open bash in app container
	docker compose exec app bash

.PHONY: jupyter
jupyter: ##@Docker start jupyter notebook in app container
	@echo "Starting Jupyter notebook on http://localhost:8888"
	docker compose exec app jupyter notebook --allow-root --ip 0.0.0.0 --port 8888 --no-browser

.PHONY: check_env
check_env: ##@Docker run environment checks inside app container
	docker compose exec app python scripts/check_runtime_env.py
