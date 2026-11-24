.PHONY: test
test:
	@echo "️🧪 Running Tests with coverage"
	coverage run -m pytest -v -p no:warnings . && coverage report --rcfile=.coveragerc

.PHONY: format
format:
	@echo "️🐶 Formatting with Ruff"
	ruff format . --line-length 120

.PHONY: format-check
format-check:
	@echo "️🐶 Checking Formatting with Ruff"
	ruff check . --line-length 120

.PHONY: build
build:
	@echo "️🐳 Building the Docker Image"
	docker build . -t nova -f infra/build/Dockerfile

.PHONY: build-light
build-light:
	@echo "️🪶 Building the Lightweight Docker Image"
	docker build . -t nova -f infra/build/Dockerfile.light

.PHONY: up
up: build
	docker compose --file infra/docker-compose.yaml down
	docker compose --file infra/docker-compose.yaml up app