.PHONY: up down logs ingest test test-pipelines test-ml test-api test-dashboard health lint

## Bring up the full local environment
up:
	docker compose up -d --build
	@echo "Postgres:   localhost:5432"
	@echo "API:        http://localhost:8000"
	@echo "Dashboard:  http://localhost:5173"

## Tear everything down (keeps volumes/data)
down:
	docker compose down

## Tear everything down AND wipe local data volumes
down-clean:
	docker compose down -v

## Tail logs across all services
logs:
	docker compose logs -f

## Run the ingestion + nightly feature batch job on demand
ingest:
	docker compose exec pipelines python -m pipelines.ingest.run \
		--customers /app/data/customers.csv \
		--transactions /app/data/transactions.csv
	docker compose exec pipelines python -m pipelines.feature_store.run_nightly_features

## Run full test suite across all apps
test: test-pipelines test-ml test-api test-dashboard

test-pipelines:
	docker compose exec pipelines pytest -q

test-ml:
	docker compose exec ml pytest -q

test-api:
	docker compose exec api pytest -q

test-dashboard:
	docker compose exec dashboard npm test -- --run

## Lint across Python apps
lint:
	docker compose exec pipelines ruff check .
	docker compose exec ml ruff check .
	docker compose exec api ruff check .

## Basic health check — confirms every service is reachable
health:
	@echo "Checking postgres..."
	@docker compose exec postgres pg_isready -U $${POSTGRES_USER:-cashflowiq} || (echo "postgres DOWN" && exit 1)
	@echo "Checking api..."
	@curl -sf http://localhost:8000/health > /dev/null && echo "api OK" || (echo "api DOWN" && exit 1)
	@echo "Checking dashboard..."
	@curl -sf http://localhost:5173 > /dev/null && echo "dashboard OK" || (echo "dashboard DOWN" && exit 1)
	@echo "All services healthy."
