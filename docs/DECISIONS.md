# Cashflow IQ — Technical Decisions Log

## Phase A retrospective

- 2026-07-27 Decision: Mount full project root (`.:/workspace`) into pipelines/ml/api containers
  instead of per-app mounts. Rationale: allows cross-package imports (`from packages.schemas
  import ...`, `from apps.ml.models import ...`) without PYTHONPATH hacks. Trade-off: container
  sees entire repo, which is fine for dev but shouldn't carry to production images.

- 2026-07-27 Decision: All pipeline scripts run via `docker compose exec` against already-running
  containers, not via a host Python venv. Rationale: single DATABASE_URL (using `postgres`
  hostname), single Python version (3.11 in container vs 3.9.6 on host), and the full-loop
  test in A8 exercises the exact same path used during development — no surprises.

- 2026-07-27 Decision: Used upsert (ON CONFLICT) for the `features` table even in Phase A.
  Rationale: the features table has a natural composite primary key (customer_id, feature_name),
  and upsert costs nothing extra while making re-runs safe. The `raw_transactions` table uses
  plain INSERT (no upsert) per Phase A rules — that will need idempotency in Phase B.

- 2026-07-27 Friction point: `docker-compose.yml` still has `version: "3.9"` which produces a
  deprecation warning on every command. Should remove in Phase B.

- 2026-07-27 Friction point: Running pipeline steps (generate → ingest → features) is manual
  and sequential. Phase B should add a `make pipeline` target that chains them.

- 2026-07-27 Friction point: The synth data generator writes CSV to `apps/pipelines/synth_data/output/`
  which is inside the bind-mounted workspace — files persist on host across container restarts,
  which is convenient for dev but means `make down-clean` doesn't wipe them. Consider whether
  generated data should live in a Docker volume instead.
