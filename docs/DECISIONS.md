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

## Phase B retrospective

- 2026-07-28 Decision: All new schema fields added as Optional with defaults to avoid breaking
  existing Phase A code paths. This is a schema-widening change, not a breaking one.

- 2026-07-28 Decision: Features table is DROP+recreated (not migrated) when transitioning from
  Phase A 3-column shape to Phase B long-format. Rationale: Phase A features are synthetic dev
  data with no preservation value, and the composite PK change (adding `computed_at`) makes
  in-place migration impractical for zero benefit.

- 2026-07-28 Decision: Transaction IDs are hash-based (SHA-256 of customer_id + date + amount +
  category + type + merchant_category). Known limitation: will collide on true duplicate
  transactions (same customer, same day, same amount, same category). Acceptable for synthetic
  data; needs revisiting before real data ingestion.

- 2026-07-28 Decision: Anomaly detector thresholds (60-day window for subscription cleansing,
  45-day reinvestment window for liquidity pooling, IsolationForest contamination=0.3 for bill
  shift) are based on domain heuristics, not empirically tuned. Phase C model training will
  validate these against actual conversion correlation.

- 2026-07-28 Decision: FEATURE_FUNCTIONS registry pattern chosen for feature orchestration —
  adding a new feature function is an addition to the registry dict, not a rewrite of the
  orchestration logic. This keeps run_nightly_features.py stable as feature count grows.

- 2026-07-28 Decision: Added `make pipeline` target chaining generate → ingest → features,
  addressing the Phase A friction point about manual sequential pipeline steps.

- 2026-07-28 Friction point: The pipelines container needs pandas, numpy, scikit-learn added
  to requirements.txt — Docker image rebuild required after Phase B changes.
