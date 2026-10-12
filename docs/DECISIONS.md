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

## Phase C decisions

- 2026-08-18 Decision: DTI computed from EMI-only recurring debits, not all debits.
  Rationale: DTI measures recurring debt obligations / income. Including one-time payments
  (e.g., builder payments for home buyers) inflated DTI to implausible values (3.65x for
  cust_008's life_event_home_buyer persona). The behavioral_features.py code already identifies
  EMI-like recurring series (CV < 0.10, ≥ 3 occurrences) — the fix uses only those for the
  DTI numerator. One-time large outflows affect liquidity_buffer and savings_ratio instead,
  which is the right signal to capture them with.

- 2026-08-18 Decision: Capacity model uses min_data_in_leaf=3 and num_leaves=8.
  Rationale: with ~36–80 synthetic training customers, LightGBM's default min_data_in_leaf=20
  causes the model to collapse to a single-leaf stump (constant prediction, zero SHAP values
  for every feature). The fix lowers the threshold so the model can actually split. These
  hyperparameters MUST be revisited when training on real production-scale data — they are
  intentionally permissive for the current synthetic-data scale.

- 2026-08-18 Decision: salary_amount_median and avg_monthly_outflow excluded from capacity
  training features.
  Rationale: the synthetic capacity target is `salary_amount_median - avg_monthly_outflow`.
  Including these features would allow trivial target leakage (model learns y = f1 - f2).
  These features are explicitly listed in CAPACITY_EXCLUDED_FEATURES and checked in the
  training pipeline. This exclusion should be reviewed if/when the target changes to a
  real underwriting-derived disposable income measure.

- 2026-08-18 Decision: Quantile regression (q=0.1, 0.5, 0.9) for capacity uncertainty.
  Rationale: a single point estimate of disposable income is insufficient for risk-aware
  underwriting. The q10/q90 interval provides a "pessimistic/optimistic" range, and the
  interval width drives the confidence score (tighter interval → higher confidence). This
  also means the SHAP explainability works on a real model, not a stump.

- 2026-08-18 Decision: Isotonic calibration for intent probabilities.
  Rationale: raw LightGBM probabilities are poorly calibrated at small scale. Isotonic
  calibration (with cv=3) converts the ranking into an approximately true probability.
  The base (uncalibrated) model is saved separately for SHAP TreeExplainer, since SHAP
  needs the raw LightGBM booster, not the scikit-learn calibration wrapper.

- 2026-08-18 Decision: Eligibility gating is a separate YAML-configured rules module.
  Rationale: per AGENTS.md, underwriting decision logic must never be buried inside model
  internals. Each gating rule (DTI, vintage, confidence) is a separate named function
  in eligibility_gate.py, and thresholds are loaded from gating_thresholds.yaml — a
  non-ML reviewer can inspect and audit the complete decision boundary without reading
  any model code.

- 2026-08-18 Friction point: Phase B schema updates (customer.py, transaction.py, label.py)
  were never committed — the ingest pipeline imported fields that didn't exist in the
  Pydantic schemas. Fixed alongside Phase C by adding Optional fields with defaults.

## Phase D decisions

- 2026-08-31 Decision: Separate `config_change_log` table for threshold changes, not overloading
  `audit_log`. Rationale: score audits track customer_id, model versions, and feature hashes;
  config changes track field name, old/new values, and the admin who changed them. Different
  schemas, different query patterns — merging them would require nullable columns and awkward
  queries for both use cases.

- 2026-08-31 Confirmed: `eligibility_gate.py` re-reads `gating_thresholds.yaml` on every
  `check_eligibility()` call via `_load_thresholds()`. No module-level caching, no fix needed.
  The admin config endpoint (PUT /config/gating-thresholds) writes to the same YAML file, and
  the gate picks up changes immediately without restart.

- 2026-08-31 Decision: `gate_result` column in `audit_log` is `TEXT`, not `BOOLEAN`.
  Rationale: the eligibility gate can produce three-valued outcomes ("pass", "fail",
  "manual_review"). A boolean column would silently collapse manual_review into one of the
  other two, destroying exactly the signal a compliance reviewer most needs to see.

- 2026-08-31 Decision: `python-jose[cryptography]` for JWT encoding/decoding (HS256).
  Future consideration: `PyJWT` is more actively maintained and has fewer historical CVEs —
  worth evaluating before this reaches real customer data, but not blocking for Phase D.

- 2026-08-31 Decision: Dev-only token endpoint (POST /auth/dev-token) is gated by
  `ENVIRONMENT=local` env var. If ENVIRONMENT is anything else or unset, the router is never
  created — the route cannot be called at all. This prevents accidental exposure in non-local
  deployments.

- 2026-08-31 Decision: Field names in the config endpoint match the actual YAML and
  eligibility_gate.py code: `max_dti`, `min_vintage_months`, `min_confidence`. The original
  Phase D spec referenced different names (`dti_threshold`, `min_account_vintage_months`,
  `min_capacity_confidence`) that don't exist in the codebase — those were not used.

- 2026-08-31 Known break: Dashboard's `/score/` call has no auth header yet — it will get 401
  against the hardened API. Expected until Phase E, which adds auth to the dashboard.

- 2026-08-31 BREAKING CHANGE: `GET /score/{customer_id}` now requires a `Bearer` JWT token
  with at least "rm" role. Callers without a token get 401. This was unauthenticated in Phase C.

- 2026-09-08 Investigation: cust_001 failing gate during Phase D verification. Initial diagnosis
  blamed small-sample training instability — **incorrect**. Actual root cause: Phase D verification
  ran `make down-clean && make up && make ingest && make train`, but `make ingest` calls
  `generate_minimal` (1 customer) while `make pipeline` calls `generate` (default 8 personas).
  Neither regenerates the full 80-persona dataset that Phase C used. `make down-clean` wipes
  Postgres volumes but NOT the host CSV files, so whichever stale CSV was on disk got re-ingested.
  
  Retesting with `--personas 80`:
  - Intent confidence: 0.0 → 0.4798 (cust_001) — data volume was the problem, not the model
  - Capacity confidence: still 0.0 for cust_001 (q10=-3610, relative_width=1.13), but cust_003
    gets 0.7789 and cust_005 gets 0.6328 — model CAN produce confidence, cust_001 just has
    legitimate per-customer uncertainty (negative q10 prediction)
  - Gate: cust_003 and cust_005 now pass (zero passed with 8 personas)
  
  Lesson: `make ingest` is a footgun for verification — it doesn't regenerate data, so any
  `make down-clean` followed by `make ingest` will silently train on whatever CSV was left from
  the last run. Consider adding a guard or deprecating `make ingest` in favor of `make pipeline`.

- 2026-09-08 Bug fix: `np.float64` values in `run_nightly_features.py` were written as literal
  `np.float64(...)` strings into SQL via psycopg2. Fixed with `value.item()` conversion before
  insert. This is a Phase B/C pipeline bug, not Phase D — should be committed separately from
  the auth/audit/config changes.

## Phase E decisions

- 2026-09-09 Decision: Added `GET /customers` and `GET /leads` endpoints in `apps/api/leads/leads_router.py`.
  Rationale: The dashboard needs to enumerate leads ranked by propensity and filtered by eligibility.
  At prototype scale, `/leads` evaluates `predict()` in a loop. For pilot scale, this should transition
  to reading from a precomputed batch ranking table.

- 2026-09-09 Decision: Lightweight batch audit entry for `/leads`.
  Rationale: Individual customer scores already log full feature hashes and model versions on `/score/{id}`.
  Logging a concise batch entry on `/leads` records who requested the ranked list and what filter parameters
  were applied without flooding the audit table on dashboard refreshes.

- 2026-09-09 Decision: Three-valued `gate_result` (`pass`, `fail`, `manual_review`) in `EligibilityResult`.
  Rationale: Low model confidence or short tenure should never result in an automated adverse action (rejection).
  Mapping these conditions to `manual_review` ensures high-risk uncertainty triggers human underwriter sign-off
  while hard financial limits (e.g. DTI > max_dti) produce `fail`.

- 2026-09-09 Decision: Client-side routing with `react-router-dom` and `ProtectedRoute`.
  Rationale: Allows distinct RM workspace views (`/leads`, `/customer/:id`) and Admin views (`/admin/config`,
  `/admin/audit`) backed by JWT role verification in `AuthContext` and automatic 401 handling in `apiFetch`.

- 2026-09-09 Decision: Prominent visual warning for model confidence < 0.30.
  Rationale: Aligns with pitch deck and Responsible AI commitments — low-confidence or high-variance estimates
  must never visually blend in with reliable ones.

