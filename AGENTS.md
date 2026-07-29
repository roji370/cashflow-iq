# AGENTS.md — Cashflow IQ

Standing instructions for any AI coding agent (Antigravity, Cursor, Claude Code, or otherwise)
working in this repository. These rules apply everywhere in the repo unless a more specific
rules file (e.g. a tool-specific `GEMINI.md`, or a nested `AGENTS.md` in a subfolder) overrides
them for that scope.

## What this project is

Cashflow IQ is a behavioral lending intelligence system for a bank: it generates high-conversion
retail loan leads and estimates real repayment capacity from transaction data, for Personal,
Home, Mortgage, and Auto loans. This is a **regulated banking domain project**. Treat every
data-handling and modeling decision with that in mind, even in prototype/hackathon stages.

## Repository layout

```
/apps
  /api            → backend scoring & orchestration service
  /dashboard      → RM/admin frontend
  /pipelines      → data ingestion + feature engineering batch jobs
  /ml             → model training, evaluation, explainability, registry logic
/packages
  /schemas        → shared data contracts (Pydantic), used by every app — SOURCE OF TRUTH
  /shared-utils
/infra            → Docker, docker-compose, Terraform, CI configs
/docs             → architecture notes, feature-catalog.md, data dictionary, DECISIONS.md files
```

## Non-negotiable rules

### Data contracts
- Every field name and type used across apps must match `/packages/schemas` exactly.
- If a schema is missing a field you need, **flag it and ask** — do not invent a new field name
  or silently add one to a schema without calling it out in the PR description.
- Any change to a schema in `/packages/schemas` that removes a field, renames a field, or changes
  a field's type is a **breaking change** — it must be called out explicitly (e.g. a
  `BREAKING CHANGE:` line in the commit/PR description) since other apps depend on these contracts
  without necessarily reviewing schema diffs closely.

### Modeling
- Only tree-based models (LightGBM, XGBoost, RandomForest) may be used for anything producing a
  customer-facing or underwriting-facing score. Never use neural networks, RNNs, LSTMs, or other
  non-interpretable architectures for tabular scoring in this repo — explainability is a hard
  requirement, not a nice-to-have, given the underwriting use case.
- Every model that outputs a score must also output a confidence/uncertainty value and have a
  corresponding SHAP explainability path. No `predict()` without a paired `explain()`.
- Never allow a feature to leak target information (e.g. a feature computed using post-outcome
  data). If you're unsure whether a feature is safe, flag it rather than proceeding.
- All model artifacts must be versioned: filename or manifest must include `model_version`,
  `training_date`, and `feature_schema_version`.
- Any eligibility/underwriting decision logic must be a separate, explicit, human-auditable rules
  module — never buried inside a model's internals where a non-ML reviewer can't inspect it.
- No fully automated adverse action (rejection) — models rank and prioritize; a human always
  signs off before a customer is declined.

### Data pipelines
- All batch jobs must be idempotent — safe to re-run against the same input without duplicating
  or corrupting data. Prefer upsert semantics over blind insert.
- Never log or persist raw PII in error messages, feature names, or debug output.
- External data (Account Aggregator, credit bureau, or any third-party consented source) may only
  be fetched when an active, current consent record exists for that customer and purpose — no
  exceptions, even in prototype/demo code paths.

### Code quality
- Every function needs type hints and a docstring.
- Every new function should have at least one corresponding unit test in the same PR — don't
  defer tests to a "later" pass.
- No hardcoded secrets, credentials, or connection strings anywhere, including in scaffold or
  example code — always read from environment variables or a secrets manager.

### Working style
- Prefer small, single-concern changes over large multi-file rewrites in one pass. If a task
  spans schemas, pipelines, and models, break it into separate reviewable steps.
- When uncertain about a design decision (schema shape, threshold value, storage backend), ask
  rather than guessing — document the decision once made in the relevant `DECISIONS.md`.
- Log significant technical decisions in `/docs/DECISIONS.md` (or the app-specific one, e.g.
  `/apps/ml/DECISIONS.md`) as you go: date, decision, rationale. This becomes model-card and
  audit documentation later — don't leave it to be reconstructed at the end.

## Who owns what (for context, not access control)

| Area | Owner |
|---|---|
| `/apps/pipelines`, `/infra`, `/packages/schemas` (platform) | Engineer 1 |
| `/apps/ml` (models, features, explainability, registry) | Engineer 2 |
| `/apps/api` | Engineer 3 |
| `/apps/dashboard` | Engineer 4 |

Anyone can read or propose changes anywhere; the owner above should review before merge for
their domain.
