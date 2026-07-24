# Behavioral Lending Intelligence Platform
## Production Build Plan — 4-Engineer Team, Cursor-Based Workflow

---

## 1. Team Structure & Ownership

With 4 engineers, avoid a "everyone touches everything" free-for-all — Cursor makes individual velocity high, but a shared codebase without clear ownership boundaries turns into merge-conflict hell fast. Assign **primary ownership**, not exclusive access — anyone can read/PR anywhere, but one person is accountable per domain.

| Role | Owns | Secondary responsibility |
|---|---|---|
| **Engineer 1 — Data/Platform Lead** | Ingestion pipelines, feature store, data contracts, infra-as-code | CI/CD |
| **Engineer 2 — ML Engineer** | Capacity model, intent model, SHAP explainability, model registry | Evaluation harness |
| **Engineer 3 — Backend/API Engineer** | Scoring API, orchestration service, auth, integration with core banking mocks | Security hardening |
| **Engineer 4 — Frontend/Product Engineer** | RM dashboard, admin console, product trigger config UI | QA/demo data |

One person (rotate weekly, or fix as Engineer 1) owns **release management** — merging to `main`, tagging versions, keeping the environment matrix (dev/staging) sane.

---

## 2. Cursor Workflow Setup (Day 0)

Before writing product code, set up the repo so Cursor is actually useful across 4 people instead of 4 people getting 4 different styles of AI-generated code.

1. **Monorepo structure** (pnpm/turborepo or a simple `/apps` + `/packages` split) — one repo, not four:
   ```
   /apps
     /api            → backend scoring & orchestration service
     /dashboard      → RM/admin frontend
     /pipelines      → data ingestion + feature engineering jobs
     /ml             → training, evaluation, model artifacts
   /packages
     /schemas        → shared data contracts (Pydantic/Zod, kept in sync)
     /shared-utils
   /infra            → IaC (Terraform), Docker, CI configs
   /docs
   ```
2. **`.cursor/rules`** committed to the repo — this is the highest-leverage thing you can do on day 0. Write explicit rules for:
   - Coding conventions (naming, error handling, logging format)
   - "Never hardcode secrets, always read from env/secrets manager"
   - "All model-facing code must produce SHAP-compatible output — no black-box additions"
   - "All new features written to the feature store must be documented in `/docs/feature-catalog.md`"
   - Preferred libraries per domain (pandas/polars for data, LightGBM for models, FastAPI for backend, React+Tailwind for frontend) so Cursor doesn't improvise inconsistent stacks across engineers
3. **Shared context files** for Cursor to reference: put your architecture diagram, data dictionary, and compliance constraints (no deep sequence models, no auto-rejection, consent-gating rules) as markdown in `/docs` and reference them in `.cursor/rules` so every AI-assisted change respects them by default.
4. **Branch + PR convention**: `feature/<domain>-<short-desc>`, mandatory PR review from the domain owner, Cursor-generated code reviewed like any other — no "AI wrote it so it's fine" exceptions, especially in the ML and API layers.

---

## 3. Phased Delivery Plan

Total horizon: **~16 weeks** to a pilot-ready product (matches the earlier roadmap's "Prototype → Pilot" stretch, compressed into a real build). Each phase has a working, demoable artifact at the end — never a "big bang" integration at week 14.

### Phase 0 — Foundations (Week 1)
**Goal:** Repo, environments, and data contracts exist before any model or UI code.

| Task | Owner | Deliverable |
|---|---|---|
| Repo scaffold, Cursor rules, CI skeleton | Eng 1 | Monorepo builds & lints on push |
| Define data contracts (customer, transaction, feature, score schemas) | Eng 1 + Eng 2 | `/packages/schemas` with versioned Pydantic/Zod models |
| Synthetic data generator (personas, 12mo transaction history) | Eng 2 | `pipelines/synth-data` producing `transactions.csv`, `customers.csv`, `labels.csv` |
| API skeleton (FastAPI, health check, auth stub) | Eng 3 | Deployed to dev, returns 200 |
| Dashboard skeleton (routing, auth stub, design tokens) | Eng 4 | Empty shell deployed to dev |

**Exit criteria:** synthetic data exists, all four apps deploy to a dev environment, schemas are agreed and frozen for v1.

---

### Phase 1 — Feature Store (Weeks 2–4)
**Goal:** Raw transactions → validated, versioned behavioral features. This is the foundation everything else depends on — don't let it slip.

- **Feature groups to implement** (from earlier design): income stability, cash flow, debt behavior, digital engagement, life-cycle/anomaly signals (subscription cleansing, liquidity pooling, out-of-cycle bill shifts).
- **Pipeline design**: batch job (Airflow, Dagster, or even a scheduled script for v1 — don't over-engineer orchestration before you have real load) computing features nightly, writing to a feature table keyed by `customer_id, feature_name, value, as_of_date`.
- **Anomaly detection**: Isolation Forest per-customer baseline for subscription cleansing / bill-shift detection — this is Eng 2's first modeling task, but it's feature engineering, not the core model, so build it early.
- **Feature catalog**: every feature documented with definition, computation logic, and owner in `/docs/feature-catalog.md` — this is what makes the store "reusable" and auditable, not just a pile of columns.

**Exit criteria:** feature store populated from synthetic data; a notebook/script can pull a customer's full feature vector; feature catalog covers 100% of shipped features.

---

### Phase 2 — Capacity & Intent Models (Weeks 4–7, overlapping Phase 1's tail)
**Goal:** Two explainable models producing scores from the feature store.

- **Capacity model**: LightGBM regressor → disposable income estimate + confidence band (based on data completeness/variance). Output contract: `{customer_id, estimated_income_low, estimated_income_high, confidence, dti}`.
- **Intent model**: LightGBM classifier per product (start with one product — Home Loan — end to end, then replicate) → propensity score 0–100.
- **Gating logic**: hard eligibility filter (DTI threshold, delinquency flags) applied before intent ranking — implement this as an explicit, reviewable rules module, not buried inside the model, so Credit Risk can audit it independently of the ML.
- **Explainability**: SHAP values computed at scoring time, stored alongside each score — not generated on-demand only for demos.
- **Evaluation harness**: precision@K, conversion lift vs. baseline, capacity MAE against synthetic ground truth — build this once, reuse for every model iteration and eventually for live monitoring.

**Exit criteria:** given a customer_id, a script produces a capacity estimate, an intent score, a gating decision, and SHAP reason codes — reproducibly, versioned as a model artifact.

---

### Phase 3 — Scoring API & Orchestration (Weeks 6–9, overlapping Phase 2)
**Goal:** Turn the model pipeline into a service the dashboard (and eventually CRM) can call.

- **Endpoints**:
  - `POST /score/{customer_id}` → triggers/reads latest score + reasons
  - `GET /leads?product=home_loan&min_score=75` → ranked lead list
  - `GET /customer/{id}/features` → feature vector (internal/debug use)
- **Model serving**: load model artifacts from a registry (start simple — versioned files in blob storage + a manifest; MLflow if the team has bandwidth) rather than embedding models directly in app code.
- **Auth & access control**: RM-role vs admin-role scoping from day one — even in a prototype, don't let this be an afterthought, since it's the first thing a bank security review will ask about.
- **Audit logging**: every score served is logged with model version, inputs hash, and timestamp — this is your compliance audit trail, not optional infra polish.

**Exit criteria:** dashboard can call real endpoints (not mocked data) and get back a ranked, explainable lead list.

---

### Phase 4 — RM Dashboard & Admin Console (Weeks 8–11, overlapping Phase 3)
**Goal:** The thing judges/stakeholders actually see and RMs actually use.

- **RM view**: ranked lead list, customer detail card (score, capacity band, reason codes, recommended product) — matches the mockup already validated in the pitch deck.
- **Admin/config view**: eligibility gate thresholds, product trigger definitions, feature catalog browser — lets Credit Risk/Product adjust rules without a code deploy, which matters a lot for adoption.
- **Churn/deposit-leakage bonus view** (stretch, from the feature-store-reuse discussion): same features, inverted signals (salary migration, FD closures) → a second dashboard tab proving the store's reusability claim.

**Exit criteria:** a non-engineer can open the dashboard, see a ranked lead list with reasons, and adjust a threshold in the admin view.

---

### Phase 5 — Compliance, Security & Responsible-AI Hardening (Weeks 10–13, overlapping)
**Goal:** The guardrails from the pitch aren't slideware — they're implemented.

| Guardrail | Implementation |
|---|---|
| Consent-first external data | Consent-flag check gating any Account Aggregator/bureau call; no external pull without an active consent record |
| No auto-rejection | Gating module only ever *filters/ranks*; no code path exists that issues a rejection without human sign-off |
| Explainable only | CI check that fails the build if a non-tree-based model is registered for scoring |
| Bias audit | Fairness metrics (disparate impact ratio across proxy groups) computed in the eval harness, reviewed before each model promotion |
| Confidence-banded output | API contract enforces `confidence` field is never null; dashboard visually flags low-confidence cases for manual review |
| PII handling | Encryption at rest for raw transaction data; feature store stores only derived features, not raw PII, wherever possible |

**Exit criteria:** a written responsible-AI checklist, signed off by the team, mapped 1:1 to implemented controls — this is what you hand to a compliance reviewer, real or judge-simulated.

---

### Phase 6 — Testing, Load, and Hardening (Weeks 12–14)
- Unit tests on feature computation logic (deterministic, easy to test — do not skip these)
- Integration tests: synthetic customer → full pipeline → expected score range
- Model regression tests: pinned synthetic personas with expected score bands, run on every model retrain to catch silent drift
- Basic load test on the scoring API (even a simple k6/Locust script) to catch obvious bottlenecks before pilot

### Phase 7 — Pilot Packaging (Weeks 14–16)
- Deployment runbook, environment configs for a real (or realistically anonymized) pilot dataset
- Monitoring dashboard: score distribution drift, feature freshness, model performance decay triggers
- Stakeholder demo script + fallback plan if a live dependency fails during demo

---

## 4. Tech Stack Summary

| Layer | Choice | Why |
|---|---|---|
| Data pipeline | Python (pandas/polars) + a lightweight scheduler | Team already knows Python; avoid Spark/Kafka overhead at this scale |
| Models | LightGBM + SHAP | Explainable, fast to train/retrain, defensible to Credit Risk |
| Feature store | Postgres (or DuckDB for local dev) with a documented schema | Simple, queryable, no need for a dedicated feature-store product at pilot scale |
| Backend API | FastAPI (Python) | Same language as ML, fast to build, auto-generated OpenAPI docs |
| Frontend | React + Tailwind | Matches earlier dashboard mockups, fast iteration in Cursor |
| Infra | Docker Compose (dev) → Terraform + a single cloud provider (pilot) | Don't over-invest in multi-cloud/k8s before there's real load |
| CI/CD | GitHub Actions | Lint, test, build on every PR; deploy on merge to `main` |
| Model registry | Versioned artifacts in blob storage + manifest JSON | Lightweight; upgrade to MLflow only if the team outgrows this |

---

## 5. Risk Register (review weekly)

| Risk | Mitigation |
|---|---|
| Synthetic data doesn't generalize to real transaction patterns | Validate feature distributions against publicly available anonymized benchmarks; flag as an explicit pilot-phase validation task, not a prototype blocker |
| Four engineers, four different Cursor-generated coding styles | `.cursor/rules` + mandatory domain-owner PR review, enforced from Phase 0 |
| Model complexity creeping toward non-explainable methods under deadline pressure | CI gate (Phase 5) blocking non-tree-based models from the registry |
| Compliance treated as a Week 15 afterthought | Guardrails phase (5) runs in parallel from Week 10, not bolted on at the end |
| Feature store becomes a dumping ground with no documentation | Feature catalog is a hard exit criterion for Phase 1, reviewed before Phase 2 starts |

---

## 6. Definition of Done (End-to-End Demo)

By Week 16, one live walkthrough should show:
1. A synthetic customer's raw transactions flowing into the feature store
2. A capacity estimate with confidence band and an intent score with SHAP reasons
3. That customer appearing, ranked, in the RM dashboard with a recommended product
4. An admin adjusting a gating threshold and seeing the lead list change live
5. An audit log entry proving the score is traceable to a model version and input snapshot
6. A one-page responsible-AI checklist mapped to what's actually running

That's the artifact that proves this is a product plan, not a deck.
