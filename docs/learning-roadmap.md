# Learning Roadmap

This project is intentionally split into an **apply-ready MVP** and optional
extensions. You do not need to master every tool before applying for junior
roles.

## Rule for learning this project

For every component, be able to answer three questions before adding the next
one:

1. What problem does this component solve?
2. What would happen if I removed it?
3. How can I verify that it is working?

Do not memorize commands without being able to answer those questions.

## Phase 0 — Foundations first

Before adding more MLOps tooling, understand enough Python/ML to explain the
existing pipeline.

Learn:

- Python virtual environments, packages and imports.
- pandas DataFrames: read/write CSV, columns, filtering and basic transforms.
- train/test split and why data leakage is bad.
- classification: probability, threshold, confusion matrix.
- precision, recall, F1, ROC-AUC and especially PR-AUC for imbalanced data.
- scikit-learn Pipeline and ColumnTransformer.
- Git basics: commit, branch, merge/pull request.

You do **not** need advanced mathematics or to implement algorithms from
scratch.

Exit test: explain the path from one raw transaction to a fraud prediction and
why accuracy alone is misleading.

## Phase 1 — Reproducible training

Project pieces:

- Pandera validation.
- DVC stages and params.
- deterministic random seed.
- sklearn preprocessing + model pipeline.
- metrics.json.

Study in depth:

- DVC pipeline concepts: deps, outs, params, metrics and `dvc repro`.
- sklearn Pipeline/ColumnTransformer.
- imbalanced classification metrics.

Study only briefly:

- Pandera syntax. Understand schema validation and why failing fast is useful;
  there is no need to master the entire library.

Exit test: change one parameter, run `dvc repro`, and explain exactly which
stages should rerun and why.

## Phase 2 — Experiment tracking

Add MLflow Tracking first. Registry comes after you understand runs.

Study in depth:

- experiment vs run.
- parameters, metrics and artifacts.
- model artifact vs model metadata.
- tracking server and backend/artifact stores at a conceptual level.

Then add:

- Model Registry.
- model versions.
- champion/challenger aliases.

Do not study every MLflow feature.

Exit test: compare two model runs in MLflow and justify which one you would
promote.

## Phase 3 — Serving and tests

Add FastAPI around the trained/registered model.

Study in depth:

- HTTP basics: request, response, status code, JSON.
- GET vs POST.
- Pydantic request/response schemas.
- FastAPI dependency/startup concepts only as needed.
- unit tests vs integration tests with pytest.

Endpoints for the MVP:

- `POST /predict`
- `GET /health`
- `GET /ready`

Add `/feedback` later when monitoring model performance.

Exit test: send a transaction with curl/Postman, get a prediction, and explain
how the same preprocessing used in training is reused at inference.

## Phase 4 — Docker, CI and a small database

This is the **apply-ready MVP**.

Study Docker in depth enough to understand:

- image vs container.
- Dockerfile.
- build, run, ports, volumes and environment variables.
- Docker Compose and service-to-service networking.

Study CI/CD in depth enough to understand:

- what CI is.
- what a GitHub Actions workflow/job/step is.
- why lint/tests/build run on a pull request.
- why a failing test should block a merge.

Learn practical SQL:

- CREATE TABLE.
- SELECT, INSERT, UPDATE.
- WHERE, ORDER BY, GROUP BY.
- basic JOIN.
- primary/foreign keys and indexes conceptually.

MVP services:

- API.
- PostgreSQL.
- MLflow.
- trainer job.

At this point, improve the README, record a short demo and **start applying**.

## Phase 5 — Observability and model monitoring

Only after the MVP works.

Learn the concepts before the tools:

- application monitoring vs model monitoring.
- latency, throughput and error rate.
- data drift vs concept drift.
- why labels can arrive after predictions.
- model performance with delayed labels.

Then learn only the necessary Prometheus/Grafana/Evidently pieces.

Prometheus:

- metric types at a high level.
- scrape endpoint.
- counters and histograms.

Grafana:

- datasource.
- dashboard/panel.
- a few useful queries; do not try to master Grafana.

Evidently:

- reference dataset vs current dataset.
- drift report/test.
- how you decide whether drift matters.

Exit test: inject a known distribution shift and explain what changed, what did
not, and why drift does not automatically mean model quality got worse.

## Phase 6 — Retraining and promotion

This is the strongest MLOps-specific extension.

Implement:

- delayed labels through `/feedback`.
- challenger training.
- evaluation against the current champion.
- explicit promotion rule.
- rejection when the challenger is worse.
- reload/rollback path.

Study:

- model lifecycle.
- champion/challenger pattern.
- rollback.
- automated retraining risks.
- why a model should not be promoted only because drift was detected.

Exit test: demonstrate both a successful promotion and a rejected challenger.

## Phase 7 — Kubernetes

Kubernetes is useful, but it is **not required before applying**.

Learn a basic tutorial first. Focus on:

- cluster, node and pod.
- Deployment.
- Service.
- ConfigMap and Secret.
- readiness/liveness probes.
- resource requests/limits.
- rolling update and rollback.

Use kind locally.

Do not study operators, CRDs, service meshes, KServe or advanced networking for
this project.

Exit test: deploy the API, scale replicas, perform a rolling update and roll it
back.

## Terraform — optional

Do this only after the previous phases make sense.

Learn:

- provider.
- resource.
- variable/output.
- state.
- plan/apply/destroy.

One small cloud resource is enough. Terraform is here to demonstrate IaC, not
to turn this project into a cloud certification.

## When to start applying

Start applying once Phase 4 is solid.

A good junior explanation is more valuable than a larger project you cannot
explain. The later phases improve the project while you are already applying.

You should be able to explain:

- why the data pipeline is reproducible.
- why fraud metrics differ from accuracy.
- how training preprocessing is reused in serving.
- how experiments/models are versioned.
- how the API is packaged and tested.
- what CI prevents.
- where predictions are stored.

Then continue adding monitoring, retraining and Kubernetes while interviewing.
