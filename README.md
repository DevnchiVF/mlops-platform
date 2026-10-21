# MLOps Fraud Detection Pipeline

A reproducible fraud-detection project using synthetic data, Pandera validation,
scikit-learn, DVC and a Random Forest baseline.

## Pipeline

The DVC pipeline runs five stages:

1. **generate** — creates synthetic transactions with learnable fraud patterns.
2. **validate** — validates the raw dataset schema with Pandera.
3. **prepare** — engineers time features and creates stratified train/test splits.
4. **train** — fits a preprocessing + Random Forest pipeline.
5. **evaluate** — writes fraud-focused metrics to `metrics.json`.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the complete pipeline

```bash
dvc repro
```

Inspect metrics with:

```bash
dvc metrics show
```

## Model inputs

The model uses:

- customer ID
- country
- merchant
- transaction amount
- transaction hour
- day of week
- weekend indicator

Categorical variables are encoded inside the scikit-learn pipeline, so the exact
same preprocessing is persisted with the model.

## Fraud generation

Fraud labels are no longer assigned independently at random. The generator
creates a noisy risk score based on combinations of high amounts, unusual hours,
selected countries/merchants and selected customers, then assigns fraud to the
highest-risk observations according to `fraud_rate`.

This makes the synthetic dataset useful for demonstrating model training while
still retaining randomness and class imbalance.

## Metrics

Because fraud is highly imbalanced, the primary metrics are:

- Precision
- Recall
- F1
- PR-AUC
- ROC-AUC

Accuracy is included only as contextual information. The classification threshold
is configurable through `decision_threshold` in `params.yaml`.
