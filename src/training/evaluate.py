import json
from pathlib import Path

import pandas as pd
import yaml
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


if __name__ == "__main__":
    with open("params.yaml", "r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze("columns")
    pipeline = load("models/model.joblib")

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    threshold = float(configuration["decision_threshold"])
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "threshold": threshold,
        "test_samples": int(len(y_test)),
        "fraud_prevalence": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    metrics_path = Path("metrics.json")
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2, sort_keys=True))
