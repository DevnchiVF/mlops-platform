from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_FEATURES = ["customer_id", "country", "merchant"]
NUMERIC_FEATURES = ["amount", "hour", "day_of_week", "is_weekend"]


def build_pipeline(params: dict, random_seed: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        min_samples_split=params["min_samples_split"],
        max_features=params["max_features"],
        class_weight=params["class_weight"],
        random_state=random_seed,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, random_seed: int) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, None]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        pipeline = build_pipeline(params, random_seed)
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = pipeline.predict_proba(X.iloc[val_idx])[:, 1]
        scores.append(average_precision_score(y.iloc[val_idx], proba))

        trial.report(float(np.mean(scores)), step=len(scores))
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


if __name__ == "__main__":
    with open("params.yaml", "r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze("columns")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=configuration["random_seed"]),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, configuration["random_seed"]),
        n_trials=50,
        show_progress_bar=True,
    )

    print("Best PR-AUC (average precision):", study.best_value)
    print("Best parameters:", study.best_params)

    best_params_path = Path("models/best_params.yaml")
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_params_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(study.best_params, file)

    final_pipeline = build_pipeline(study.best_params, configuration["random_seed"])
    final_pipeline.fit(X_train, y_train)

    model_path = Path("models/model_optuna.joblib")
    dump(final_pipeline, model_path)
    print(f"Final model saved in {model_path}")
