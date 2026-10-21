from pathlib import Path

import pandas as pd
import yaml
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


if __name__ == "__main__":
    with open("params.yaml", "r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    categorical_features = ["customer_id", "country", "merchant"]
    numeric_features = ["amount", "hour", "day_of_week", "is_weekend"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=configuration["n_estimators"],
                    max_depth=configuration["max_depth"],
                    min_samples_leaf=configuration["min_samples_leaf"],
                    class_weight=configuration["class_weight"],
                    random_state=configuration["random_seed"],
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze("columns")

    pipeline.fit(X_train, y_train)

    model_path = Path("models/model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(pipeline, model_path)

    print(
        f"Model trained on {len(X_train)} rows and saved to {model_path}."
    )
