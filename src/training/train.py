from joblib import dump
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        configuration = yaml.safe_load(f)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ["customer_id", "country", "merchant"]),
            ('num', 'passthrough', ["amount", "hour"])
        ],
        remainder='drop'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(class_weight=configuration["class_weight"], 
                                          random_state=configuration["random_seed"]))
    ])

    X_test = pd.read_csv("data/processed/X_test.csv")
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")

    pipeline.fit(X_train, y_train.values.ravel())

    dump(pipeline, "models/model.joblib")
