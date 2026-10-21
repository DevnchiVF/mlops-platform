import pandas as pd
from joblib import load
from sklearn.metrics import classification_report

if __name__ == "__main__":
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")
    pipeline = load("models/model.joblib")

    y_probs = pipeline.predict_proba(X_test)[:, 1]
    umbral = 0.30
    y_pred_custom = (y_probs >= umbral).astype(int)
    print(classification_report(y_test, y_pred_custom))