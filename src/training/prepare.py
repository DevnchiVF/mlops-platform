from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    df = pd.read_csv("data/raw/transactions.csv")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df.drop(columns=["timestamp", "transaction_id"], inplace=True)

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    with open("params.yaml", "r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=configuration["test_size"],
        random_state=configuration["random_seed"],
        stratify=y,
    )

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    print(
        f"Prepared train={len(X_train)} and test={len(X_test)} rows. "
        f"Train fraud rate={y_train.mean():.2%}, "
        f"test fraud rate={y_test.mean():.2%}."
    )
