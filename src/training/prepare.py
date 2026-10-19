import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df = pd.read_csv("data/raw/transactions.csv")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    df.drop(columns=["timestamp", "transaction_id"], inplace=True)

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    with open("params.yaml", "r") as f:
        configuration = yaml.safe_load(f)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=configuration["test_size"], random_state=configuration["random_seed"], stratify=y)

    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)