from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CLIENTS = [f"C{i:03d}" for i in range(1, 11)]
COUNTRIES = ["USA", "Canada", "UK", "Germany", "Italy", "France", "Australia",
"Japan", "China", "India", "Brazil", "Peru", "Mexico", "Argentina"]
MERCHANTS = ["Amazon", "Sony", "Samsung", "KFC", "eBay", "Apple", "Microsoft",
"Starbucks", "McDonald's", "Nike", "Adidas"]

RISKY_COUNTRIES = {"China", "India", "Brazil", "Peru", "Mexico", "Argentina"}
RISKY_MERCHANTS = {"eBay", "KFC", "McDonald's"}
RISKY_CUSTOMERS = {"C008", "C009", "C010"}

def generate_timestamp(start_date: datetime, end_date: datetime) -> datetime:
    total_seconds = int((end_date - start_date).total_seconds())
    random_seconds = np.random.randint(0, total_seconds)
    return start_date + timedelta(seconds=int(random_seconds))

def fraud_risk_score(
    amount: float,
    country: str,
    merchant: str,
    customer_id: str,
    timestamp: datetime,
) -> float:
    """Build a noisy but learnable fraud propensity score."""
    score = float(np.random.normal(0.0, 0.65))
    hour = timestamp.hour

    high_amount = amount >= 7500
    very_high_amount = amount >= 9000
    night_transaction = hour <= 4 or hour >= 23
    risky_country = country in RISKY_COUNTRIES
    risky_merchant = merchant in RISKY_MERCHANTS

    score += 1.2 if high_amount else 0.0
    score += 1.0 if very_high_amount else 0.0
    score += 1.5 if night_transaction else 0.0
    score += 1.0 if risky_country else 0.0
    score += 0.9 if risky_merchant else 0.0
    score += 0.5 if customer_id in RISKY_CUSTOMERS else 0.0

    # Interactions make suspicious combinations more informative than any
    # individual feature on its own.
    score += 1.4 if high_amount and night_transaction else 0.0
    score += 1.2 if risky_country and risky_merchant else 0.0

    return score

def generate_transaction(transaction_id: int) -> dict:
    customer_id = str(np.random.choice(CLIENTS))
    amount = round(float(np.random.uniform(1, 10000)), 2)
    country = str(np.random.choice(COUNTRIES))
    merchant = str(np.random.choice(MERCHANTS))
    timestamp = generate_timestamp(datetime(2023, 1, 1), datetime(2026, 12, 31))

    return {
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "amount": amount,
        "country": country,
        "merchant": merchant,
        "timestamp": timestamp.isoformat(sep=" "),
        "_fraud_score": fraud_risk_score(
            amount=amount,
            country=country,
            merchant=merchant,
            customer_id=customer_id,
            timestamp=timestamp,
        ),
    }

def assign_fraud_labels(df: pd.DataFrame, fraud_rate: float) -> pd.DataFrame:
    """Assign fraud to the highest-risk observations at the configured rate."""
    if not 0 <= fraud_rate <= 1:
        raise ValueError("fraud_rate must be between 0 and 1.")

    target_frauds = int(round(len(df) * fraud_rate))
    labels = np.zeros(len(df), dtype=int)

    if target_frauds > 0:
        fraud_indices = df["_fraud_score"].nlargest(target_frauds).index
        labels[fraud_indices] = 1

    result = df.drop(columns=["_fraud_score"]).copy()
    result["is_fraud"] = labels
    return result

if __name__ == "__main__":
    with open("params.yaml", "r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    np.random.seed(configuration["random_seed"])

    transactions = [
        generate_transaction(i)
        for i in range(1, configuration["num_records"] + 1)
    ]
    df = assign_fraud_labels(
        pd.DataFrame(transactions),
        fraud_rate=configuration["fraud_rate"],
    )

    output_path = Path("data/raw/transactions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
