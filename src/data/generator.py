import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timedelta

clients = ["C001", "C002", "C003", "C004", "C005", "C006", "C007", "C008", "C009", "C010"]
countries = ["USA", "Canada", "UK", "Germany", "Italy", 
           "France", "Australia", "Japan", "China", "India", "Brazil", "Peru", "Mexico", "Argentina"]
merchants = ["Amazon", "Sony", "Samsung", "KFC", "eBay",
             "Apple", "Microsoft", "Starbucks", "McDonald's", "Nike", "Adidas"]

def generate_timestamp(start_date: datetime, end_date: datetime):
    total_seconds = int((end_date - start_date).total_seconds())
    random_seconds = np.random.randint(0, total_seconds)

    return start_date + timedelta(seconds=random_seconds)

def generate_transaction(id_transaction: int, fraud_rate: float):
    customer_id = str(np.random.choice(clients))
    amount = round(np.random.uniform(1, 10000), 2)
    country = str(np.random.choice(countries))
    merchant = str(np.random.choice(merchants))
    timestamp =str(generate_timestamp(datetime(2023, 1, 1), datetime(2026, 12, 31)))
    is_fraud = int(np.random.choice([0, 1], p=[1 - fraud_rate, fraud_rate]))

    return {
        "transaction_id": id_transaction,
        "customer_id": customer_id,
        "amount": amount,
        "country": country,
        "merchant": merchant,
        "timestamp": timestamp,
        "is_fraud": is_fraud
    }

if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        configuration = yaml.safe_load(f)

    np.random.seed(configuration["random_seed"])

    transactions = []
    for i in range(1, configuration["num_records"] + 1):
        transactions.append(generate_transaction(i, configuration["fraud_rate"]))

    df = pd.DataFrame(transactions)
    df.to_csv("data/raw/transactions.csv", index=False)
