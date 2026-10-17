import pandas as pd
import pandera.pandas as pa

class TransactionModel(pa.DataFrameModel):
    transaction_id: int = pa.Field(unique=True, ge=0)
    customer_id: str = pa.Field(nullable=False)
    amount:float = pa.Field(gt=0.0)
    country: str = pa.Field(nullable=False, isin=["USA", "Canada", "UK", "Germany", "Italy", 
           "France", "Australia", "Japan", "China", "India", "Brazil", "Peru", "Mexico", "Argentina"])
    merchant: str = pa.Field(nullable=False, isin=["Amazon", "Sony", "Samsung", "KFC", "eBay",
             "Apple", "Microsoft", "Starbucks", "McDonald's", "Nike", "Adidas"])
    timestamp: str = pa.Field(nullable=False)
    is_fraud: int = pa.Field(isin=[0, 1])

    class Config:
        strict = True

if __name__ == "__main__":
    df = pd.read_csv("data/raw/transactions.csv")

    validated_df = TransactionModel.validate(df)
    print("Data validation successful. The DataFrame conforms to the schema.")