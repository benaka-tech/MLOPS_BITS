"""
Data Engineering & Ingestion Pipeline for Health Insurance
Example: Claims data cleaning and feature engineering
"""
import pandas as pd

def load_claims_data(path):
    # Load raw claims data
    return pd.read_csv(path)

def clean_data(df):
    # Example cleaning: drop missing values, filter outliers
    df = df.dropna()
    # ... more cleaning steps ...
    return df

def feature_engineering(df):
    # Example: create total_claim_amount feature
    df['total_claim_amount'] = df['claim_amount'] * df['num_claims']
    # ... more feature engineering ...
    return df

if __name__ == "__main__":
    df = load_claims_data('data_engineering/claims.csv')
    df = clean_data(df)
    df = feature_engineering(df)
    df.to_csv('data_engineering/claims_features.csv', index=False)
