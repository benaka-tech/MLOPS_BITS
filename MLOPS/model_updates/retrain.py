"""
Model Updates & Retraining for Health Insurance
Full pipeline: load new claims data, preprocess, retrain fraud detection model, evaluate, and save.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump

def load_new_data(path):
    """Load new claims data for retraining."""
    return pd.read_csv(path)

def preprocess_data(df):
    """Preprocess new claims data (cleaning, feature engineering)."""
    df = df.dropna()
    if 'total_claim_amount' not in df.columns:
        df['total_claim_amount'] = df['claim_amount'] * df['num_claims']
    # ... add more feature engineering as needed ...
    return df

def retrain_model(df):
    """Retrain fraud detection model and evaluate."""
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Retrained model evaluation:")
    print(classification_report(y_test, y_pred))
    dump(model, 'fraud_model.joblib')
    print("Model retrained and saved as fraud_model.joblib.")

if __name__ == "__main__":
    df = load_new_data('new_claims_features.csv')
    df = preprocess_data(df)
    retrain_model(df)
