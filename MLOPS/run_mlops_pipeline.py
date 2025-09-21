"""
End-to-end MLOps pipeline for Health Insurance Fraud Detection
Stages: Data Engineering, Model Development, Validation, Retraining, Audit Logging
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from joblib import dump, load
import logging
from datetime import datetime, timezone
from prometheus_client import Gauge, Counter, start_http_server
from sklearn.metrics import f1_score

## --- Prometheus Metrics ---
MODEL_ACCURACY = Gauge('fraud_model_accuracy', 'Model accuracy')
MODEL_PRECISION = Gauge('fraud_model_precision', 'Model precision')
MODEL_RECALL = Gauge('fraud_model_recall', 'Model recall')
PREDICTION_COUNT = Counter('fraud_model_prediction_count', 'Total predictions made')
MODEL_F1 = Gauge('fraud_model_f1', 'Model F1-score')

# --- Data Engineering ---
def load_claims_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.dropna()
    return df

def feature_engineering(df):
    df['total_claim_amount'] = df['claim_amount'] * df['num_claims']
    return df

# --- Model Development ---
def train_model(df):
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))
    dump(model, 'fraud_model.joblib')
    return model, X_test, y_test

# --- Model Validation ---
def validate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Validation Report:")
    print(classification_report(y_test, y_pred))
    # Add fairness/drift checks as needed

# --- Model Retraining ---
def retrain_model(new_data_path):
    df = load_claims_data(new_data_path)
    df = clean_data(df)
    df = feature_engineering(df)
    feature_cols = [col for col in df.columns if col != 'is_fraud']
    X = df[feature_cols]
    y = df['is_fraud']
    model = RandomForestClassifier()
    model.fit(X, y)
    dump(model, 'fraud_model.joblib')
    print("Model retrained and saved.")

# --- Audit Logging ---
logging.basicConfig(filename='audit.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

def log_prediction(prediction, member_id, model_name, user_id):
    entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'member_id': member_id,
        'prediction': prediction,
        'model_name': model_name,
        'user_id': user_id
    }
    logging.info(f"AUDIT: {entry}")

if __name__ == "__main__":
    # Start Prometheus metrics server
    start_http_server(9100)

    # Data Engineering
    df = load_claims_data('data_engineering/claims.csv')
    df = clean_data(df)
    df = feature_engineering(df)
    df.to_csv('data_engineering/claims_features.csv', index=False)

    # Model Development
    model, X_test, y_test = train_model(df)

    # Model Validation
    validate_model(model, X_test, y_test)

    # Export metrics for monitoring
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    MODEL_ACCURACY.set(acc)
    MODEL_PRECISION.set(prec)
    MODEL_RECALL.set(rec)
    MODEL_F1.set(f1)
    PREDICTION_COUNT.inc(len(y_pred))

    # Model Retraining (optional, using same data for demo)
    retrain_model('data_engineering/claims_features.csv')

    # Audit Logging (example)
    feature_cols = [col for col in df.columns if col != 'is_fraud']
    for idx, row in df.iterrows():
        pred = model.predict([row[feature_cols].values])[0]
        log_prediction(pred, row.get('member_id', idx), 'fraud_model_v1', 'analyst_001')
