from prometheus_client import start_http_server, Summary, Gauge, Counter
import time
import pandas as pd
from joblib import load
import os

# Define Prometheus metrics.
# A Summary for tracking request processing time.
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
# Gauges for model metrics, as these values can go up and down.
MODEL_ACCURACY = Gauge('fraud_model_accuracy', 'Model accuracy')
MODEL_PRECISION = Gauge('fraud_model_precision', 'Model precision')
MODEL_RECALL = Gauge('fraud_model_recall', 'Model recall')
# A Counter for the total number of predictions, as it only ever increases.
PREDICTION_COUNT = Counter('fraud_model_prediction_count', 'Total predictions made')

PORT = 8000  # Port for Prometheus metrics HTTP server

def compute_metrics():
    """
    Computes model metrics and updates the Prometheus gauges and counter.
    This function expects 'claims_features.csv' and 'fraud_model.joblib' to exist.
    """
    # Check if data and model files exist to prevent errors.
    if not os.path.exists('data_engineering/claims_features.csv') or not os.path.exists('fraud_model.joblib'):
        print("Required files not found. Skipping metric computation.")
        return
        
    # Load the dataset and the pre-trained model.
    df = pd.read_csv('data_engineering/claims_features.csv')
    model = load('fraud_model.joblib')
    
    # Prepare data for prediction.
    feature_cols = [col for col in df.columns if col != 'is_fraud']
    X = df[feature_cols]
    y = df['is_fraud']
    
    # Make predictions and compute evaluation metrics.
    y_pred = model.predict(X)
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    
    # Update the Prometheus metrics with the new values.
    MODEL_ACCURACY.set(acc)
    MODEL_PRECISION.set(prec)
    MODEL_RECALL.set(rec)
    PREDICTION_COUNT.inc(len(y_pred))
    # Log metric values
    print(f"Model Accuracy: {acc}")
    print(f"Model Precision: {prec}")
    print(f"Model Recall: {rec}")
    print(f"Prediction Count: {len(y_pred)}")

@REQUEST_TIME.time()
def process_prediction():
    """
    A function that orchestrates the metric computation.
    The @REQUEST_TIME.time() decorator automatically measures the function's duration.
    """
    compute_metrics()

if __name__ == "__main__":
    print(f"Starting Prometheus metrics HTTP server on port {PORT}...")
    start_http_server(PORT)
    print("Processing and exposing metrics...")
    while True:
        process_prediction()
        time.sleep(30)  # Update metrics every 30 seconds