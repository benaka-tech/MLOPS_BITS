"""
Model Development for Health Insurance
Example: Fraud detection model training
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import mlflow

def train_model(data_path):
    df = pd.read_csv(data_path)
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    mlflow.sklearn.log_model(model, "fraud_model")
    return model

if __name__ == "__main__":
    train_model('claims_features.csv')
