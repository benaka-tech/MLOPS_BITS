"""
CI/CD for Health Insurance Models
Example: Package and deploy fraud detection model
"""
import mlflow
import docker
# ... Kubernetes deployment code ...

def package_model(model_path):
    # Example: log model with MLflow
    mlflow.sklearn.log_model(model_path, "fraud_model")
    # ... Docker packaging ...
    # ... Kubernetes deployment ...

if __name__ == "__main__":
    package_model('fraud_model.joblib')
