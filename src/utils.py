import os
import joblib

def save_model(model, model_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        joblib.dump(model, f)

def load_model(model_path):
    # Load and return the model if the file exists
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return joblib.load(f)
    else:
        raise FileNotFoundError(f"No model found at {model_path}")
