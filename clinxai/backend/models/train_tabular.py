import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import sys

# Ensure backend modules can be imported
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)
from backend.utils.preprocess_patient import PreprocessPatient

def train_tabular_model():
    print("Initializing Tabular Model Training (RandomForest)...")
    
    # 1. Load Data
    csv_path = os.path.join(project_root, "data", "raw", "mock_patients.csv")
    if not os.path.exists(csv_path):
        print("Mock CSV not found.")
        return

    df = pd.read_csv(csv_path)
    X = df.drop(columns=['Label'])
    y = df['Label']

    # 2. Preprocess
    artifact_path = os.path.join(project_root, "backend", "models")
    preprocessor = PreprocessPatient(artifact_path=artifact_path)
    X_processed = preprocessor.fit_transform(X)

    # 3. Train RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_processed, y)

    # 4. Save Model
    save_path = os.path.join(artifact_path, "rf_patient.pkl")
    
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Tabular Model Saved: {save_path}")

if __name__ == "__main__":
    train_tabular_model()
