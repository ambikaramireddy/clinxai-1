import torch
from torchvision import models
import pickle
import os
import pandas as pd
import sys
import torchxrayvision as xrv

# Ensure backend imports work
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from backend.utils.preprocess_patient import PreprocessPatient
from backend.explainability.shap_explainer import TabularExplainer
from backend.models.fusion_model import FusionModel

def load_models(base_path=None):
    if base_path is None:
        base_path = os.path.join(project_root, "backend", "models")
    print("Loading models...")
    print("Loading models...")
    artifacts = {}
    
    # 1. Vision Model (Pre-trained DenseNet121 from TorchXRayVision)
    print(" - Loading pre-trained DenseNet121 (RSNA Pneumonia Challenge)...")
    try:
        vision_model = xrv.models.DenseNet(weights="densenet121-res224-rsna")
        vision_model.eval()
        artifacts['vision_model'] = vision_model
        
        # Store pneumonia index for extraction
        # RSNA model pathologies: model.pathologies lists all 18 classes
        artifacts['pneumonia_idx'] = list(vision_model.pathologies).index('Pneumonia')
        
        print(f" - Vision model loaded. Pneumonia index: {artifacts['pneumonia_idx']}")
        print(f" - Available pathologies: {vision_model.pathologies}")
    except Exception as e:
        print(f"ERROR loading pre-trained model: {e}")
        print("Falling back to placeholder model...")
        # Fallback for development
        vision_model = xrv.models.DenseNet(weights="densenet121-res224-all")
        vision_model.eval()
        artifacts['vision_model'] = vision_model
        artifacts['pneumonia_idx'] = list(vision_model.pathologies).index('Pneumonia')

    # 2. Tabular Model (RandomForest)
    tabular_path = os.path.join(base_path, "rf_patient.pkl")
    if os.path.exists(tabular_path):
        with open(tabular_path, "rb") as f:
            artifacts['tabular_model'] = pickle.load(f)
        print(" - Tabular model loaded.")
    else:
        print(f"WARNING: Tabular model not found at {tabular_path}")

    # 3. Patient Preprocessor (Scaler)
    # PreprocessPatient handles loading its own scaler internally during transform, 
    # but we initialize the class here to ensure paths are set.
    artifacts['patient_preprocessor'] = PreprocessPatient(artifact_path=base_path)
    
    # 4. SHAP Explainer
    # Needs background data. We'll use mock data for now or sample from training data if available.
    mock_csv = os.path.join(project_root, "data", "raw", "mock_patients.csv")
    if 'tabular_model' in artifacts and os.path.exists(mock_csv):
        df_bg = pd.read_csv(mock_csv).drop(columns=['Label'])
        # Transform background data
        X_bg = artifacts['patient_preprocessor'].transform(df_bg)
        artifacts['shap_explainer'] = TabularExplainer(artifacts['tabular_model'], X_bg)
        print(" - SHAP explainer initialized.")

    # 5. Fusion Model
    artifacts['fusion_model'] = FusionModel()

    return artifacts
