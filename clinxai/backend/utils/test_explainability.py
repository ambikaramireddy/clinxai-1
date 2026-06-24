import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torchvision import models
import pickle
import pandas as pd
from PIL import Image

# Ensure path is correct
sys.path.append(project_root)

from backend.explainability.grad_cam import GradCAM
from backend.explainability.shap_explainer import TabularExplainer
from backend.utils.preprocess_image import PreprocessImage
from backend.utils.preprocess_patient import PreprocessPatient

def test_explainability():
    print("Testing Explainability Engine...")
    
    # --- 1. Test Grad-CAM ---
    print("\n[1/2] Testing Grad-CAM...")
    
    # Load Model
    model_path = os.path.join(project_root, "backend", "models", "resnet18_pneumonia.pt")
    if not os.path.exists(model_path):
        print("Model not found. Run Phase 1 training first.")
        return

    model = models.resnet18(weights=None) # Structure only
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Find mock image
    data_dir = os.path.join(project_root, "data", "raw")
    images = [f for f in os.listdir(data_dir) if f.endswith('.png')]
    if not images:
        print("No mock images found.")
        return
    
    img_path = os.path.join(data_dir, images[0])
    
    # Preprocess
    img_pre = PreprocessImage()
    input_tensor = img_pre.preprocess(img_path)
    
    # Run Grad-CAM
    # ResNet-18 last conv layer is inside layer4
    target_layer = model.layer4[-1].conv2
    
    # Bug fix: backward hook signature might vary in newer torch versions
    # For now relying on simple implementation. 
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate_heatmap(input_tensor)
    
    print(f"Heatmap generated. Shape: {heatmap.shape}, Max: {heatmap.max()}, Min: {heatmap.min()}")
    
    # Overlay
    overlay = grad_cam.overlay_heatmap(heatmap, img_path)
    save_path = os.path.join(project_root, "backend", "utils", "test_gradcam_result.jpg")
    
    from PIL import Image as PILImage
    PILImage.fromarray(overlay).save(save_path)
    print(f"Overlay saved to: {save_path}")


    # --- 2. Test SHAP ---
    print("\n[2/2] Testing SHAP...")
    
    # Load Model
    rf_path = os.path.join(project_root, "backend", "models", "rf_patient.pkl")
    if not os.path.exists(rf_path):
        print("Tabular model not found.")
        return
        
    with open(rf_path, "rb") as f:
        rf_model = pickle.load(f)
        
    # Mock data for inference
    patient_pre = PreprocessPatient(artifact_path=os.path.join(project_root, "backend", "models"))
    
    # Create single row DF
    sample = pd.DataFrame([{
        "Age": 65, "Gender": "M", "Fever": 1, "Cough": 1, "Oxygen": 90
    }])
    
    input_vector = patient_pre.transform(sample)
    
    # Init Explainer (usually needs background data, here using same vector for simplicity or loading mock csv)
    # Ideally should use training data as background.
    csv_path = os.path.join(project_root, "data", "raw", "mock_patients.csv")
    df_train = pd.read_csv(csv_path).drop(columns=['Label'])
    X_train = patient_pre.transform(df_train)
    
    explainer = TabularExplainer(rf_model, X_train)
    
    explanation = explainer.explain_local(input_vector, ['Age', 'Gender', 'Fever', 'Cough', 'Oxygen'])
    print("SHAP Explanation:")
    for item in explanation:
        print(item)

if __name__ == "__main__":
    test_explainability()
