from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import numpy as np
import pandas as pd
import io
import base64
import cv2
from PIL import Image
import sys
import time
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from backend.schemas import PatientData, PredictionResponse, ExplanationImageResponse, ExplanationPatientResponse
from backend.inference.loader import load_models
from backend.utils.preprocess_image import PreprocessImage
from backend.explainability.grad_cam import GradCAM

app = FastAPI(title="ClinXAI API", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
class AppState:
    models = {}
    image_preprocessor = PreprocessImage()

state = AppState()

@app.on_event("startup")
async def startup_event():
    state.models = load_models()
    print("System verified and ready.")

# --- Endpoints ---

@app.post("/predict/fused", response_model=PredictionResponse)
async def predict_fused(
    age: int, gender: str, fever: int, cough: int, oxygen: int,
    file: UploadFile = File(...)
):
    """
    Full inference: Image + Patient Data -> Fused Prediction
    """
    try:
        t_start = time.time()
        timing_stats = {}

        # 1. Process Image
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            img_tensor = state.image_preprocessor.preprocess_pil(image)
            timing_stats['image_proc'] = time.time() - t_start
        except Exception as e:
            print(f"Image processing error: {e}")
            raise HTTPException(status_code=400, detail="Invalid image file")

        # 2. Process Patient Data
        t_tab_start = time.time()
        patient_df = pd.DataFrame([{
            "Age": age, "Gender": gender, "Fever": fever, "Cough": cough, "Oxygen": oxygen
        }])
        
        if 'patient_preprocessor' not in state.models:
            print("Model loader error: patient_preprocessor missing")
            raise HTTPException(status_code=500, detail="Models not loaded")
            
        patient_vec = state.models['patient_preprocessor'].transform(patient_df)
        timing_stats['tabular_proc'] = time.time() - t_tab_start

        # 3. Vision Inference
        t_vis_start = time.time()
        vision_model = state.models['vision_model']
        pneumonia_idx = state.models.get('pneumonia_idx', 0)  # Default to 0 if not set
        
        with torch.no_grad():
            vision_out = vision_model(img_tensor)
            # TorchXRayVision outputs raw logits for 18 pathologies
            # Apply sigmoid (not softmax) since it's multi-label classification
            vision_probs = torch.sigmoid(vision_out)[0]
            vision_prob = vision_probs[pneumonia_idx].item()  # Extract pneumonia probability
            
        timing_stats['vision_infer'] = time.time() - t_vis_start

        # 4. Tabular Inference
        t_tab_infer_start = time.time()
        tabular_model = state.models['tabular_model']
        # RandomForest predict_proba returns [ [prob_0, prob_1] ]
        tabular_prob = tabular_model.predict_proba(patient_vec)[0][1]
        timing_stats['tabular_infer'] = time.time() - t_tab_infer_start

        # 5. Fusion
        t_fusion_start = time.time()
        fusion_model = state.models['fusion_model']
        final_score = fusion_model.fuse(vision_prob, tabular_prob)
        timing_stats['fusion'] = time.time() - t_fusion_start

        # 6. Response Logic & Risk Mapping
        # Logic: 
        # - Diagnosis is binary based on 0.5 threshold.
        # - Risk is 3-tier: Low (<0.3), Medium (0.3-0.7), High (>0.7).
        # Note: Medium Risk (e.g. 0.45) with Normal Diagnosis is valid; implies "Borderline Normal".
        diagnosis = "Pneumonia" if final_score > 0.5 else "Normal"
        
        if final_score < 0.3: risk = "Low"
        elif final_score < 0.7: risk = "Medium"
        else: risk = "High"

        total_elapsed = time.time() - t_start
        print(f"--- Inference Timing ---")
        print(f"Image Preproc: {timing_stats['image_proc']:.4f}s")
        print(f"Vision Infer:  {timing_stats['vision_infer']:.4f}s")
        print(f"Tabular Proc:  {timing_stats['tabular_proc']:.4f}s")
        print(f"Tabular Infer: {timing_stats['tabular_infer']:.4f}s")
        print(f"Fusion / Logic: {timing_stats['fusion']:.4f}s")
        print(f"Total Latency: {total_elapsed:.4f}s")
        print(f"------------------------")

        with open("latency.log", "a") as f:
            f.write(f"{time.ctime()} | Total: {total_elapsed:.4f}s | Vision: {timing_stats['vision_infer']:.4f}s | Tabular: {timing_stats['tabular_infer']:.4f}s | Fusion: {timing_stats['fusion']:.4f}s\n")
        
        return {
            "diagnosis": diagnosis,
            "confidence": float(final_score),
            "risk_level": risk
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain/image", response_model=ExplanationImageResponse)
async def explain_image(file: UploadFile = File(...)):
    """
    Returns Grad-CAM heatmap overlay as Base64 string.
    """
    # 1. Process Image
    contents = await file.read()
    image_pil = Image.open(io.BytesIO(contents))
    img_tensor = state.image_preprocessor.preprocess_pil(image_pil)
    
    # 2. Grad-CAM
    vision_model = state.models['vision_model']
    pneumonia_idx = state.models.get('pneumonia_idx', 0)
    
    # GradCAM will auto-detect the DenseNet target layer
    grad_cam = GradCAM(vision_model)  # Auto-detect denseblock4
    heatmap = grad_cam.generate_heatmap(img_tensor, class_idx=pneumonia_idx)
    
    # 3. Overlay
    # Need original image as numpy (H,W,3) for overlay
    # Resize PIL to 224x224 to match heatmap
    img_resized = image_pil.resize((224, 224)).convert('RGB')
    
    # Temporary save for cv2 read pattern used in previous code, 
    # BUT let's refactor overlay_heatmap to accept numpy array to avoid disk IO for speed
    # We'll update the GradCAM class/method inline here or in util file. 
    # For now, let's adapt here.
    
    # Convert PIL to BGR (cv2 format)
    img_cv2 = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
    
    # Manual Overlay Logic (copied from util for speed/direct access)
    t_cam_start = time.time()
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    alpha = 0.4
    overlay = heatmap_color * alpha + img_cv2 * (1 - alpha)
    overlay = np.uint8(overlay)
    
    # 4. Convert directly to Base64
    _, buffer = cv2.imencode('.jpg', overlay)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    print(f"Grad-CAM Generation: {time.time() - t_cam_start:.4f}s")
    
    return {"heatmap_base64": img_base64}

@app.post("/explain/patient", response_model=ExplanationPatientResponse)
async def explain_patient(data: PatientData):
    """
    Returns SHAP values for the patient data.
    """
    if 'shap_explainer' not in state.models:
         raise HTTPException(status_code=500, detail="SHAP Explainer not initialized")

    t_shap_start = time.time()
    df = pd.DataFrame([data.dict()])
    input_vec = state.models['patient_preprocessor'].transform(df)
    
    explanation = state.models['shap_explainer'].explain_local(
        input_vec, 
        ['Age', 'Gender', 'Fever', 'Cough', 'Oxygen']
    )
    print(f"SHAP Generation: {time.time() - t_shap_start:.4f}s")
    
    return {"features": explanation}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
