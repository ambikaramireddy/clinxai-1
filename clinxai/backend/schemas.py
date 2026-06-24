from pydantic import BaseModel
from typing import Optional, List

# --- Request Models ---

class PatientData(BaseModel):
    Age: int
    Gender: str  # "M" or "F"
    Fever: int   # 0 or 1
    Cough: int   # 0 or 1
    Oxygen: int  # 0-100

# --- Response Models ---

class PredictionResponse(BaseModel):
    diagnosis: str          # "Pneumonia" or "Normal"
    confidence: float       # 0.0 to 1.0 (Probability of Pneumonia)
    risk_level: str         # "Low", "Medium", "High"
    
class ExplanationImageResponse(BaseModel):
    heatmap_base64: str     # Base64 encoded image
    
class SHAPFeature(BaseModel):
    feature: str
    value: float
    shap_value: float

class ExplanationPatientResponse(BaseModel):
    features: List[SHAPFeature]
