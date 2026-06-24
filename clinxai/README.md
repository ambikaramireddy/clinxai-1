# ClinXAI - Real-Time Explainable Medical Diagnosis

## 📌 Project Overview
ClinXAI is a real-time, explainable AI system designed to assist medical professionals in diagnosing **Pneumonia** from Chest X-rays and patient symptoms. It focuses on providing instant inference (< 5s) with visual explainability (Grad-CAM, SHAP).

## 🔒 Verified Scope (Functional Integration)
> [!IMPORTANT]
> This project is a **Functional Integration Prototype**. It verifies the system wiring, real-time performance, and logic implementation. **It is NOT medically validated for clinical use.**

- **Disease Domain**: Pneumonia (Binary Classification).
- **Core Verification Status**:
  - **Latency**: `0.56s` Total Inference Time (Goal: <5s).
  - **Logic**: Risk Thresholds and Fusion Logic verified.
  - **Integration**: Backend <-> Frontend end-to-end flow verified.


## 🏗 Architecture
- **Backend**: FastAPI (Python) - Async inference, fusion logic.
- **Frontend**: HTML5/CSS3/Three.js - Interactive 3D visuals, real-time feedback.
- **AI Models**:
  - **Vision**: Pre-trained DenseNet-121 from TorchXRayVision
    - Trained on: RSNA Pneumonia Challenge Dataset
    - Source: [mlmed/torchxrayvision](https://github.com/mlmed/torchxrayvision)
    - Citation: Cohen et al. (2020)
  - **Tabular**: XGBoost / Random Forest (patient symptoms).
  - **Fusion**: Late fusion (Probability concatenation).
- **Logic**:
  - **Diagnosis**: Positive if Probability > 50%.
  - **Risk Assessment**:
    - Low Risk: < 30%
    - Medium Risk: 30% - 70% (Borderline / Uncertainty)
    - High Risk: > 70%
    
> **Why Pre-trained Models?**  
> This system uses pre-trained medical models due to data constraints and to ensure predictions are based on real clinical distributions. The RSNA Pneumonia Challenge model provides academically defensible weights and significantly better accuracy than training from scratch on limited data.

## 📂 Folder Structure
```
clinxai/
├── backend/            # FastAPI Application
│   ├── models/         # Saved model weights (.pt, .pkl)
│   ├── inference/      # Inference logic
│   ├── explainability/ # Grad-CAM, SHAP logic
│   └── utils/          # Helper functions
├── frontend/           # Web UI
│   ├── css/            # Styles
│   ├── js/             # Logic (API, Three.js)
│   └── assets/         # Images, 3D models
├── data/               # Dataset storage (Raw/Processed)
└── notebooks/          # Experimental notebooks
```

## 🚀 Getting Started

### ⚡ One-Command Setup & Launch
Simply double-click **`setup.bat`** and it will:
- ✓ Check if Python and pip are installed
- ✓ Analyze and install all required dependencies automatically (first time only)
- ✓ Launch both backend and frontend servers
- ✓ Open your browser to the application

**No manual setup required!** Perfect for sharing with others. Run it every time - it's smart enough to skip dependency installation if everything is already installed.

---

### 🛠 Manual Startup (Advanced)
1. **Install Dependencies**: `pip install -r backend/requirements.txt`
2. **Run Backend**: `uvicorn backend.main:app --reload`
3. **Open Frontend**: Serve `frontend/index.html` via Live Server or Python http.server.
