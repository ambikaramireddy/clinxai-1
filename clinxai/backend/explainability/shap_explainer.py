import shap
import pandas as pd
import numpy as np

class TabularExplainer:
    def __init__(self, model, background_data):
        """
        model: Trained sklearn estimator (RandomForest/XGBoost)
        background_data: numpy array of background samples for SHAP (e.g. training set sample)
        """
        self.model = model
        # For tree models, simple TreeExplainer is fast
        self.explainer = shap.TreeExplainer(model)
        
    def explain_local(self, input_vector: np.ndarray, feature_names: list):
        """
        Generates SHAP values for a single prediction.
        input_vector: shape (1, n_features)
        Returns list of dicts: [{'feature': 'Age', 'value': 25, 'shap': 0.1}, ...]
        """
        # Calculate SHAP values
        # shap_values shape: (1, n_features, n_classes) or (1, n_features) depending on model type
        shap_values = self.explainer.shap_values(input_vector)
        
        # Handle multi-class output - taking values for class 1 (Pneumonia)
        if isinstance(shap_values, list):
            # RandomForestClassifier returns list of arrays [class0_shap, class1_shap]
            shap_vals = shap_values[1][0] 
        else:
            # XGBoost binary 
            shap_vals = shap_values[0]

        explanation = []
        for i, name in enumerate(feature_names):
            val = shap_vals[i]
            # Safely handle various numpy/scalar types
            if isinstance(val, (np.ndarray, list)):
                 s_val = float(val) if np.size(val) == 1 else float(val[0])
            else:
                 s_val = float(val)

            explanation.append({
                "feature": name,
                "value": float(input_vector[0][i]),
                "shap_value": s_val
            })
            
        # Sort by absolute impact
        explanation.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        return explanation
