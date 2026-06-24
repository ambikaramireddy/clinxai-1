import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

class PreprocessPatient:
    def __init__(self, artifact_path="backend/models"):
        self.artifact_path = artifact_path
        self.scaler_path = os.path.join(self.artifact_path, "scaler.pkl")
        self.scaler = StandardScaler()
        os.makedirs(self.artifact_path, exist_ok=True)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fits scaler on data and transforms it. Saves scaler for inference.
        Assumes input dataframe has columns: ['Age', 'Gender', 'Fever', 'Cough', 'Oxygen']
        """
        processed = self._process_dataframe(df)
        scaled_data = self.scaler.fit_transform(processed)
        
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
            
        return scaled_data

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms data using loaded scaler.
        """
        if not os.path.exists(self.scaler_path):
             raise FileNotFoundError("Scaler not found. Train model first.")
        
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
            
        processed = self._process_dataframe(df)
        return self.scaler.transform(processed)

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Encode Gender: M=0, F=1 (Simple binary map for this scope)
        df['Gender'] = df['Gender'].map({'M': 0, 'F': 1}).fillna(0)
        
        # Ensure correct order
        cols = ['Age', 'Gender', 'Fever', 'Cough', 'Oxygen']
        return df[cols]
