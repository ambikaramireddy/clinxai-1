import torch
import numpy as np

class FusionModel:
    def __init__(self):
        # Weighted average for late fusion
        # Heuristic: Image (60%) + Symptoms (40%)
        self.alpha = 0.6 

    def fuse(self, image_prob: float, patient_prob: float) -> float:
        """
        Combines probabilities from Vision and Tabular models.
        Returns fused probability of Pneumonia.
        """
        fused_score = (self.alpha * image_prob) + ((1 - self.alpha) * patient_prob)
        return fused_score
