import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from PIL import Image

def create_mock_data():
    base_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(base_dir, exist_ok=True)

    # 1. Create Mock Patient CSV
    print("Creating mock patient data...")
    data = {
        "Age": [25, 45, 65, 30, 70],
        "Gender": ["M", "F", "M", "F", "F"],
        "Fever": [0, 1, 1, 0, 1],
        "Cough": [0, 1, 1, 0, 1],
        "Oxygen": [98, 95, 90, 99, 88],
        "Label": [0, 1, 1, 0, 1] # 0=Normal, 1=Pneumonia
    }
    df = pd.DataFrame(data)
    csv_path = os.path.join(base_dir, "mock_patients.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # 2. Create Mock X-ray Images (224x224 grayscale)
    print("Creating mock X-ray images...")
    for i in range(3):
        img_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img_path = os.path.join(base_dir, f"mock_xray_{i}.png")
        img.save(img_path)
        print(f"Saved: {img_path}")

if __name__ == "__main__":
    create_mock_data()
