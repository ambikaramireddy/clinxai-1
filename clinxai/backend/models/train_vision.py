"""
ARCHIVE NOTICE:
===============
This training script is now ARCHIVED. The ClinXAI system uses pre-trained 
medical models from TorchXRayVision instead of training from scratch.

Specifically, we use:
- Model: DenseNet121 (densenet121-res224-rsna)
- Trained on: RSNA Pneumonia Challenge Dataset
- Source: mlmed/torchxrayvision library
- Citation: Cohen et al. (2020) - TorchXRayVision: A library of chest X-ray 
  datasets and models. https://github.com/mlmed/torchxrayvision

This approach provides:
1. Better accuracy with real medical data
2. Academically defensible pretrained weights
3. Faster deployment without data collection

For academic references, cite the TorchXRayVision paper and the RSNA dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
import os

def train_vision_model():
    print("Initializing Vision Model Training...")
    
    # 1. Setup Data
    data_dir = "c:/Users/ijgam/OneDrive/Desktop/soha/clinxai/data/raw"
    # For this mock/demo, we'll create a dummy folder structure for ImageFolder if it doesn't exist
    # or just manually load the mock images. 
    # To keep it standard, let's assume we have a custom dataset or we reorganize mock data.
    # checking valid mock images exist:
    
    # Create a simplified dataset wrapper for our mock files
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, root_dir):
            self.files = [f for f in os.listdir(root_dir) if f.endswith('.png')]
            self.root = root_dir
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            # Dummy labels: alternates 0 and 1
            self.labels = [i % 2 for i in range(len(self.files))]

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
             from PIL import Image
             path = os.path.join(self.root, self.files[idx])
             image = Image.open(path).convert('RGB')
             return self.transform(image), self.labels[idx]

    dataset = MockDataset(data_dir)
    if len(dataset) == 0:
        print("No images found. Run mock_data_gen.py first.")
        return

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    # 2. Setup Model (ResNet-18)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # 2 classes: Normal, Pneumonia

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Training Loop (1 mock epoch)
    print("Training for 1 epoch...")
    model.train()
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Batch Loss: {loss.item():.4f}")

    # 4. Save Model
    save_path = "c:/Users/ijgam/OneDrive/Desktop/soha/clinxai/backend/models/resnet18_pneumonia.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Vision Model Saved: {save_path}")

if __name__ == "__main__":
    train_vision_model()
