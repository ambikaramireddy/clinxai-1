import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torchxrayvision as xrv

class PreprocessImage:
    def __init__(self):
        """
        Preprocessing for medical chest X-rays using TorchXRayVision standards.
        Converts to grayscale and normalizes to [-1024, 1024] range (DICOM-like).
        """
        self.transform = transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])

    def preprocess(self, image_path: str) -> torch.Tensor:
        """
        Loads a chest X-ray image and applies medical preprocessing.
        Returns tensor of shape (1, 1, 224, 224) - grayscale medical X-ray.
        """
        # Load image using skimage (handles various formats)
        import skimage.io
        img = skimage.io.imread(image_path)
        
        # CRITICAL: Convert to float32 BEFORE normalization
        img = img.astype(np.float32)
        
        # Normalize 8-bit image to [-1024, 1024] range (medical standard)
        # maxval parameter indicates the input range (255 for 8-bit images)
        img = xrv.datasets.normalize(img, maxval=255)
        
        # Convert to single channel if needed (grayscale)
        if len(img.shape) == 3:
            img = img.mean(2)  # Average RGB channels
        
        # Add channel dimension: (H, W) -> (1, H, W)
        img = img[None, ...]
        
        # Apply transforms (center crop and resize)
        img = self.transform(img)
        
        # Convert to torch tensor and add batch dimension
        img = torch.from_numpy(img).unsqueeze(0)  # (1, 1, 224, 224)
        
        return img

    def preprocess_pil(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocesses a PIL Image object (for uploaded files).
        """
        # Convert PIL to numpy array
        img = np.array(image)
        
        # CRITICAL: Convert to float32 BEFORE normalization
        img = img.astype(np.float32)
        
        # Normalize to medical range (maxval=255 for standard 8-bit images)
        img = xrv.datasets.normalize(img, maxval=255)
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = img.mean(2)
        
        # Add channel dimension
        img = img[None, ...]
        
        # Apply transforms
        img = self.transform(img)
        
        # Convert to torch tensor
        img = torch.from_numpy(img).unsqueeze(0)
        
        return img
