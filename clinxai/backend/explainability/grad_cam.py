import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer=None):
        """
        Initialize Grad-CAM for the given model.
        
        Args:
            model: The vision model (e.g., DenseNet121 from TorchXRayVision)
            target_layer: The layer to hook for gradient computation.
                         For DenseNet121, defaults to features.denseblock4
        """
        self.model = model
        
        # Auto-detect target layer if not specified
        if target_layer is None:
            # For TorchXRayVision DenseNet models
            if hasattr(model, 'features') and hasattr(model.features, 'denseblock4'):
                target_layer = model.features.denseblock4
                print("Grad-CAM: Using DenseNet denseblock4 as target layer")
            elif hasattr(model, 'layer4'):  # Fallback for ResNet
                target_layer = model.layer4
                print("Grad-CAM: Using ResNet layer4 as target layer")
            else:
                raise ValueError("Could not auto-detect target layer. Please specify manually.")
        
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook for gradients and activations
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generates Grad-CAM heatmap for the given input tensor.
        input_tensor: (1, 3, 224, 224)
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # Backward pass
        output[:, class_idx].backward()
        
        # Pool gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight activations by gradients
        activations = self.activations[0] # (C, H, W)
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the activations
        heatmap = torch.mean(activations, dim=0).detach().cpu()
        
        # ReLU
        heatmap = F.relu(heatmap)
        
        # Normalize
        heatmap /= torch.max(heatmap)
        
        return heatmap.numpy()

    def overlay_heatmap(self, heatmap, original_image_path, alpha=0.4):
        """
        Overlays heatmap on original image.
        original_image_path: Path to the original image file
        alpha: transparency of heatmap
        """
        img = cv2.imread(original_image_path)
        img = cv2.resize(img, (224, 224))
        
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        
        heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        superimposed_img = heatmap_img * alpha + img * (1 - alpha)
        return np.uint8(superimposed_img)
