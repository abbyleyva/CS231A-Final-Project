import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np

class DPTDepthModel:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_type = "DPT_Hybrid"
        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        self.model.to(self.device)
        self.model.eval()

    def scale_depth_for_kitti(self, normalized_depth):
        """Scale normalized depth to real-world meters for KITTI driving scenarios"""
        # KITTI vehicles typically 5-50 meters away
        return 5.0 + normalized_depth * 45.0

    def predict(self, image: Image.Image, scale_to_meters=True) -> np.ndarray:
        image_np = np.array(image)
        input_tensor = self.transform(image_np).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.size[::-1],  # (H, W)
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()  # <- FIXED: Added this line
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        
        # NEW: Scale to meters if requested
        if scale_to_meters:
            depth = self.scale_depth_for_kitti(depth)
        
        return depth


def load_model():
    return DPTDepthModel()