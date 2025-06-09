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

    def predict(self, image: Image.Image) -> np.ndarray:
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

        depth = prediction.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        return depth


def load_model():
    return DPTDepthModel()
