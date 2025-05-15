"""
Basic KITTI dataset loader for 3D object detection
"""

import os
import cv2
import numpy as np

class KITTIDataset:
    def __init__(self, data_root):
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, 'synced+rectified/image_02/data')
        self.calib_dir = os.path.join(data_root, 'calibration')
        
        # Get list of image files
        self.image_files = sorted(os.listdir(self.image_dir))
        print(f"Found {len(self.image_files)} images")
    
    def load_image(self, idx):
        """Load a single image"""
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        return image
    
    def load_calibration(self):
        """Load camera calibration matrix"""
        calib_file = os.path.join(self.calib_dir, 'calib_cam_to_cam.txt')
        with open(calib_file, 'r') as f:
            lines = f.readlines()
        
        # Find P2 (left color camera projection matrix)
        for line in lines:
            if 'P_rect_02:' in line:
                P2 = np.array([float(x) for x in line.split()[1:]])
                P2 = P2.reshape(3, 4)
                # Extract intrinsic matrix K from projection matrix
                K = P2[:3, :3]
                return K
        return None

if __name__ == "__main__":
    # Test the data loader
    dataset = KITTIDataset('data/KITTI/2011_09_26_drive_0027')
    
    # Load first image
    img = dataset.load_image(0)
    print(f"Image shape: {img.shape}")
    
    # Load calibration
    K = dataset.load_calibration()
    print(f"Camera matrix K:\n{K}")
