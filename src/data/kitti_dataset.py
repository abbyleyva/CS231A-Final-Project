"""
KITTI dataset loader for 2D bounding box detection
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
    
    def draw_2d_bbox(self, image, bbox, label="Car", color=(0, 255, 0)):
        """
        Draw 2D bounding box on image
        
        Args:
            image: numpy array of image
            bbox: [x1, y1, x2, y2] coordinates
            label: text label
            color: RGB color tuple
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return image

if __name__ == "__main__":
    # Test with a sample 2D bounding box
    dataset = KITTIDataset('data/KITTI/2011_09_26_drive_0027')
    
    # Load first image
    img = dataset.load_image(0)
    print(f"Image shape: {img.shape}")
    
    # Draw a sample bounding box (you'll replace this with actual detections)
    sample_bbox = [500, 200, 700, 300]  # [x1, y1, x2, y2]
    img_with_bbox = dataset.draw_2d_bbox(img.copy(), sample_bbox, "Sample Car")
    
    # Save the result to see it
    result_path = 'test_2d_bbox.jpg'
    cv2.imwrite(result_path, cv2.cvtColor(img_with_bbox, cv2.COLOR_RGB2BGR))
    print(f"Saved test image with 2D bbox to {result_path}")
