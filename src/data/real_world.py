"""
Real world dataset loader for 2D bounding box detection
"""

import os
import cv2
import numpy as np

class RWDataset:
    def __init__(self, data_root):
        self.data_root = data_root
        # Get list of image files
        self.image_files = [f for f in os.listdir(data_root)]
        print(f"Found {len(self.image_files)} images")
    
    def load_image(self, idx):
        """Load a single image"""
        img_path = os.path.join(self.data_root, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        return image
    
    def load_calibration(self):
        """ Creates default camera calibration matrix"""
        K = np.array([[1000, 0, 640],[0, 1000, 360],[0, 0, 1]])
        return K
    
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
    dataset = RWDataset('data/real_world_test_images')
    
    # Load first image
    img = dataset.load_image(0)
    print(f"Image shape: {img.shape}")
    
    # Draw a sample bounding box (you'll replace this with actual detections)
    sample_bbox = [500, 200, 700, 300]  # [x1, y1, x2, y2]
    img_with_bbox = dataset.draw_2d_bbox(img.copy(), sample_bbox, "Sample Car")
    
    # Save the result to see it
    result_path = 'test_realworld2d_bbox.jpg'
    cv2.imwrite(result_path, cv2.cvtColor(img_with_bbox, cv2.COLOR_RGB2BGR))
    print(f"Saved test image with 2D bbox to {result_path}")
