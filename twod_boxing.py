"""
2D bounding box detection for vehicles in KITTI dataset
Using computer vision techniques without ground truth
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.data.kitti_dataset import KITTIDataset

def detect_vehicles_2d(image):
    """
    Detect vehicles using image processing techniques
    Returns list of 2D bounding boxes
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive threshold to find regions of interest
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
    
    # Focus on road area (bottom 60% of image)
    height, width = adaptive_thresh.shape
    road_mask = np.zeros_like(adaptive_thresh)
    road_mask[int(height*0.4):, :] = 255
    adaptive_thresh = cv2.bitwise_and(adaptive_thresh, road_mask)
    
    # Morphological operations to connect nearby pixels
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours for vehicle-like properties
    vehicle_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by size and aspect ratio
        area = w * h
        aspect_ratio = w / h
        
        # Vehicle criteria
        if (area > 800 and area < 20000 and 
            aspect_ratio > 1.2 and aspect_ratio < 4.0 and
            h > 20 and w > 30):
            
            # Additional check: vehicles should be dark objects
            roi = gray[y:y+h, x:x+w]
            mean_intensity = np.mean(roi)
            
            # Vehicles tend to be darker than background
            if mean_intensity < 120:
                vehicle_boxes.append([x, y, x+w, y+h])
    
    return vehicle_boxes

def test_detection_on_multiple_frames():
    """
    Test vehicle detection on multiple frames
    """
    dataset = KITTIDataset('data/KITTI/2011_09_26_drive_0027')
    
    # Test on several frames
    test_frames = [50, 100, 150]
    
    for frame_idx in test_frames:
        img = dataset.load_image(frame_idx)
        
        # Detect vehicles
        detected_boxes = detect_vehicles_2d(img)
        
        # Draw detections
        img_with_detections = img.copy()
        for i, bbox in enumerate(detected_boxes):
            img_with_detections = dataset.draw_2d_bbox(
                img_with_detections, bbox, f"Vehicle {i+1}", color=(255, 0, 0)
            )
        
        # Display results
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Original Frame {frame_idx}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_with_detections)
        plt.title(f"Detected Vehicles ({len(detected_boxes)} found)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Frame {frame_idx}: {len(detected_boxes)} vehicles detected")

if __name__ == "__main__":
    test_detection_on_multiple_frames()
    print("\n2D vehicle detection complete!")
    print("This provides bounding boxes for 3D estimation pipeline.")
