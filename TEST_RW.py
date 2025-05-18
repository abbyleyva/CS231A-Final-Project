"""
Test 3D bounding box estimation on real-world images (non-KITTI)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pipeline_2d_to_3d import BoundingBox3DEstimator
from src.data.real_world import RWDataset

def test_real_world_images():
    """
    Test the 3D bounding box estimation pipeline on real-world images
    """
    # Create results directory & load dataset
    os.makedirs('rw_results', exist_ok=True)
    dataset = RWDataset('data/real_world_test_images')
    
    # Process each image
    model = YOLO("yolov8m.pt")
    for i in range(len(dataset.image_files)):
        img = dataset.load_image(i)
        ### calculating new calibration matrix for each image
        ### because dimensions may differ

        def estimate_focal_length(img_width):
            return img_width * 1.2

        ## K is the default at first
        K = dataset.load_calibration()
        height, width = img.shape[:2]
        mid_width = width / 2
        mid_height = height / 2
        K[0, 0] = estimate_focal_length(width)  # fx
        K[1, 1] = estimate_focal_length(width)  # fy
        K[0, 2] = mid_width  # cx
        K[1, 2] = mid_height
        estimator = BoundingBox3DEstimator(K)
        results = model(img)
        
        # Process detections for each image
        detections_3d = []
        car = 2
        bus = 5
        truck = 7
        vehicle_classes = [car, bus, truck] 
        for box in results[0].boxes.data:
            x1, y1, x2, y2, confidence, cls = box[:6]
            cls_id = int(cls)
            
            # Vehicles
            if cls_id in vehicle_classes:
                class_names = {2: "Car", 5: "Bus", 7: "Truck"}
                class_name = class_names.get(cls_id, "Vehicle")
                bbox_2d = [float(x1), float(y1), float(x2), float(y2)]
                # Skip very small boxes (likely to be unreliable for 3D estimation)
                width = bbox_2d[2] - bbox_2d[0]
                height = bbox_2d[3] - bbox_2d[1]
                if width < 50 or height < 50:
                    continue
                # Else, we can Estimate 3D box
                try:
                    bbox_3d = estimator.estimate_3d_from_2d(bbox_2d, object_type=class_name)
                    # And Project back to 2D for visualization
                    corners_2d = estimator.project_3d_to_2d_corners(bbox_3d)
                    detections_3d.append({'bbox_2d': bbox_2d, 'bbox_3d': bbox_3d, 'corners_2d': corners_2d,
                        'confidence': float(confidence),'class': class_name})
                except Exception as e:
                    print(f"Error processing detection: {e}")

        # Visualize results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

        ax1.set_xlim([0, img.shape[1]])
        ax1.set_ylim([img.shape[0], 0])  
        ax2.set_xlim([0, img.shape[1]])
        ax2.set_ylim([img.shape[0], 0])

        # Original with 2D boxes
        ax1.imshow(img)
        for det in detections_3d:
            x1, y1, x2, y2 = det['bbox_2d']
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
            ax1.add_patch(rect)
            ax1.text(x1, y1-5, f"{det['class']} ({det['confidence']:.2f})", 
                    color='white', backgroundcolor='red', fontsize=8)
        ax1.set_title(f'2D Detections ({len(detections_3d)} vehicles)')
        ax1.axis('off')
        
        # With 3D boxes
        ax2.imshow(img)
        for det in detections_3d:
            corners = det['corners_2d']
            # Draw 3D box wireframe
            # Bottom face 
            for j, k in [(0,1), (1,2), (2,3), (3,0)]:
                ax2.plot([corners[j,0], corners[k,0]], [corners[j,1], corners[k,1]], 'g-', linewidth=2)
            # Top face 
            for j, k in [(4,5), (5,6), (6,7), (7,4)]:
                ax2.plot([corners[j,0], corners[k,0]], [corners[j,1], corners[k,1]], 'b-', linewidth=2)
            # Connecting lines 
            for j, k in [(0,4), (1,5), (2,6), (3,7)]:
                ax2.plot([corners[j,0], corners[k,0]], [corners[j,1], corners[k,1]], 'r-', linewidth=2)
        ax2.set_title(f'Estimated 3D Boxes ({len(detections_3d)} vehicles)')
        ax2.axis('off')
        plt.subplots_adjust(wspace=0.05)
        plt.tight_layout()

        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        ax2.set_position([pos2.x0, pos1.y0, pos1.width, pos1.height])
        
        # Save result and print info
        result_filename = f"rw_results_{dataset.image_files[i].replace('.jpg', '').replace('.png', '')}.jpg"
        result_path = os.path.join('rw_results', result_filename)
        plt.savefig(result_path)
        plt.close()
        print(f"Processed image {i}: Found {len(detections_3d)} vehicles")
        print(f"Result saved to: {result_path}")
        for j, det in enumerate(detections_3d):
            center = det['bbox_3d']['center_3d']
            dimensions = det['bbox_3d']['dimensions_3d']
            rotation = det['bbox_3d']['rotation_y']
            print(f"  {det['class']} {j+1}: "
                  f"Center=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}m), "
                  f"Dims=({dimensions[0]:.1f}x{dimensions[1]:.1f}x{dimensions[2]:.1f}m), "
                  f"Rot={np.rad2deg(rotation):.1f}°, Conf={det['confidence']:.2f}")

if __name__ == "__main__":
    print("Estimating 3D bounding box on real-world images...")
    test_real_world_images()
    print("\nProcessing complete!")