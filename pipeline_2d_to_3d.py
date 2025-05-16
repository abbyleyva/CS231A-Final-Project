"""
3D Bounding Box Estimation Pipeline
Following Mousavian et al. approach: 2D detection -> 3D estimation
"""

from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.data.kitti_dataset import KITTIDataset

class BoundingBox3DEstimator:
    def __init__(self, camera_matrix):
        self.K = camera_matrix
        
    def estimate_3d_from_2d(self, bbox_2d, object_type="Car"):
        """
        Estimate 3D bounding box from 2D detection
        Following geometric constraints from Mousavian et al.
        """
        # extract 2D box dimensions
        x1, y1, x2, y2 = bbox_2d
        width_2d = x2 - x1
        height_2d = y2 - y1
        center_2d = [(x1 + x2) / 2, (y1 + y2) / 2]
        
        # prior knowledge of typical car dimensions (in meters)
        if object_type == "Car":
            typical_dims = {'length': 4.0, 'width': 1.8, 'height': 1.5}
        elif object_type == "Van":
            typical_dims = {'length': 5.0, 'width': 2.0, 'height': 2.5}
        else:
            typical_dims = {'length': 4.0, 'width': 1.8, 'height': 1.5}
        
        # estimate depth using geometric constraints
        # this is simplified - real implementation would use the full constraint system
        focal_length = self.K[0, 0]  # Assuming fx ≈ fy
        estimated_depth = (typical_dims['length'] * focal_length) / width_2d
        
        # estimate 3D center
        # back-project 2D center to 3D using estimated depth
        x_3d = (center_2d[0] - self.K[0, 2]) * estimated_depth / self.K[0, 0]
        y_3d = (center_2d[1] - self.K[1, 2]) * estimated_depth / self.K[1, 1]
        z_3d = estimated_depth
        
        center_3d = [x_3d, y_3d, z_3d]
        
        # for now, assume zero rotation (can be improved with MultiBin)
        rotation_y = 0.0
        
        dimensions_3d = [typical_dims['length'], typical_dims['width'], typical_dims['height']]
        
        return {
            'center_3d': center_3d,
            'dimensions_3d': dimensions_3d,
            'rotation_y': rotation_y
        }
    
    def project_3d_to_2d_corners(self, bbox_3d):
        """
        Project 3D bounding box corners back to 2D for visualization
        """
        center = np.array(bbox_3d['center_3d'])
        l, w, h = bbox_3d['dimensions_3d']
        ry = bbox_3d['rotation_y']
        
        # 8 corners of 3D box
        corners_3d = np.array([
            [-l/2, -w/2, -h/2], [l/2, -w/2, -h/2],
            [l/2, w/2, -h/2], [-l/2, w/2, -h/2],
            [-l/2, -w/2, h/2], [l/2, -w/2, h/2],
            [l/2, w/2, h/2], [-l/2, w/2, h/2]
        ])
        
        # rotation
        R_y = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        corners_3d_rotated = (R_y @ corners_3d.T).T + center
        
        # project to 2D
        corners_2d = []
        for corner in corners_3d_rotated:
            # convert to homogeneous coordinates
            corner_homo = np.append(corner, 1)
            # project
            corner_2d_homo = self.K @ corner_homo[:3]
            corner_2d = corner_2d_homo[:2] / corner_2d_homo[2]
            corners_2d.append(corner_2d)
        
        return np.array(corners_2d)

def demo_pipeline():
    """
    Demonstrate the complete 2D to 3D pipeline
    """
    dataset = KITTIDataset('data/KITTI/2011_09_26_drive_0027')
    camera_matrix = dataset.load_calibration()
    estimator = BoundingBox3DEstimator(camera_matrix)
    
    # YOLO model
    model = YOLO("yolov8m.pt")
    
    # test 
    img_path = 'tracklet_cars_frame_0.jpg'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_path)
    
    detections_3d = []
    
    for box in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = box[:6]
        
        # filter for vehicles (cars, trucks, buses in YOLO)
        if int(cls) in [2, 7, 5]:  # car, truck, bus in COCO classes
            bbox_2d = [float(x1), float(y1), float(x2), float(y2)]
            
            # estimate 3D box
            bbox_3d = estimator.estimate_3d_from_2d(bbox_2d)
            
            # project back to 2D for visualization
            corners_2d = estimator.project_3d_to_2d_corners(bbox_3d)
            
            detections_3d.append({
                'bbox_2d': bbox_2d,
                'bbox_3d': bbox_3d,
                'corners_2d': corners_2d,
                'confidence': float(conf)
            })
    
    # visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # original with 2D boxes
    ax1.imshow(img)
    for det in detections_3d:
        x1, y1, x2, y2 = det['bbox_2d']
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
        ax1.add_patch(rect)
    ax1.set_title(f'2D Detections ({len(detections_3d)} vehicles)')
    ax1.axis('off')
    
    # with projected 3D boxes
    ax2.imshow(img)
    for det in detections_3d:
        corners = det['corners_2d']
        # draw 3D wireframe with different colors
        # front face (green)
        bottom_edges = [(0,1), (1,2), (2,3), (3,0)]
        for i, j in bottom_edges:
            ax2.plot([corners[i,0], corners[j,0]], [corners[i,1], corners[j,1]], 'g-', linewidth=2)
        
        # back face (blue)
        top_edges = [(4,5), (5,6), (6,7), (7,4)]
        for i, j in top_edges:
            ax2.plot([corners[i,0], corners[j,0]], [corners[i,1], corners[j,1]], 'b-', linewidth=2)
        
        # horizontal edges (red)
        vertical_edges = [(0,4), (1,5), (2,6), (3,7)]
        for i, j in vertical_edges:
            ax2.plot([corners[i,0], corners[j,0]], [corners[i,1], corners[j,1]], 'r-', linewidth=2)
    
    ax2.set_title(f'Estimated 3D Boxes ({len(detections_3d)} vehicles)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print(f"Processed {len(detections_3d)} vehicle detections")
    for i, det in enumerate(detections_3d):
        center = det['bbox_3d']['center_3d']
        dims = det['bbox_3d']['dimensions_3d']
        print(f"Vehicle {i+1}: Center=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}), "
              f"Dims=({dims[0]:.1f}x{dims[1]:.1f}x{dims[2]:.1f}), Conf={det['confidence']:.2f}")

if __name__ == "__main__":
    demo_pipeline()
    print("\nTechnical Approach Summary:")
    print("1. 2D Detection: YOLO for vehicle detection")
    print("2. Geometric Constraints: Use camera calibration and typical car dimensions")
    print("3. 3D Estimation: Back-project 2D boxes to 3D using depth estimation")
    print("4. Validation: Project 3D boxes back to 2D and compare")
    print("\nNext steps: Implement MultiBin orientation estimation and geometric constraint optimization")
