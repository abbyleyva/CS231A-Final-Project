"""
Parse KITTI tracklet_labels.xml to extract car positions and create 2D bounding boxes
"""

import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from src.data.kitti_dataset import KITTIDataset

def parse_tracklets_xml(xml_file):
    """
    Parse KITTI tracklet XML file to extract object information
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    tracklets = []
    
    for tracklet in root.findall('item'):
        # Get object type
        objectType = tracklet.find('objectType').text
        
        # Only process cars
        if objectType != 'Car':
            continue
            
        # Get dimensions
        h = float(tracklet.find('h').text)    # height
        w = float(tracklet.find('w').text)    # width  
        l = float(tracklet.find('l').text)    # length
        
        # Get poses for each frame
        poses = tracklet.find('poses')
        
        tracklet_data = {
            'type': objectType,
            'dimensions': [l, w, h],
            'poses': []
        }
        
        # Extract pose for each frame
        for i, pose in enumerate(poses.findall('item')):
            tx = float(pose.find('tx').text)  # x translation
            ty = float(pose.find('ty').text)  # y translation  
            tz = float(pose.find('tz').text)  # z translation
            rx = float(pose.find('rx').text)  # x rotation
            ry = float(pose.find('ry').text)  # y rotation
            rz = float(pose.find('rz').text)  # z rotation
            
            tracklet_data['poses'].append({
                'frame': i,
                'translation': [tx, ty, tz],
                'rotation': [rx, ry, rz]
            })
        
        tracklets.append(tracklet_data)
    
    return tracklets

def project_3d_bbox_to_2d(center, dimensions, rotation_y, camera_matrix):
    """
    Project 3D bounding box to 2D and get min/max bounding box
    """
    l, w, h = dimensions
    
    # 8 corners of 3D bounding box in object coordinate system
    corners_3d = np.array([
        [-l/2, -w/2, -h/2],  # bottom-front-left
        [ l/2, -w/2, -h/2],  # bottom-front-right
        [ l/2,  w/2, -h/2],  # bottom-back-right
        [-l/2,  w/2, -h/2],  # bottom-back-left
        [-l/2, -w/2,  h/2],  # top-front-left
        [ l/2, -w/2,  h/2],  # top-front-right
        [ l/2,  w/2,  h/2],  # top-back-right
        [-l/2,  w/2,  h/2],  # top-back-left
    ])
    
    # Rotation matrix around Y axis
    cos_ry = np.cos(rotation_y)
    sin_ry = np.sin(rotation_y)
    R_y = np.array([
        [ cos_ry, 0, sin_ry],
        [ 0,      1, 0     ],
        [-sin_ry, 0, cos_ry]
    ])
    
    # Apply rotation and translation
    corners_3d_world = (R_y @ corners_3d.T).T + center
    
    # Convert to homogeneous coordinates
    corners_3d_homo = np.column_stack([corners_3d_world, np.ones(8)])
    
    # Project to 2D
    corners_2d_homo = camera_matrix @ corners_3d_homo.T
    corners_2d = corners_2d_homo[:2] / corners_2d_homo[2]
    
    # Get 2D bounding box
    x_min, x_max = corners_2d[0].min(), corners_2d[0].max()
    y_min, y_max = corners_2d[1].min(), corners_2d[1].max()
    
    return [x_min, y_min, x_max, y_max]

# Load dataset and tracklets
dataset = KITTIDataset('data/KITTI/2011_09_26_drive_0027')
tracklets_file = 'data/KITTI/2011_09_26_drive_0027/tracklets/tracklet_labels.xml'

print("Parsing tracklets...")
tracklets = parse_tracklets_xml(tracklets_file)
print(f"Found {len(tracklets)} car tracklets")

# Load camera calibration
K = dataset.load_calibration()
# Add the missing translation column to make it a 3x4 projection matrix
P = np.column_stack([K, np.zeros(3)])

# Load an image (frame 0)
frame_idx = 0
img = dataset.load_image(frame_idx)

# Get cars for this frame
cars_in_frame = []
for tracklet in tracklets:
    if frame_idx < len(tracklet['poses']):
        pose = tracklet['poses'][frame_idx]
        cars_in_frame.append({
            'dimensions': tracklet['dimensions'],
            'center': pose['translation'],
            'rotation_y': pose['rotation'][1]  # Y rotation
        })

print(f"Found {len(cars_in_frame)} cars in frame {frame_idx}")

# Draw the cars
img_with_cars = img.copy()
for i, car in enumerate(cars_in_frame):
    # Project 3D bbox to 2D
    bbox_2d = project_3d_bbox_to_2d(
        car['center'], 
        car['dimensions'], 
        car['rotation_y'], 
        P
    )
    
    # Draw bounding box
    img_with_cars = dataset.draw_2d_bbox(img_with_cars, bbox_2d, f"Car {i+1}", color=(0, 255, 0))
    print(f"Car {i+1} 2D bbox: {bbox_2d}")

# Display result
plt.figure(figsize=(15, 8))
plt.imshow(img_with_cars)
plt.title(f"KITTI Frame {frame_idx} with Real Car Tracklets ({len(cars_in_frame)} cars)")
plt.axis('off')
plt.show()

# Save result
result_path = f'tracklet_cars_frame_{frame_idx}.jpg'
import cv2
cv2.imwrite(result_path, cv2.cvtColor(img_with_cars, cv2.COLOR_RGB2BGR))
print(f"Saved result to {result_path}")
