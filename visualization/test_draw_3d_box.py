import cv2
import numpy as np
from visualization.draw_3d_box import draw_3d_box
from geometry.projection import project_points  # You'll need this utility

# 1. Load the test image
image_path = '/Users/rcarino/Documents/Stanford University - Academics/2024-2025 Academic Year/2025 Spring Quarter/CS 231A Computer Vision From 3D Perception to 3D Reconstruction and Beyond/Final Project/CS231A-Final-Project/tracklet_cars_frame_0.jpg'
image = cv2.imread(image_path)

# 2. Define intrinsic matrix (KITTI sample)
K = np.array([
    [721.5377, 0, 609.5593],
    [0, 721.5377, 172.8540],
    [0, 0, 1]
])

# 3. Define a dummy 3D box (you'd use PCA in real cases)
# Box centered at (x=10, y=1, z=20), size 2x2x4
center = np.array([10, 1, 20])
dims = np.array([2, 2, 4])  # width, height, depth
dx, dy, dz = dims / 2

# Define corners (8x3)
box_3d = np.array([
    [center[0]-dx, center[1]-dy, center[2]-dz],
    [center[0]+dx, center[1]-dy, center[2]-dz],
    [center[0]+dx, center[1]+dy, center[2]-dz],
    [center[0]-dx, center[1]+dy, center[2]-dz],
    [center[0]-dx, center[1]-dy, center[2]+dz],
    [center[0]+dx, center[1]-dy, center[2]+dz],
    [center[0]+dx, center[1]+dy, center[2]+dz],
    [center[0]-dx, center[1]+dy, center[2]+dz],
])

# 4. Project to 2D
def project_points(points_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    points_3d = points_3d.T  # shape (3, N)
    points_2d = K @ points_3d
    points_2d = points_2d[:2] / points_2d[2:]
    return points_2d.T  # shape (N, 2)

projected_corners = project_points(box_3d, K)

# 5. Draw and show
img_with_box = draw_3d_box(image, projected_corners)
cv2.imshow("3D Box Overlay", img_with_box)
cv2.waitKey(0)
cv2.destroyAllWindows()