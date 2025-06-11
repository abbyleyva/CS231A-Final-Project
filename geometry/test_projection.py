import numpy as np
import cv2
from projection import project_points  # Correct because it's in the same folder
import os

# Resolve full path to image
root = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(root, "..", "tracklet_cars_frame_0.jpg")
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found at: {image_path}")

K = np.array([
    [721.5377, 0, 609.5593],
    [0, 721.5377, 172.854],
    [0, 0, 1]
])

w, h, l = 2.0, 2.0, 4.0
center = np.array([5, 0, 20])
x, y, z = center

corners_3d = np.array([
    [x - w/2, y - h/2, z - l/2],
    [x + w/2, y - h/2, z - l/2],
    [x + w/2, y + h/2, z - l/2],
    [x - w/2, y + h/2, z - l/2],
    [x - w/2, y - h/2, z + l/2],
    [x + w/2, y - h/2, z + l/2],
    [x + w/2, y + h/2, z + l/2],
    [x - w/2, y + h/2, z + l/2]
])

projected_2d = project_points(corners_3d, K).astype(int)

def draw_box(img, corners):
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for start, end in edges:
        pt1 = tuple(corners[start])
        pt2 = tuple(corners[end])
        cv2.line(img, pt1, pt2, (0, 255, 0), 2)

draw_box(image, projected_2d)

cv2.imshow("Projected 3D Box", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
