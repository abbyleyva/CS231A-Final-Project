import cv2
import numpy as np

def draw_3d_box(image: np.ndarray, projected_corners: np.ndarray, color=(0, 255, 0), thickness=2) -> np.ndarray:
    """
    Draws a 3D bounding box on the image by connecting projected corners.

    Args:
        image (np.ndarray): Original image (HxWx3).
        projected_corners (np.ndarray): (8, 2) array of 2D corner points.
        color (tuple): BGR color for the lines.
        thickness (int): Line thickness.

    Returns:
        np.ndarray: Image with 3D box drawn.
    """
    assert projected_corners.shape == (8, 2), "Expected (8, 2) array for 3D box corners"

    img = image.copy()
    corners = projected_corners.astype(int)

    # Define connections between box corners (0-7 indexing)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom square
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top square
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical lines
    ]

    for i, j in edges:
        pt1 = tuple(corners[i])
        pt2 = tuple(corners[j])
        cv2.line(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

    return img
