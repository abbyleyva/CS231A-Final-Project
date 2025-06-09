import numpy as np

def crop_points_from_2d_box(depth_map: np.ndarray, box: list, K: np.ndarray) -> np.ndarray:
    """
    Crop a depth region corresponding to a 2D bounding box and backproject to 3D points.

    Args:
        depth_map (np.ndarray): HxW array of depth values.
        box (list): 2D bounding box [x1, y1, x2, y2].
        K (np.ndarray): 3x3 camera intrinsics.

    Returns:
        np.ndarray: Nx3 array of 3D points within the box.
    """
    x1, y1, x2, y2 = map(int, box)
    H, W = depth_map.shape

    # Clip box to image dimensions
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    cropped_depth = depth_map[y1:y2, x1:x2]
    if cropped_depth.size == 0:
        return np.empty((0, 3))

    # Generate meshgrid of pixel coordinates
    u_coords, v_coords = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    ones = np.ones_like(u_coords)
    pixel_coords = np.stack([u_coords, v_coords, ones], axis=0).reshape(3, -1)  # (3, N)
    depths = cropped_depth.flatten()  # (N,)

    # Backproject
    K_inv = np.linalg.inv(K)
    cam_coords = K_inv @ pixel_coords
    cam_coords = cam_coords * depths

    return cam_coords.T  # (N, 3)
