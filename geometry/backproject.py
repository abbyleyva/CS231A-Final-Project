import numpy as np

def backproject(depth_map: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Backproject depth map into 3D space using camera intrinsics.
    
    Args:
        depth_map (np.ndarray): HxW array of depth values (in meters).
        K (np.ndarray): 3x3 camera intrinsic matrix.

    Returns:
        np.ndarray: Nx3 array of 3D points in camera coordinates.
    """
    assert depth_map.ndim == 2, "Depth map must be 2D"
    assert K.shape == (3, 3), "Camera intrinsics must be 3x3"

    H, W = depth_map.shape
    u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(u_coords)

    # Stack into shape (3, N)
    pixel_coords = np.stack([u_coords, v_coords, ones], axis=0).reshape(3, -1)  # (3, N)
    depths = depth_map.flatten()  # (N,)

    K_inv = np.linalg.inv(K)
    cam_coords = K_inv @ pixel_coords  # (3, N)
    cam_coords = cam_coords * depths  # scale each ray by corresponding depth

    return cam_coords.T  # (N, 3)
