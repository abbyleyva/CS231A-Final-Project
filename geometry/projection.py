import numpy as np

def project_points(points_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Projects 3D points to 2D using the camera intrinsics matrix.

    Args:
        points_3d (np.ndarray): Nx3 array of 3D points.
        K (np.ndarray): 3x3 camera intrinsic matrix.

    Returns:
        np.ndarray: Nx2 array of 2D projected points (u, v).
    """
    assert points_3d.shape[1] == 3, "3D points must be Nx3"
    assert K.shape == (3, 3), "Camera intrinsics must be 3x3"

    # Convert to homogeneous coordinates (Nx4 → 3xN)
    points_3d_h = points_3d.T  # (3, N)

    # Project using camera intrinsics
    points_2d_h = K @ points_3d_h  # (3, N)

    # Normalize homogeneous coordinates
    u = points_2d_h[0] / points_2d_h[2]
    v = points_2d_h[1] / points_2d_h[2]

    return np.stack([u, v], axis=1)  # (N, 2)
