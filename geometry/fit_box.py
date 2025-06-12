import numpy as np

def fit_oriented_bbox(points: np.ndarray):
    """
    Fit an oriented 3D bounding box to a set of 3D points using PCA.

    Args:
        points (np.ndarray): Nx3 array of 3D points.

    Returns:
        dict: {
            'center': (3,), center of the box,
            'axes': (3, 3), rotation matrix with principal axes as rows,
            'dims': (3,), box dimensions along each axis
        }
    """
    if points.shape[0] < 3:
        raise ValueError("Not enough points to fit a bounding box.")

    # Step 1: Center the points
    mean = np.mean(points, axis=0)
    centered = points - mean

    # Step 2: PCA via SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    axes = Vt  # (3, 3), rows = principal axes

    # Step 3: Project points into PCA frame
    proj = centered @ axes.T  # (N, 3)

    # Step 4: Find min/max extents in PCA space
    min_proj = np.min(proj, axis=0)
    max_proj = np.max(proj, axis=0)
    dims = max_proj - min_proj  # box dimensions (L, W, H)

    ### trying to fix the problem posed by inverting depth in depth_model.py
    #dims = np.sort(dims)  # Sort: [smallest, middle, largest]
    #dims = [dims[2], dims[1], dims[0]] 
    print(f"Raw PCA dims: {dims}") 
    

    # Step 5: Compute center in PCA space and transform back
    center_pca = (min_proj + max_proj) / 2.0
    center_world = mean + center_pca @ axes  # back to original coords

    return {
        'center': center_world,
        'axes': axes,
        'dims': dims
    }
