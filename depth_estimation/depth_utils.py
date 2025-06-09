import matplotlib.pyplot as plt
import numpy as np

def show_depth_map(depth_map: np.ndarray, title="Depth Map"):
    plt.imshow(depth_map, cmap='plasma')
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    plt.show()