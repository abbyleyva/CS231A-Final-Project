"""
MultiBin Orientation Estimation
Based on Mousavian et al. approach
"""

import numpy as np
import torch
import torch.nn as nn

class MultiBinLoss(nn.Module):
    def __init__(self, num_bins=4):
        super(MultiBinLoss, self).__init__()
        self.num_bins = num_bins
        self.bin_size = 2 * np.pi / num_bins
        
    def angle_to_bin(self, angle):
        """Convert angle to bin index and residual"""
        # normalize angle to [0, 2π]
        angle = angle % (2 * np.pi)
        
        # find which bin it belongs to
        bin_idx = int(angle / self.bin_size)
        bin_idx = min(bin_idx, self.num_bins - 1)
        
        # calculate residual within the bin
        bin_center = (bin_idx + 0.5) * self.bin_size
        residual = angle - bin_center
        
        return bin_idx, residual
    
# test the basic concept
if __name__ == "__main__":
    multibin = MultiBinLoss(num_bins=4)
    
    # test angles
    test_angles = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
    
    print("Testing MultiBin concept:")
    for angle in test_angles:
        bin_idx, residual = multibin.angle_to_bin(angle)
        print(f"Angle: {angle:.2f} rad ({np.rad2deg(angle):6.1f}°) -> Bin: {bin_idx}, Residual: {residual:.3f}")