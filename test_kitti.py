# Test KITTI data loading
import os

# Check if KITTI data exists
kitti_path = 'data/KITTI/2011_09_26_drive_0027'

print('Checking KITTI data...')
print(f'Calibration folder exists: {os.path.exists(kitti_path + "/calibration")}')
print(f'Images folder exists: {os.path.exists(kitti_path + "/synced+rectified")}')
print(f'Tracklets folder exists: {os.path.exists(kitti_path + "/tracklets")}')

# List what's in each folder
print('\nCalibration files:')
print(os.listdir(kitti_path + '/calibration'))
