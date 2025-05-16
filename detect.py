from ultralytics import YOLO

# Load a pretrained model: We'll use the medium model
model = YOLO("yolov8m.pt") 

# Run detection: For the File, We could change this to the directory and
# iterate through the respective images
# Change the File Path for an image
#results = model(FILEPATH)

#Here is an example
# results = model('/Users/rcarino/Documents/Stanford University - Academics/2024-2025 Academic Year/2025 Spring Quarter/CS 231A Computer Vision From 3D Perception to 3D Reconstruction and Beyond/Final Project/CS231A-Final-Project/tracklet_cars_frame_0.jpg')
results = results = model('tracklet_cars_frame_0.jpg')

# Show image with detections
results[0].show()

# Print detections to console: For now it's just one image
for box in results[0].boxes.data:
    top_left_x, top_left_y, bottom_right_x, bottom_right_y, conf_score, obj_class = box[:6]
    print(f"Detected class {int(obj_class)} with confidence {conf_score:.2f} at [{top_left_x}, {top_left_y}, {bottom_right_x}, {bottom_right_y}]")
