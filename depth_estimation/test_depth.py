import sys
from PIL import Image
from depth_model import load_model
from depth_utils import show_depth_map





def main(image_path):
    model = load_model()
    image = Image.open(image_path).convert("RGB")
    depth = model.predict(image)
    show_depth_map(depth, title="Predicted Depth Map")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_depth.py <path_to_image>")
    else:
        main(sys.argv[1])
