import os
import json
import cv2  # We need this to get image height/width for denormalization

# --- SETUP PATHS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(script_dir, 'IMAGES')
classes_file = os.path.join(img_dir, 'classes.txt')

# --- LOAD CLASSES ---
if not os.path.exists(classes_file):
    print(f"❌ Critical Error: classes.txt not found at {classes_file}")
    exit()

with open(classes_file, 'r') as f:
    # Read lines and strip whitespace
    class_names = [line.strip() for line in f.readlines() if line.strip()]

print(f"ℹ️  Loaded {len(class_names)} classes: {class_names}")

# --- PROCESSING LOOP ---
files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"🚀 Processing {len(files)} images...")

for file_name in files:
    image_path = os.path.join(img_dir, file_name)
    txt_name = os.path.splitext(file_name)[0] + ".txt"
    txt_path = os.path.join(img_dir, txt_name)
    json_path = os.path.join(img_dir, os.path.splitext(file_name)[0] + ".json")

    # Skip if no corresponding text label file
    if not os.path.exists(txt_path):
        continue

    # 1. Load Image to get dimensions (YOLO needs this to un-normalize)
    img = cv2.imread(image_path)
    if img is None:
        continue
    height, width, _ = img.shape

    # 2. Read YOLO text file
    shapes = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))

        # Get Class Name
        if class_id < len(class_names):
            label_name = class_names[class_id]
        else:
            label_name = "unknown"

        # Un-normalize coordinates (0-1 -> 0-1920 pixels)
        points = []
        for i in range(0, len(coords), 2):
            px = coords[i] * width
            py = coords[i+1] * height
            points.append([px, py])

        # Create JSON shape object
        shape = {
            "label": label_name,
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        shapes.append(shape)

    # 3. Save as AnyLabeling/LabelMe JSON
    data = {
        "version": "0.3.3",  # Generic version
        "flags": {},
        "shapes": shapes,
        "imagePath": file_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

print("✅ Conversion Complete! You can now open AnyLabeling.")