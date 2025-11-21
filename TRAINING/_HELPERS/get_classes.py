import os
from ultralytics import YOLO

# 1. SETUP PATHS
script_dir = os.path.dirname(os.path.abspath(__file__))
ecobin_root = os.path.dirname(script_dir)
model_path = os.path.join(ecobin_root, 'MODELS', 'yoloe-v8m-seg-pf.pt')
output_path = os.path.join(script_dir, 'IMAGES', 'classes.txt')

# 2. LOAD MODEL
print(f"Loading model details from: {model_path}")
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 3. EXTRACT CLASS NAMES
# The model.names attribute is a dictionary {0: 'person', 1: 'bicycle', ...}
class_dict = model.names
print(f"Found {len(class_dict)} classes in the model.")

# 4. WRITE TO classes.txt
# We must sort by ID (0, 1, 2...) to ensure the line numbers match the class IDs
sorted_ids = sorted(class_dict.keys())

with open(output_path, 'w') as f:
    for class_id in sorted_ids:
        name = class_dict[class_id]
        f.write(f"{name}\n")
        print(f"ID {class_id}: {name}")

print(f"\n✅ Success! Saved to: {output_path}")
print("You can now open AnyLabeling, and it will automatically read these names.")