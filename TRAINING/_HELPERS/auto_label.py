import os
from ultralytics import YOLO

# --- 1. SETUP PATHS ---
# Get the folder where this script is running (ECOBIN/TRAINING)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent folder (ECOBIN)
ecobin_root = os.path.dirname(script_dir)

# PATH TO MODEL: Go up to ECOBIN, then down to MODELS
model_path = os.path.join(ecobin_root, 'MODELS', 'yoloe-v8m-seg-pf.pt')

# PATH TO IMAGES: Inside TRAINING folder
images_path = os.path.join(script_dir, 'IMAGES')

# PATH TO OUTPUT: We will create this inside TRAINING
output_dir = os.path.join(script_dir, 'AUTO_LABELS')

# --- 2. VERIFY PATHS (Debugging) ---
print(f"Looking for model at:   {model_path}")
print(f"Looking for images at:  {images_path}")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ CRITICAL ERROR: Could not find model file at {model_path}")
if not os.path.exists(images_path):
    raise FileNotFoundError(f"❌ CRITICAL ERROR: Could not find images folder at {images_path}")

# --- 3. LOAD MODEL ---
# Explicitly load the local file so it doesn't try to download from the internet
print("✅ Paths verified. Loading model...")
model = YOLO(model_path)

# --- 4. RUN AUTO-LABELING ---
print("🚀 Starting inference...")
results = model.predict(
    source=images_path,
    
    # SAVE OPTIONS
    save=True,       # Save jpgs with boxes drawn (for visual check)
    save_txt=True,   # Save txt files with polygon coordinates (for training)
    
    # OUTPUT FOLDER
    project=output_dir, # Saves to ECOBIN/TRAINING/AUTO_LABELS
    name='batch_1',     # Subfolder for this specific run
    exist_ok=True,      # Don't crash if folder exists
    
    # DETECTION SETTINGS
    conf=0.4,        # Confidence Threshold (Adjust: Higher = fewer, strictly accurate labels)
    iou=0.5          # Overlap Threshold
)

print(f"\n🎉 DONE! Results saved to:\n{os.path.join(output_dir, 'batch_1')}")