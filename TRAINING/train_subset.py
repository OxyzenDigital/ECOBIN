import os
import torch
from ultralytics import YOLO

# --- 1. DYNAMIC PATH SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
ecobin_root = os.path.dirname(script_dir)
models_dir = os.path.join(ecobin_root, 'MODELS')
dataset_yaml = os.path.join(script_dir, 'DATASETS', 'batch_01', 'data.yaml')

# Define the model name you WANT to use
model_filename = 'yoloe-v8m-seg-pf.pt' # Your custom model
# model_filename = 'yolo11n.pt'        # Or standard model if you switch

model_path = os.path.join(models_dir, model_filename)

# --- SMART DOWNLOADER (The Fix) ---
# This ensures models land in ECOBIN/MODELS, not random folders
def check_and_download_model(name, dest_path):
    if not os.path.exists(dest_path):
        print(f"📥 Model {name} not found in MODELS folder.")
        print(f"   Downloading to: {dest_path}...")
        
        # We use YOLO to download it to the current dir, then move it
        # (Ultralytics doesn't have a clean 'download_to_path' function for models yet)
        temp_model = YOLO(name) 
        
        # The download happens automatically on init. Now we find where it went.
        # Usually it lands in the script_dir
        default_download = os.path.join(script_dir, name)
        
        if os.path.exists(default_download):
            import shutil
            shutil.move(default_download, dest_path)
            print("✅ Moved model to correct folder.")
        else:
            print("⚠️ Could not move model. It might be in a cache folder.")
    else:
        print(f"✅ Found model in MODELS folder: {name}")

# Check before training
check_and_download_model(model_filename, model_path)

# --- CONFIGURATION ---
start_model = model_path 
run_name = 'batch_01_run'

# --- TRAINING CODE ---
def train_model():
    if not os.path.exists(dataset_yaml):
        print(f"❌ CRITICAL ERROR: Dataset config not found at: {dataset_yaml}")
        return

    # Auto-detect GPU/CPU
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    print(f"🚀 Loading weights from: {start_model}")
    model = YOLO(start_model)

    results = model.train(
        data=dataset_yaml,
        epochs=100,        
        imgsz=640,
        batch=16,
        project=os.path.join(script_dir, 'runs', 'segment'),
        name=run_name,
        device=device,
        patience=15        
    )
    print(f"\n✅ Training Complete!")

if __name__ == '__main__':
    train_model()