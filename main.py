import time
import sys

# --- STARTUP ---
print("[-] Initializing EcoBin System...")
start_time = time.time()

import cv2
import json
import serial
import collections
import numpy as np
from ultralytics import YOLO
import os

# --- LOAD SETTINGS ---
try:
    with open('settings.json', 'r') as f:
        settings = json.load(f)
    with open(settings['paths']['class_map'], 'r') as f:
        class_map = json.load(f)
except Exception as e:
    print(f"[!] Settings Error: {e}")
    sys.exit()

# --- SYSTEM STATES ---
STATE_IDLE = "IDLE"
STATE_RESET_BG = "RESET_BG"
STATE_WAITING_OBJECT = "WAITING"
STATE_SETTLING = "SETTLING"
STATE_DETECTING = "DECIDING"
STATE_BUSY = "SORTING"

current_state = STATE_IDLE
last_state_change = time.time()
settle_start_time = 0

# --- LOGGING ---
serial_log = collections.deque(maxlen=10)

# --- HARDWARE: CAMERA ---
print(f"[-] Opening Camera Index {settings['camera']['index']}...")
cap = cv2.VideoCapture(settings['camera']['index'], cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['camera']['width'])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['camera']['height'])

if not cap.isOpened():
    print("[!] Camera failed. Retrying Index 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# --- HARDWARE: ARDUINO ---
print("[-] Connecting to Arduino...")
try:
    arduino = serial.Serial(settings['serial']['port'], settings['serial']['baud_rate'], timeout=0.05)
    time.sleep(2)
    arduino_status = "ONLINE"
    # Clear buffers on connect so old commands don't execute
    arduino.reset_input_buffer()
    arduino.reset_output_buffer()
except Exception as e:
    arduino = None
    arduino_status = "OFFLINE"
    print(f"[!] Arduino Warning: {e}")

# --- AI MODELS ---
print("[-] Loading Models...")
viz_threshold = 0.4 

# Conditionally load Standard Model
model_std = None
if settings['yolo'].get('use_standard_model', True):
    if os.path.exists(settings['yolo']['standard_model']):
        print("... Loading Standard Model")
        model_std = YOLO(settings['yolo']['standard_model'])
    else:
        print(f"[!] Warning: Standard model enabled but not found at {settings['yolo']['standard_model']}")

# Conditionally load Custom Model
model_custom = None
if os.path.exists(settings['yolo']['custom_model']):
    print("... Loading Custom Model")
    model_custom = YOLO(settings['yolo']['custom_model'])
else:
    print("[!] Info: Custom model not found, it will not be used.")

# Exit if no models could be loaded.
if not model_std and not model_custom:
    print("[!] CRITICAL ERROR: No models were loaded. Check 'settings.json' paths and flags.")
    sys.exit()

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

print(f"[-] System Ready! [{time.time() - start_time:.2f}s]")

# --- FUNCTIONS ---
def log_message(msg, type="INFO"):
    timestamp = time.strftime("%H:%M:%S")
    prefix = ">>" if type == "TX" else "<<" if type == "RX" else "--"
    serial_log.append(f"[{timestamp}] {prefix} {msg}")

def send_to_arduino(signal):
   # [cite: 25] "Understand the gravity of the sequence... when UNO is ready"
    # STRICT CHECK: Do not send if we think we are busy, unless it's a reset
    if current_state == STATE_BUSY and signal != 'X':
        log_message("Blocked: System Busy", "INFO")
        return False

    if arduino:
        try:
            arduino.write(signal.encode())
            log_message(f"SENT: {signal}", "TX")
            return True
        except:
            pass
    return False

def run_live_detection(frame):
    """
    Run enabled models.
    Compare confidence scores to find the single BEST match and visualize only that one.
    """
    all_detections = []

    # Helper to gather all valid detections
    def gather_detections(results):
        if results is None:
            return
        
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = results.names[cls_id]
            
            if name in class_map:
                all_detections.append({
                    "box": list(map(int, box.xyxy[0])),
                    "name": name,
                    "conf": conf
                })

    # 1. Run Standard Model (if loaded)
    if model_std:
        results_std = model_std(frame, verbose=False, conf=viz_threshold)[0]
        gather_detections(results_std)
    
    # 2. Run Custom Model (if loaded)
    if model_custom:
        results_cust = model_custom(frame, verbose=False, conf=viz_threshold)[0]
        gather_detections(results_cust)

    viz_frame = frame.copy()
    best_name = None
    best_conf = 0

    # Find the best detection from the aggregated list
    if all_detections:
        best_detection = max(all_detections, key=lambda x: x['conf'])
        best_name = best_detection['name']
        best_conf = best_detection['conf']

        # Draw only the best one
        x1, y1, x2, y2 = best_detection['box']
        name = best_detection['name']
        conf = best_detection['conf']

        classification = class_map[name]
        color = (0, 0, 255) if classification == "T" else (0, 255, 0)
        label = f"{name} [{classification}] {int(conf*100)}%"
        
        cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(viz_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return viz_frame, best_name, best_conf

def draw_dashboard(camera_frame):
    h, w, _ = camera_frame.shape
    sidebar_w = 350
    canvas = np.zeros((h, w + sidebar_w, 3), dtype=np.uint8)
    
    canvas[0:h, 0:w] = camera_frame
    cv2.rectangle(canvas, (w, 0), (w + sidebar_w, h), (30, 30, 30), -1)
    
    # Header
    cv2.putText(canvas, "ECOBIN STATUS", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
    # State Colors
    st_c = (0, 255, 0)
    if current_state == STATE_BUSY: st_c = (0, 255, 255)
    if current_state == STATE_DETECTING: st_c = (255, 0, 255)
    if current_state == STATE_WAITING_OBJECT: st_c = (255, 255, 255)
    
    cv2.putText(canvas, current_state, (w + 20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, st_c, 2)
    
    # UNO Status
    conn_color = (0, 255, 0) if arduino_status == "ONLINE" else (0, 0, 255)
    cv2.putText(canvas, f"UNO: {arduino_status}", (w + 20, 120), cv2.FONT_HERSHEY_PLAIN, 1.2, conn_color, 1)
    
    cv2.line(canvas, (w + 20, 140), (w + sidebar_w - 20, 140), (80, 80, 80), 1)
    
    # Logs
    y_start = 170
    for i, line in enumerate(serial_log):
        c = (180, 180, 180)
        if ">>" in line: c = (0, 255, 0)
        if "<<" in line: c = (0, 200, 255)
        if "MATCH" in line: c = (255, 0, 255)
        cv2.putText(canvas, line, (w + 10, y_start + (i * 25)), cv2.FONT_HERSHEY_PLAIN, 0.9, c, 1)

    # Controls Footer
    # Dynamic Text based on Busy State
    if current_state == STATE_BUSY:
        cv2.putText(canvas, "KEYS LOCKED (BUSY)", (w + 20, h - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
    else:
        cv2.putText(canvas, "TEST: [1] Trash  [2] Recycle", (w + 20, h - 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 255), 1)
        
    cv2.putText(canvas, "CTRL: [S] Start  [R] Reset  [Q] Quit", (w + 20, h - 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (100, 100, 100), 1)
    
    return canvas

# --- MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret: break

    # --- 1. READ ARDUINO ---
    # Check input buffer to see if UNO is done
    if arduino and arduino.in_waiting > 0:
        try:
            raw_bytes = arduino.read(arduino.in_waiting)
            raw_str = raw_bytes.decode('utf-8', errors='ignore').strip()
            
            if raw_str:
                lines = raw_str.split('\n')
                for line in lines:
                    clean = line.strip()
                    if not clean: continue
                    log_message(f"{clean}", "RX")
                    
                    # Unlock if UNO says READY/A
                    if "READY" in clean or clean.startswith(settings['codes']['uno_ready']):
                         if current_state == STATE_BUSY:
                            current_state = STATE_RESET_BG
                            last_state_change = time.time()
                            # Flush buffers to prevent stacked commands
                            arduino.reset_input_buffer()
                    
                    # Lock if UNO says BUSY/B
                    if "BUSY" in clean or clean.startswith(settings['codes']['uno_busy']):
                        current_state = STATE_BUSY
        except: pass

    # --- 2. VISION PROCESSING ---
    mask = bg_subtractor.apply(frame)
    motion = cv2.countNonZero(mask)
    viz_frame, live_best_name, live_best_conf = run_live_detection(frame)
    
    # --- 3. STATE MACHINE ---
    if current_state == STATE_IDLE:
        pass

    elif current_state == STATE_RESET_BG:
        if time.time() - last_state_change > 1.5:
            current_state = STATE_WAITING_OBJECT
            log_message("System Ready.", "INFO")

    elif current_state == STATE_WAITING_OBJECT:
        if motion > 2000:
            current_state = STATE_SETTLING
            settle_start_time = time.time()

    elif current_state == STATE_SETTLING:
        elapsed = time.time() - settle_start_time
        cv2.putText(viz_frame, f"Settling: {elapsed:.1f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        if motion > 1000:
            settle_start_time = time.time() 
        elif elapsed > settings['timing']['settle_delay_s']:
            current_state = STATE_DETECTING

    elif current_state == STATE_DETECTING:
        if live_best_name and live_best_name in class_map:
            category = class_map[live_best_name]
            log_message(f"MATCH: {live_best_name} -> {category}", "INFO")
            sig = settings['codes']['trash'] if category == "T" else settings['codes']['recyclable']
            
            # Check if successfully sent
            if send_to_arduino(sig):
                current_state = STATE_BUSY
            else:
                current_state = STATE_WAITING_OBJECT # Retry
        else:
            if live_best_name:
                log_message(f"Unknown: {live_best_name}", "INFO")
            current_state = STATE_WAITING_OBJECT

    elif current_state == STATE_BUSY:
        cv2.putText(viz_frame, "SORTING IN PROGRESS...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # --- 4. RENDER ---
    dashboard = draw_dashboard(viz_frame)
    cv2.imshow("Oxyzen EcoBin", dashboard)

    # --- 5. CONTROLS (Strict Logic) ---
    key = cv2.waitKey(1) & 0xFF
    
    # [A] ALWAYS ACTIVE: QUIT
   # [cite: 23] "Q to quit the active session at anytime"
    if key == ord('q'): 
        print("[-] User requested Quit.")
        if arduino:
            arduino.write(b'X') # Reset UNO
            time.sleep(0.2)
        break

    # [B] BUSY LOCKOUT
   # [cite: 26] "This is key... when UNO is ready for taking new orders"
    if current_state == STATE_BUSY:
        # Ignore all other inputs if busy
        pass 
    
    # [C] READY STATE CONTROLS
    else:
        if key == ord('s'): 
            current_state = STATE_RESET_BG
            last_state_change = time.time()
        
        if key == ord('r'):
            current_state = STATE_RESET_BG
            last_state_change = time.time()
        
        # Manual Triggers
        if key == ord('1'): 
            send_to_arduino(settings['codes']['trash'])
            current_state = STATE_BUSY # Manually force state lock
            
        if key == ord('2'): 
            send_to_arduino(settings['codes']['recyclable'])
            current_state = STATE_BUSY # Manually force state lock

cap.release()
if arduino: arduino.close()
cv2.destroyAllWindows()