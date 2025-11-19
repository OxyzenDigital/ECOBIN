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

# --- DIAGNOSTICS VARIABLES ---
last_tx = "None"   # Last command sent to UNO
last_rx = "None"   # Last message received from UNO
rx_timer = 0       # To flash the RX light
serial_log = collections.deque(maxlen=8)

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
    # [cite: 13] Serial.begin(9600)
    arduino = serial.Serial(settings['serial']['port'], settings['serial']['baud_rate'], timeout=0.05)
    time.sleep(2) # Wait for UNO reboot
    arduino_status = "ONLINE"
    
    # Clear any startup junk
    arduino.reset_input_buffer()
    
except Exception as e:
    arduino = None
    arduino_status = "OFFLINE"
    print(f"[!] Arduino Warning: {e}")

# --- AI MODELS ---
print("[-] Loading Models...")
viz_threshold = 0.4 
model_std = YOLO(settings['yolo']['standard_model'])

if os.path.exists(settings['yolo']['custom_model']):
    model_custom = YOLO(settings['yolo']['custom_model'])
    has_custom_model = True
else:
    model_custom = model_std
    has_custom_model = False

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

print(f"[-] System Ready! [{time.time() - start_time:.2f}s]")

# --- FUNCTIONS ---
def log_message(msg, type="INFO"):
    global last_tx, last_rx, rx_timer
    timestamp = time.strftime("%H:%M:%S")
    
    if type == "TX": 
        last_tx = msg
        prefix = ">>"
    elif type == "RX": 
        last_rx = msg
        rx_timer = 10 # Frames to keep RX light bright
        prefix = "<<"
    else: 
        prefix = "--"
        
    serial_log.append(f"[{timestamp}] {prefix} {msg}")

def send_to_arduino(signal):
    if arduino:
        try:
            # [cite: 10, 11] UNO expects char 'T' or 'R'
            arduino.write(signal.encode())
            log_message(f"SENT: {signal}", "TX")
            return True
        except Exception as e:
            log_message(f"Write Err: {e}", "INFO")
    else:
        log_message(f"SIMULATED: {signal}", "TX")
    return False

def run_live_detection(frame):
    results = model_std(frame, verbose=False, conf=viz_threshold)[0]
    best_conf = 0
    best_name = None
    viz_frame = frame.copy()

    for box in results.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        name = results.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        if name in class_map:
            classification = class_map[name]
            color = (0, 0, 255) if classification == "T" else (0, 255, 0)
            label = f"{name} [{classification}]"
            if conf > best_conf:
                best_conf = conf
                best_name = name
        else:
            color = (100, 100, 100) 
            label = f"{name} (?)"

        cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(viz_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return viz_frame, best_name, best_conf

def draw_dashboard(camera_frame):
    global rx_timer
    h, w, _ = camera_frame.shape
    sidebar_w = 350
    canvas = np.zeros((h, w + sidebar_w, 3), dtype=np.uint8)
    
    canvas[0:h, 0:w] = camera_frame
    cv2.rectangle(canvas, (w, 0), (w + sidebar_w, h), (30, 30, 30), -1)
    
    # --- HEADER & CONNECTION LIGHT ---
    cv2.putText(canvas, "ECOBIN DIAGNOSTIC", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
    # Connection Light (Circle)
    light_color = (0, 255, 0) if arduino_status == "ONLINE" else (0, 0, 255)
    cv2.circle(canvas, (w + 320, 30), 10, light_color, -1) 
    cv2.putText(canvas, arduino_status, (w + 220, 35), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)

    # --- SERIAL MONITOR BOX ---
    # Draw a "screen" for TX/RX
    cv2.rectangle(canvas, (w + 20, 60), (w + sidebar_w - 20, 150), (0, 0, 0), -1)
    cv2.rectangle(canvas, (w + 20, 60), (w + sidebar_w - 20, 150), (100, 100, 100), 1)
    
    # TX Line
    cv2.putText(canvas, "LAST TX (Sent):", (w + 30, 85), cv2.FONT_HERSHEY_PLAIN, 0.9, (150, 150, 150), 1)
    cv2.putText(canvas, last_tx, (w + 160, 85), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 1)
    
    # RX Line
    rx_color = (0, 200, 255) if rx_timer > 0 else (150, 150, 150)
    cv2.putText(canvas, "LAST RX (Recv):", (w + 30, 115), cv2.FONT_HERSHEY_PLAIN, 0.9, rx_color, 1)
    # Truncate RX if too long
    disp_rx = (last_rx[:20] + '..') if len(last_rx) > 20 else last_rx
    cv2.putText(canvas, disp_rx, (w + 160, 115), cv2.FONT_HERSHEY_PLAIN, 1.0, rx_color, 1)

    if rx_timer > 0: rx_timer -= 1

    # --- STATUS ---
    st_c = (0, 255, 0)
    if current_state == STATE_BUSY: st_c = (0, 255, 255)
    if current_state == STATE_DETECTING: st_c = (255, 0, 255)
    if current_state == STATE_WAITING_OBJECT: st_c = (255, 255, 255)
    
    cv2.putText(canvas, f"STATE: {current_state}", (w + 20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, st_c, 2)
    
    # --- LOGS ---
    y_start = 220
    for i, line in enumerate(serial_log):
        c = (180, 180, 180)
        if "TX" in line: c = (0, 255, 0)
        if "RX" in line: c = (0, 200, 255)
        if "MATCH" in line: c = (255, 0, 255)
        cv2.putText(canvas, line, (w + 10, y_start + (i * 20)), cv2.FONT_HERSHEY_PLAIN, 0.9, c, 1)

    # --- FOOTER ---
    cv2.putText(canvas, "TEST: [1] Trash  [2] Recycle", (w + 20, h - 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 255), 1)
    cv2.putText(canvas, "CTRL: [S] Start  [R] Reset", (w + 20, h - 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (100, 100, 100), 1)
    
    return canvas

# --- MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret: break

    # --- READ ARDUINO (PARSER) ---
    if arduino and arduino.in_waiting > 0:
        try:
            # [cite: 19, 21] UNO prints text like "Command received..."
            # We read everything available to clear the buffer and log it
            raw_bytes = arduino.read(arduino.in_waiting) 
            raw_str = raw_bytes.decode('utf-8', errors='ignore').strip()
            
            if raw_str:
                # Split by newline in case we got multiple messages
                lines = raw_str.split('\n')
                for line in lines:
                    clean_line = line.strip()
                    if not clean_line: continue
                    
                    log_message(f"{clean_line}", "RX")
                    
                    # [cite: 10, 23] Check for READY ('A')
                    if "A" in clean_line or "READY" in clean_line:
                        if current_state == STATE_BUSY:
                            current_state = STATE_RESET_BG
                            last_state_change = time.time()
                    
                    # [cite: 10, 24] Check for BUSY ('B')
                    if "B" in clean_line or "BUSY" in clean_line:
                        current_state = STATE_BUSY

        except Exception as e:
            log_message(f"Serial Error: {e}", "INFO")

    # --- MOTION ---
    mask = bg_subtractor.apply(frame)
    motion = cv2.countNonZero(mask)
    viz_frame, live_best_name, live_best_conf = run_live_detection(frame)
    
    # --- STATE MACHINE ---
    if current_state == STATE_IDLE:
        pass

    elif current_state == STATE_RESET_BG:
        if time.time() - last_state_change > 2.0:
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
            send_to_arduino(sig)
            current_state = STATE_BUSY
        else:
            if live_best_name: log_message(f"Unknown: {live_best_name}", "INFO")
            current_state = STATE_WAITING_OBJECT

    elif current_state == STATE_BUSY:
        cv2.putText(viz_frame, "SORTING...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # --- RENDER ---
    dashboard = draw_dashboard(viz_frame)
    cv2.imshow("Oxyzen EcoBin", dashboard)

    # --- CONTROLS ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('s'): 
        current_state = STATE_RESET_BG
        last_state_change = time.time()
    if key == ord('r'):
        current_state = STATE_RESET_BG
        last_state_change = time.time()
    
    # [cite: 11] Manual testing keys
    if key == ord('1'): 
        send_to_arduino(settings['codes']['trash'])
    if key == ord('2'): 
        send_to_arduino(settings['codes']['recyclable'])

cap.release()
if arduino: arduino.close()
cv2.destroyAllWindows()