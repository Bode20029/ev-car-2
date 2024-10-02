import time
import cv2
from ultralytics import YOLO
import numpy as np
import requests
import json
import serial
import logging
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DISTANCE_THRESHOLD = 150  # cm
STABLE_DETECTION_TIME = 5  # seconds
CONFIDENCE_THRESHOLD = 0.5
PZEM_MONITORING_TIME = 600  # 10 minutes in seconds
LINE_NOTIFY_TOKEN = 'J0oQ74OftbCNdiPCCfV4gs75aqtz4aAL8NiGfHERvZ4'

# Load YOLOv8 model
model = YOLO('path/to/your/yolov8_model.pt')

# Initialize video capture
cap = cv2.VideoCapture(0)

def get_filtered_distance():
    # Implement the logic from hcsr04new.py here
    # For brevity, I'm using a simplified version
    distances = []
    for _ in range(3):
        # Replace with actual distance measurement code
        distance = np.random.uniform(0, 200)
        distances.append(distance)
    return np.mean(distances)

def is_ev(detections):
    for det in detections:
        if det.cls in ['EV', 'electric vehicle']:
            return True
    return False

def send_line_notify(message):
    url = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {LINE_NOTIFY_TOKEN}'}
    data = {'message': message}
    response = requests.post(url, headers=headers, data=data)
    return response.status_code == 200

def connect_to_pzem():
    try:
        ser = serial.Serial(
            port='/dev/ttyUSB0',
            baudrate=9600,
            bytesize=8,
            parity='N',
            stopbits=1,
            xonxoff=0
        )
        master = modbus_rtu.RtuMaster(ser)
        master.set_timeout(2.0)
        master.set_verbose(True)
        logger.info("Successfully connected to the PZEM sensor")
        return master
    except Exception as e:
        logger.error(f"Failed to connect to PZEM: {e}")
        return None

def read_pzem_data(master):
    try:
        data = master.execute(1, cst.READ_INPUT_REGISTERS, 0, 10)
        return {
            "voltage": data[0] / 10.0,
            "current_A": (data[1] + (data[2] << 16)) / 1000.0,
            "power_W": (data[3] + (data[4] << 16)) / 10.0,
            "energy_Wh": data[5] + (data[6] << 16),
            "frequency_Hz": data[7] / 10.0,
            "power_factor": data[8] / 100.0,
            "alarm": data[9]
        }
    except Exception as e:
        logger.error(f"Error reading PZEM data: {e}")
        return None

def monitor_pzem():
    master = connect_to_pzem()
    if not master:
        return False

    start_time = time.time()
    initial_power = None

    while time.time() - start_time < PZEM_MONITORING_TIME:
        data = read_pzem_data(master)
        if data:
            if initial_power is None:
                initial_power = data['power_W']
            elif abs(data['power_W'] - initial_power) > 10:  # Assuming 10W change is significant
                master.close()
                return True  # Power change detected
        time.sleep(5)

    master.close()
    return False  # No significant power change detected

def main():
    try:
        while True:
            # Check distance
            distances = [get_filtered_distance() for _ in range(3)]
            if np.mean(distances) <= DISTANCE_THRESHOLD:
                # Object detected, start camera
                detection_start = time.time()
                stable_detection = None

                while time.time() - detection_start < STABLE_DETECTION_TIME:
                    ret, frame = cap.read()
                    if not ret:
                        logger.error("Failed to grab frame")
                        break

                    results = model(frame)
                    
                    if results and len(results) > 0:
                        r = results[0]
                        if r.boxes:
                            box = r.boxes[0]
                            if box.conf > CONFIDENCE_THRESHOLD:
                                current_detection = is_ev(r.boxes)
                                if stable_detection is None:
                                    stable_detection = current_detection
                                elif stable_detection != current_detection:
                                    stable_detection = None
                                    break

                if stable_detection is True:  # EV detected
                    logger.info("EV detected, monitoring charging...")
                    if not monitor_pzem():
                        send_line_notify("An EV car isn't charging for 10 minutes")
                elif stable_detection is False:  # Non-EV detected
                    send_line_notify("There's a non-EV car parking")

                # Check if vehicle has left
                while True:
                    ret, frame = cap.read()
                    results = model(frame)
                    if not results or len(results) == 0 or not results[0].boxes:
                        time.sleep(5)  # Wait for 5 seconds to confirm
                        ret, frame = cap.read()
                        results = model(frame)
                        if not results or len(results) == 0 or not results[0].boxes:
                            break  # Vehicle has left, go back to distance checking

            time.sleep(1)  # Small delay before next distance check

    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        send_line_notify(f"Error in EV monitoring system: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
