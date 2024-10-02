import time
import threading
from queue import Queue
import cv2
from ultralytics import YOLO
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# Import sensor functionalities
from hc_sr04p_distance import get_distance, GPIO_setup, GPIO_cleanup
from Updated_PZEM_Sensor_Reader_Script import connect_to_sensor, read_sensor_data
from line_notify import LineNotifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for system logic
DISTANCE_THRESHOLD = 30  # cm
CHARGING_TIMEOUT = 10  # seconds
CURRENT_THRESHOLD = 0.1  # A

# LINE Notify token (replace with your actual token)
LINE_NOTIFY_TOKEN = "YOUR_LINE_NOTIFY_TOKEN_HERE"

class SensorManager:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.data_queue = Queue()
        
        # Load YOLOv8 model
        self.model = YOLO('yolov9s_ev.pt')  # or your custom trained model
        
        # Set up GPIO for HC-SR04P
        GPIO_setup()
        
        # Set up PZEM
        self.pzem_master = connect_to_sensor()

        # Set up LINE Notifier
        self.line_notifier = LineNotifier(LINE_NOTIFY_TOKEN)

    def pzem_thread(self):
        while True:
            try:
                pzem_data = read_sensor_data(self.pzem_master)
                self.data_queue.put(('pzem', pzem_data))
            except Exception as e:
                logger.error(f"Error reading PZEM data: {e}")
                self.pzem_master = connect_to_sensor()  # Reconnect
            time.sleep(5)

    def distance_thread(self):
        while True:
            distance = get_distance()
            if distance is not None:
                self.data_queue.put(('distance', distance))
            time.sleep(1)  # Using the MEASUREMENT_INTERVAL from your original file

    def camera_thread(self):
        while True:
            ret, frame = self.camera.read()
            if ret:
                results = self.model(frame)
                car_detected, is_ev = self.process_yolo_results(results)
                self.data_queue.put(('camera', (car_detected, is_ev, frame)))
            time.sleep(0.1)

    def process_yolo_results(self, results):
        car_detected = False
        is_ev = False
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 2:  # Assuming class 2 is 'car' in your model
                    car_detected = True
                elif cls == 80:  # Assuming class 80 is 'ev' in your model
                    is_ev = True
        return car_detected, is_ev

    def process_data(self):
        car_present = False
        ev_charging_start_time = None
        while True:
            sensor, data = self.data_queue.get()
            if sensor == 'pzem':
                current = data['current_A']
                if car_present and ev_charging_start_time:
                    if current > CURRENT_THRESHOLD:
                        ev_charging_start_time = None  # Reset timer if charging detected
                    elif time.time() - ev_charging_start_time > CHARGING_TIMEOUT:
                        message = "Alert: EV car parked but not charging!"
                        logger.info(message)
                        self.line_notifier.send_notification(message)
                        ev_charging_start_time = None  # Reset timer
            elif sensor == 'distance':
                distance = data
                if distance <= DISTANCE_THRESHOLD:
                    car_present = True
                else:
                    car_present = False
                    ev_charging_start_time = None
            elif sensor == 'camera':
                car_detected, is_ev, frame = data
                if car_detected:
                    if not is_ev:
                        message = "Alert: Non-EV car detected in charging spot!"
                        logger.info(message)
                        image_path = "non_ev_car.jpg"
                        cv2.imwrite(image_path, frame)
                        self.line_notifier.send_image(message, image_path)
                        os.remove(image_path)  # Clean up the image file
                    elif car_present and not ev_charging_start_time:
                        ev_charging_start_time = time.time()

    def run(self):
        threads = [
            threading.Thread(target=self.pzem_thread),
            threading.Thread(target=self.distance_thread),
            threading.Thread(target=self.camera_thread),
            threading.Thread(target=self.process_data)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def cleanup(self):
        self.camera.release()
        GPIO_cleanup()
        if self.pzem_master:
            self.pzem_master.close()

if __name__ == "__main__":
    manager = SensorManager()
    try:
        manager.run()
    except KeyboardInterrupt:
        logger.info("Program stopped by user")
    finally:
        manager.cleanup()
        logger.info("Cleanup completed")