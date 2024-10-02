import time
import threading
from queue import Queue
import cv2
from ultralytics import YOLO
import logging
import os
from datetime import datetime
from playsound import playsound
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
DISTANCE_THRESHOLD = 150  # cm
CHARGING_TIMEOUT = 600  # 10 minutes in seconds
CURRENT_THRESHOLD = 0.1  # A
DETECTION_STABILITY_TIME = 5  # seconds
DISTANCE_READINGS = 5  # Number of distance readings to average

# LINE Notify token (replace with your actual token)
LINE_NOTIFY_TOKEN = os.getenv('LINE_NOTIFY_TOKEN')

class SensorManager:
    def __init__(self):
        self.camera = None
        self.data_queue = Queue()
        
        # Load YOLOv8 model
        self.model = YOLO('yolov9s_ev.pt')  # or your custom trained model
        
        # Set up GPIO for HC-SR04P
        GPIO_setup()
        
        # Set up PZEM
        self.pzem_master = None

        # Set up LINE Notifier
        self.line_notifier = LineNotifier()

    def distance_thread(self):
        while True:
            distances = []
            for _ in range(DISTANCE_READINGS):
                distance = get_distance()
                if distance is not None:
                    distances.append(distance)
                time.sleep(0.1)
            
            if distances:
                avg_distance = sum(distances) / len(distances)
                if avg_distance <= DISTANCE_THRESHOLD:
                    self.data_queue.put(('distance', avg_distance))
            time.sleep(1)

    def camera_thread(self):
        detection_start_time = None
        last_bbox = None
        while True:
            if self.camera is None:
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    logger.error("Failed to open camera")
                    time.sleep(5)
                    continue

            ret, frame = self.camera.read()
            if not ret:
                logger.error("Failed to capture frame")
                self.camera.release()
                self.camera = None
                continue

            results = self.model(frame)
            car_detected, is_ev, current_bbox = self.process_yolo_results(results)

            if car_detected:
                if detection_start_time is None:
                    detection_start_time = time.time()
                    last_bbox = current_bbox
                elif current_bbox != last_bbox:
                    detection_start_time = time.time()
                    last_bbox = current_bbox
                elif time.time() - detection_start_time >= DETECTION_STABILITY_TIME:
                    self.data_queue.put(('camera', (is_ev, frame)))
                    detection_start_time = None
                    last_bbox = None
            else:
                detection_start_time = None
                last_bbox = None

            time.sleep(0.1)

    def process_yolo_results(self, results):
        car_detected = False
        is_ev = False
        current_bbox = None
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 2:  # Assuming class 2 is 'car' in your model
                    car_detected = True
                    current_bbox = box.xyxy[0].tolist()
                elif cls == 80:  # Assuming class 80 is 'ev' in your model
                    is_ev = True
                    current_bbox = box.xyxy[0].tolist()
        return car_detected, is_ev, current_bbox

    def pzem_thread(self):
        while True:
            if self.pzem_master is None:
                self.pzem_master = connect_to_sensor()
            
            try:
                pzem_data = read_sensor_data(self.pzem_master)
                self.data_queue.put(('pzem', pzem_data))
            except Exception as e:
                logger.error(f"Error reading PZEM data: {e}")
                self.pzem_master.close()
                self.pzem_master = None
            time.sleep(5)

    def process_data(self):
        ev_charging_start_time = None
        while True:
            sensor, data = self.data_queue.get()
            if sensor == 'distance':
                # Start camera detection
                threading.Thread(target=self.camera_thread).start()
            elif sensor == 'camera':
                is_ev, frame = data
                if not is_ev:
                    self.handle_non_ev_detection(frame)
                else:
                    # Start PZEM monitoring
                    threading.Thread(target=self.pzem_thread).start()
                    ev_charging_start_time = time.time()
            elif sensor == 'pzem':
                current = data['current_A']
                if current <= CURRENT_THRESHOLD:
                    if ev_charging_start_time and time.time() - ev_charging_start_time > CHARGING_TIMEOUT:
                        self.handle_non_charging_ev()
                        ev_charging_start_time = None
                else:
                    ev_charging_start_time = time.time()  # Reset timer if charging detected

    def handle_non_ev_detection(self, frame):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"{timestamp} a non-ev car is detected"
        logger.info(message)
        
        image_path = "non_ev_car.jpg"
        cv2.imwrite(image_path, frame)
        self.line_notifier.send_image(message, image_path)
        os.remove(image_path)  # Clean up the image file
        
        playsound("alert.mp3")
        playsound("Warning.mp3")

    def handle_non_charging_ev(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"{timestamp} an Electric vehicle is not charging for 10 minutes"
        logger.info(message)
        
        ret, frame = self.camera.read()
        if ret:
            image_path = "non_charging_ev.jpg"
            cv2.imwrite(image_path, frame)
            self.line_notifier.send_image(message, image_path)
            os.remove(image_path)  # Clean up the image file
        
        playsound("not_charging.mp3")

    def run(self):
        threads = [
            threading.Thread(target=self.distance_thread),
            threading.Thread(target=self.process_data)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def cleanup(self):
        if self.camera:
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
