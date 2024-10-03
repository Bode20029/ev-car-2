import cv2
import numpy as np
import pygame
import time
import os
from line_notify import LineNotifier
from pymongo import MongoClient
from bson.binary import Binary
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LineNotifier
line_notifier = LineNotifier()
line_notifier.token = os.getenv("LINE_NOTIFIER_TOKEN")

# Initialize pygame for audio
pygame.mixer.init()

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# MongoDB setup
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("DB_NAME")]
collection = db[os.getenv("COLLECTION_NAME")]

last_notification_time = 0

def connect_to_camera():
    return cv2.VideoCapture(0)

def handle_detection(frame):
    global last_notification_time
    
    # Play alert sounds
    pygame.mixer.music.load("alert.mp3")
    pygame.mixer.music.play()
    pygame.time.wait(int(pygame.mixer.Sound("alert.mp3").get_length() * 1000))
    pygame.mixer.music.load("Warning.mp3")
    pygame.mixer.music.play()

    # Save the frame
    cv2.imwrite("detected_face.jpg", frame)

    # Send Line notification with image
    current_time = time.time()
    if current_time - last_notification_time > 60:  # Limit notifications to once per minute
        line_notifier.send_image("A face is detected", "detected_face.jpg")
        last_notification_time = current_time

    # Convert image to binary for MongoDB
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_binary = img_encoded.tobytes()

    # Create a document to store in MongoDB
    document = {
        "filename": f"image{time.strftime('%Y%m%d_%H%M%S')}.jpg",
        "image": Binary(img_binary),
        "timestamp": time.time()
    }

    # Insert the document into MongoDB
    result = collection.insert_one(document)
    print(f"Image uploaded with ID: {result.inserted_id}")

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0

def main():
    cap = connect_to_camera()
    face_detected = False
    counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            if detect_faces(frame):
                if not face_detected:
                    face_detected = True
                    counter = 0
                counter += 1
                
                if counter == 10:
                    handle_detection(frame)
                    counter = 0
                    
                    # Close the camera
                    cap.release()
                    cv2.destroyAllWindows()
                    
                    print("Camera closed. Waiting for 5 seconds...")
                    time.sleep(5)  # Wait for 5 seconds
                    
                    print("Reopening camera...")
                    cap = connect_to_camera()  # Reconnect to the camera
                    
                    face_detected = False  # Reset face detection state
            else:
                face_detected = False
                counter = 0

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        client.close()

if __name__ == "__main__":
    main()