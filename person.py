import cv2
import numpy as np
import pygame
import time
from line_notify import LineNotifier
import os

# Initialize LineNotifier
line_notifier = LineNotifier()
line_notifier.token = "J0oQ74OftbCNdiPCCfV4gs75aqtz4aAL8NiGfHERvZ4"

# Initialize pygame for audio
pygame.mixer.init()

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)

face_detected = False
counter = 0
last_notification_time = 0
process_complete = True

def handle_detection():
    global last_notification_time, process_complete
    
    # Play alert sounds
    pygame.mixer.music.load("alert.mp3")
    pygame.mixer.music.play()
    pygame.time.wait(int(pygame.mixer.Sound("alert.mp3").get_length() * 1000))
    pygame.mixer.music.load("Warning.mp3")
    pygame.mixer.music.play()

    # Capture and save the frame
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("detected_face.jpg", frame)

        # Send Line notification with image
        current_time = time.time()
        if current_time - last_notification_time > 60:  # Limit notifications to once per minute
            line_notifier.send_image("A face is detected", "detected_face.jpg")
            last_notification_time = current_time

    # Wait for a moment to allow the process to complete
    time.sleep(5)
    process_complete = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    face_detected_this_frame = len(faces) > 0

    if face_detected_this_frame and process_complete:
        if not face_detected:
            face_detected = True
            counter = 0
        counter += 1
        
        if counter == 10:
            process_complete = False
            handle_detection()
            counter = 0
    elif not face_detected_this_frame:
        face_detected = False
        counter = 0

    # Draw rectangles around detected faces (optional, for debugging)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the frame (optional, for debugging)
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()