import cv2
import io
import time
import threading
from flask import Flask, Response
from picamera2 import Picamera2
from ultralytics import YOLO
from PIL import Image
import numpy as np
from Interfaces.relay_class import RelayClass

relay = RelayClass()

# Set up the camera with Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 640)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load a lighter YOLO model
model = YOLO("yolo11n_ncnn_model")  # or smaller model if available

# Flask app initialization
app = Flask(__name__)

# Global variable to store the latest frame and a frame lock
latest_frame = None
frame_lock = threading.Lock()

# Define the YOLO class labels
class_labels = ["person", "dog"]  # Replace with the labels corresponding to your model

# Confidence threshold for filtering detections
CONFIDENCE_THRESHOLD = 0.5  # Set the desired confidence threshold (0.5 is an example)

# Variables to track detection states
dog_detected_frames = 0  # Counter for consecutive frames with dog detections
DOG_DETECTION_THRESHOLD = 1  # Number of consecutive frames with detection required to trigger relay

# Function to capture and process frames in a background thread
def capture_frames():
    global latest_frame, dog_detected_frames
    last_time = time.time()
    
    while True:
        frame = picam2.capture_array()
        
        # Run YOLO model and store results
        results = model(frame)
        dog_detected = False  # Flag to check if dog is detected in the current frame

        # Filter detections for specific labels and confidence
        annotated_frame = frame.copy()  # Copy original frame to draw on
        for result in results:
            for box in result.boxes:
                # Get the class ID, confidence, and map it to the label
                class_id = int(box.cls)  # Ensure class_id is an integer
                label = class_labels[class_id] if class_id < len(class_labels) else None
                confidence = float(box.conf)

                # Only process "person" or "dog" if confidence is above threshold
                if label in ["person", "dog"] and confidence >= CONFIDENCE_THRESHOLD:
                    # Check if "dog" is detected
                    if label == "dog":
                        dog_detected = True  # Mark dog detected in this frame
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label and confidence text
                    text = f"{label} ({confidence:.2f})"
                    cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Check dog detection status
        if dog_detected:
            dog_detected_frames += 1
        else:
            dog_detected_frames = 0  # Reset counter if no dog detected

        # Trigger relay if dog is detected in consecutive frames
        if dog_detected_frames > DOG_DETECTION_THRESHOLD:
            relay.turn_on_relay()
            time.sleep(2)  # Keep the relay on for 2 seconds
            relay.turn_off_relay()
            dog_detected_frames = 0  # Reset counter after triggering relay

        # Calculate FPS every second
        current_time = time.time()
        fps = 1 / (current_time - last_time)
        last_time = current_time
        fps_text = f'FPS: {fps:.1f}'
        
        # Draw FPS text on the annotated frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(fps_text, font, 1, 2)[0]
        text_x = annotated_frame.shape[1] - text_size[0] - 10
        text_y = text_size[1] + 10
        cv2.putText(annotated_frame, fps_text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Lock and update the global latest_frame variable
        with frame_lock:
            latest_frame = annotated_frame

# Start frame capture in a separate thread
threading.Thread(target=capture_frames, daemon=True).start()

# Route to serve the video stream
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            time.sleep(0.1)  # Reduced frame rate to reduce load on Flask
            with frame_lock:
                if latest_frame is None:
                    continue
                _, jpeg = cv2.imencode('.jpg', latest_frame)
                frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Main page route
@app.route('/')
def index():
    return '''
    <html>
        <head>
            <title>YOLO Detection</title>
            <style>
                /* CSS to make the image fit within the viewport without stretching */
                img {
                    max-width: 100%;         /* Scale down the width to fit the browser width */
                    max-height: 100vh;       /* Limit height to viewport height to avoid scrolling */
                    width: auto;             /* Maintain aspect ratio */
                    height: auto;            /* Maintain aspect ratio */
                    display: block;
                    margin: 0 auto;          /* Center the image */
                }
                body {
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    flex-direction: column;
                    font-family: Arial, sans-serif;
                }
                h1 {
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <h1>Live YOLO Detection Feed</h1>
            <img src="/video_feed">
        </body>
    </html>
    '''

# Start Flask web server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
