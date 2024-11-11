import cv2
import io
import time
import threading
from flask import Flask, Response
from picamera2 import Picamera2
from ultralytics import YOLO
from PIL import Image
import numpy as np

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

# Function to capture and process frames in a background thread
def capture_frames():
    global latest_frame
    last_time = time.time()
    
    while True:
        frame = picam2.capture_array()
        
        # Run YOLO model and store results
        results = model(frame)

        # Filter detections for specific labels
        filtered_boxes = []
        for result in results:
            for box in result.boxes:
                # Get the class ID and map it to the label
                class_id = int(box.cls)  # Ensure class_id is an integer
                label = class_labels[class_id] if class_id < len(class_labels) else None

                if label in ["person", "dog"]:  # Only keep person and dog detections
                    filtered_boxes.append(box)

        # Plot only filtered results
        annotated_frame = frame.copy()  # Copy original frame to draw on
        for box in filtered_boxes:
            # Draw bounding boxes and labels on the frame
            annotated_frame = box.plot(annotated_frame)

        # Calculate FPS every second
        current_time = time.time()
        fps = 1 / (current_time - last_time)
        last_time = current_time
        text = f'FPS: {fps:.1f}'
        
        # Draw FPS text on the annotated frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = annotated_frame.shape[1] - text_size[0] - 10
        text_y = text_size[1] + 10
        cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
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
                /* CSS to make the image responsive and take full width of the browser */
                img {
                    width: 100%;
                    height: auto;
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
