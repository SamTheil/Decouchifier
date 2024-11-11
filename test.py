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
picam2.preview_configuration.main.size = (640, 640)  # Reduced resolution
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load a lighter YOLO model
model = YOLO("yolov8n_ncnn_model")  # or smaller model if available

# Flask app initialization
app = Flask(__name__)

# Global variable to store the latest frame and a frame lock
latest_frame = None
frame_lock = threading.Lock()

# Function to capture and process frames in a background thread
def capture_frames():
    global latest_frame
    last_time = time.time()
    
    while True:
        frame = picam2.capture_array()
        
        # Resize frame for faster processing
        frame = cv2.resize(frame, (320, 320))  # Further reduced resolution
        
        # Run YOLO model and store results
        results = model(frame)
        annotated_frame = results[0].plot()
        
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
        </head>
        <body>
            <h1>Live YOLO Detection Feed</h1>
            <img src="/video_feed" width="80%">
        </body>
    </html>
    '''

# Start Flask web server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)