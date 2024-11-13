import cv2
import io
import time
import threading
from flask import Flask, Response, request
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

# Global variables
latest_frame = None
frame_lock = threading.Lock()
detection_enabled = False  # Controls whether detection is active
CONFIDENCE_THRESHOLD = 0.5  # Default confidence threshold
dog_detected_frames = 0  # Counter for consecutive frames with dog detections
DOG_DETECTION_THRESHOLD = 1  # Frames required to trigger relay

# Define the YOLO class labels
class_labels = ["person", "dog"]  # Replace with the labels corresponding to your model

# Function to capture and process frames in a background thread
def capture_frames():
    global latest_frame, dog_detected_frames, detection_enabled, CONFIDENCE_THRESHOLD
    last_time = time.time()
    
    while True:
        frame = picam2.capture_array()
        
        if detection_enabled:
            # Run YOLO model and store results
            results = model(frame)
            dog_detected = False  # Flag to check if dog is detected in the current frame

            # Filter detections for specific labels and confidence
            for result in results:
                for box in result.boxes:
                    # Get the class ID, confidence, and map it to the label
                    class_id = int(box.cls)  # Ensure class_id is an integer
                    label = class_labels[class_id] if class_id < len(class_labels) else None
                    confidence = float(box.conf)

                    # Only process "dog" if confidence is above threshold
                    if label == "dog" and confidence >= CONFIDENCE_THRESHOLD:
                        dog_detected = True  # Mark dog detected in this frame

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

            # Do not update latest_frame to stop video feed
            with frame_lock:
                latest_frame = None

        else:
            # When detection is disabled, just update latest_frame for streaming
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - last_time)
            last_time = current_time
            fps_text = f'FPS: {fps:.1f}'

            # Draw FPS text on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(fps_text, font, 1, 2)[0]
            text_x = frame.shape[1] - text_size[0] - 10
            text_y = text_size[1] + 10
            cv2.putText(frame, fps_text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Lock and update the global latest_frame variable
            with frame_lock:
                latest_frame = frame

# Start frame capture in a separate thread
threading.Thread(target=capture_frames, daemon=True).start()

# Route to serve the video stream
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            time.sleep(0.1)  # Approximate 10fps
            with frame_lock:
                if latest_frame is None:
                    continue
                _, jpeg = cv2.imencode('.jpg', latest_frame)
                frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Main page route
@app.route('/', methods=['GET'])
def index():
    global detection_enabled, CONFIDENCE_THRESHOLD
    # Display video feed only when detection is off
    video_feed = ''
    if not detection_enabled:
        video_feed = '<img src="/video_feed">'

    return f'''
    <html>
        <head>
            <title>YOLO Detection</title>
            <style>
                img {{
                    max-width: 100%;
                    max-height: 100vh;
                    width: auto;
                    height: auto;
                    display: block;
                    margin: 0 auto;
                }}
                body {{
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    flex-direction: column;
                    font-family: Arial, sans-serif;
                }}
                h1 {{
                    margin-top: 20px;
                }}
                form {{
                    margin: 10px;
                }}
                input[type="submit"], button {{
                    padding: 10px 20px;
                    font-size: 16px;
                    margin: 5px;
                }}
                label {{
                    font-size: 16px;
                }}
            </style>
        </head>
        <body>
            <h1>YOLO Detection Control Panel</h1>
            {video_feed}
            <form action="/start_detection" method="post">
                <input type="submit" value="Start Detection" {'disabled' if detection_enabled else ''}>
            </form>
            <form action="/stop_detection" method="post">
                <input type="submit" value="Stop Detection" {'disabled' if not detection_enabled else ''}>
            </form>
            <form action="/test_alarm" method="post">
                <input type="submit" value="Test Alarm">
            </form>
            <form action="/set_confidence" method="post">
                <label for="confidence">Confidence Threshold:</label>
                <input type="number" id="confidence" name="confidence" step="0.1" min="0" max="1" value="{CONFIDENCE_THRESHOLD}" {'disabled' if detection_enabled else ''}>
                <input type="submit" value="Set Confidence Threshold" {'disabled' if detection_enabled else ''}>
            </form>
        </body>
    </html>
    '''

# Route to start detection
@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_enabled, latest_frame
    detection_enabled = True
    with frame_lock:
        latest_frame = None  # Clear the latest frame so video feed stops
    return '<p>Detection started. <a href="/">Go back</a></p>'

# Route to stop detection
@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_enabled
    detection_enabled = False
    return '<p>Detection stopped. <a href="/">Go back</a></p>'

# Route to test alarm
@app.route('/test_alarm', methods=['POST'])
def test_alarm():
    threading.Thread(target=alarm_thread).start()
    return '<p>Alarm tested. <a href="/">Go back</a></p>'

def alarm_thread():
    relay.turn_on_relay()
    time.sleep(2)  # Keep the relay on for 2 seconds
    relay.turn_off_relay()

# Route to set confidence threshold
@app.route('/set_confidence', methods=['POST'])
def set_confidence():
    global CONFIDENCE_THRESHOLD
    confidence = request.form.get('confidence')
    try:
        confidence_value = float(confidence)
        if 0 <= confidence_value <= 1:
            CONFIDENCE_THRESHOLD = confidence_value
            return f'<p>Confidence threshold set to {CONFIDENCE_THRESHOLD}. <a href="/">Go back</a></p>'
        else:
            return '<p>Invalid confidence value. Must be between 0 and 1. <a href="/">Go back</a></p>'
    except ValueError:
        return '<p>Invalid confidence value. Must be a number. <a href="/">Go back</a></p>'

# Start Flask web server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
