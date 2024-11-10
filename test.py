import cv2
import io
import threading
from flask import Flask, Response
from picamera2 import Picamera2
from ultralytics import YOLO
from PIL import Image

# Set up the camera with Picam
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 1280)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Flask app initialization
app = Flask(__name__)

# Global variable to store the latest frame
latest_frame = None
frame_lock = threading.Lock()

# Function to capture and process frames in a background thread
def capture_frames():
    global latest_frame
    while True:
        # Capture a frame from the camera
        frame = picam2.capture_array()
        
        # Run YOLO model on the captured frame and store the results
        results = model(frame)
        
        # Draw detection boxes on the frame
        annotated_frame = results[0].plot()
        
        # Get inference time and calculate FPS
        inference_time = results[0].speed['inference']
        fps = 1000 / inference_time
        text = f'FPS: {fps:.1f}'
        
        # Define font and position for FPS text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = annotated_frame.shape[1] - text_size[0] - 10
        text_y = text_size[1] + 10

        # Draw the FPS text on the annotated frame
        cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Lock the frame and update the global latest_frame variable
        with frame_lock:
            latest_frame = annotated_frame

# Start the capture_frames function in a separate thread
threading.Thread(target=capture_frames, daemon=True).start()

# Define a route to serve the video stream
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                # Convert the latest frame to JPEG format
                if latest_frame is None:
                    continue
                _, jpeg = cv2.imencode('.jpg', latest_frame)
                frame = jpeg.tobytes()
            # Yield the frame as a multipart response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Define a route for the main page
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

# Start the Flask web server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)