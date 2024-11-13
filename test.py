import cv2
import io
import time
import threading
import base64
import os
from flask import Flask, Response, request, jsonify
from picamera2 import Picamera2
from ultralytics import YOLO
from PIL import Image
import numpy as np
from Interfaces.relay_class import RelayClass

relay = RelayClass()

# Initialize Picamera2
picam2 = Picamera2()

# Get full sensor resolution
sensor_resolution = picam2.sensor_resolution  # For example, (3280, 2464)

# Desired downscaled resolution (maintain aspect ratio)
desired_width = 640
desired_height = int(sensor_resolution[1] * (desired_width / sensor_resolution[0]))
desired_resolution = (desired_width, desired_height)

# Create still configuration with full resolution
camera_config = picam2.create_still_configuration(
    main={"size": sensor_resolution, "format": "RGB888"}
)
picam2.configure(camera_config)

# Set ScalerCrop to full sensor area to avoid cropping
picam2.set_controls({"ScalerCrop": (0, 0, sensor_resolution[0], sensor_resolution[1])})

picam2.start()

# Load the YOLO model
model = YOLO("yolo11n_ncnn_model")  # or a smaller model if available

# Get the class names from the model (dictionary mapping class IDs to labels)
class_labels = model.names  # e.g., {0: 'person', 16: 'dog', ...}

# Flask app initialization
app = Flask(__name__)

# Global variables
latest_frame = None
frame_lock = threading.Lock()
detection_enabled = False  # Controls whether detection is active
CONFIDENCE_THRESHOLD = 0.5  # Default confidence threshold
object_detected_frames = 0  # Counter for consecutive frames with object detections
DETECTION_THRESHOLD = 1  # Frames required to trigger relay
mask = None
mask_lock = threading.Lock()
MASK_FILE = 'mask.npy'  # File to save the mask

# Find class IDs for 'person' and 'dog'
person_class_id = None
dog_class_id = None

for id, name in class_labels.items():
    if name == 'person':
        person_class_id = id
    elif name == 'dog':
        dog_class_id = id

# Default detection class ID (set to 'dog' by default)
detection_class_id = dog_class_id

# Load the mask from disk if it exists
def load_mask():
    global mask
    if os.path.exists(MASK_FILE):
        with mask_lock:
            mask = np.load(MASK_FILE)
        print("Mask loaded from disk.")
    else:
        print("No saved mask found.")

load_mask()

# Function to capture and process frames in a background thread
def capture_frames():
    global latest_frame, object_detected_frames, detection_enabled, CONFIDENCE_THRESHOLD, mask
    last_time = time.time()

    while True:
        # Capture at full resolution
        frame = picam2.capture_array()
        # Rotate the image 180 degrees if needed
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        # Downscale to desired resolution
        frame = cv2.resize(frame, desired_resolution, interpolation=cv2.INTER_LINEAR)

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

        if detection_enabled:
            # Run YOLO model and store results
            results = model(frame)
            object_detected = False  # Flag to check if the object is detected in the current frame

            # Process results
            for result in results:
                for box in result.boxes:
                    # Get the class ID, confidence, and map it to the label
                    class_id = int(box.cls)  # Ensure class_id is an integer
                    label = class_labels.get(class_id, f"ID {class_id}")
                    confidence = float(box.conf)

                    # Debug print statements
                    print(f"Detected object: {label} (ID: {class_id}), Confidence: {confidence}")

                    # Only process if class_id matches detection_class_id and confidence is above threshold
                    if class_id == detection_class_id and confidence >= CONFIDENCE_THRESHOLD:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # [x1, y1, x2, y2]
                        # Calculate center point
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        # Check if center point is inside the mask
                        inside_mask = True
                        if mask is not None:
                            with mask_lock:
                                if (0 <= center_y < mask.shape[0]) and (0 <= center_x < mask.shape[1]):
                                    if mask[center_y, center_x]:
                                        inside_mask = True
                                    else:
                                        inside_mask = False
                                else:
                                    inside_mask = False
                        else:
                            inside_mask = True  # If no mask is defined, default to entire screen

                        if inside_mask:
                            object_detected = True  # Object detected inside the selected area
                            # Draw a dot over the object
                            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                            # Optionally, draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            # Draw label and confidence
                            label_text = f"{label}: {confidence:.2f}"
                            cv2.putText(frame, label_text, (x1, y1 - 10), font, 0.5, (0, 255, 0), 2)

                # Break after processing the first result
                break  # Since we only process one frame at a time

            # Check object detection status
            if object_detected:
                object_detected_frames += 1
            else:
                object_detected_frames = 0  # Reset counter if no object detected

            # Trigger relay if object is detected in consecutive frames
            if object_detected_frames >= DETECTION_THRESHOLD:
                relay.turn_on_relay()
                time.sleep(2)  # Keep the relay on for 2 seconds
                relay.turn_off_relay()
                object_detected_frames = 0  # Reset counter after triggering relay

        # Update latest_frame with the annotated frame
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

# Route to capture a still image
@app.route('/capture_image')
def capture_image():
    # Capture at full resolution
    frame = picam2.capture_array()
    # Rotate the image 180 degrees if needed
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    # Downscale to desired resolution
    frame = cv2.resize(frame, desired_resolution, interpolation=cv2.INTER_LINEAR)
    # Encode as JPEG
    _, jpeg = cv2.imencode('.jpg', frame)
    response = Response(jpeg.tobytes(), mimetype='image/jpeg')
    return response

# Route to save the mask
@app.route('/save_mask', methods=['POST'])
def save_mask():
    global mask
    data = request.get_json()
    mask_data = data.get('mask')
    if mask_data:
        # Decode the base64 image
        header, encoded = mask_data.split(',', 1)
        mask_bytes = base64.b64decode(encoded)
        # Convert bytes to numpy array
        nparr = np.frombuffer(mask_bytes, np.uint8)
        mask_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        # Resize mask to match frame size
        mask_img = cv2.resize(mask_img, desired_resolution)
        # Convert to binary mask (0 and 1)
        _, binary_mask = cv2.threshold(mask_img, 127, 1, cv2.THRESH_BINARY)
        with mask_lock:
            mask = binary_mask
            # Save the mask to disk
            np.save(MASK_FILE, mask)
        return jsonify({'message': 'Mask saved successfully.'})
    else:
        return jsonify({'message': 'No mask data received.'}), 400

# Main page route
@app.route('/', methods=['GET'])
def index():
    global detection_enabled, CONFIDENCE_THRESHOLD, detection_class_id, class_labels
    # Video feed is always displayed
    video_feed = '<img src="/video_feed" id="video-stream">'

    # Generate detection class options
    options = ''
    for id, label in class_labels.items():
        if label in ['person', 'dog']:  # Limit to 'person' and 'dog' for simplicity
            selected = 'selected' if detection_class_id == id else ''
            options += f'<option value="{id}" {selected}>{label}</option>'

    return f'''
    <html>
        <head>
            <title>YOLO Detection Control Panel</title>
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
                #canvas-container {{
                    position: relative;
                    display: none;
                }}
                #snapshot {{
                    position: absolute;
                    top: 0;
                    left: 0;
                }}
                #drawingCanvas {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    background: transparent;
                }}
            </style>
            <script>
                function sendRequest(endpoint, data) {{
                    fetch(endpoint, {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify(data)
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        alert(data.message);
                        // Reload the page to update the UI
                        window.location.reload();
                    }})
                    .catch((error) => {{
                        console.error('Error:', error);
                    }});
                }}

                function defineDetectionZone() {{
                    // Fetch the image from the server
                    fetch('/capture_image')
                    .then(response => response.blob())
                    .then(blob => {{
                        const img = document.getElementById('snapshot');
                        img.src = URL.createObjectURL(blob);
                        img.onload = function() {{
                            // Set up the canvas dimensions
                            img.width = img.naturalWidth;
                            img.height = img.naturalHeight;
                            const canvas = document.getElementById('drawingCanvas');
                            canvas.width = img.width;
                            canvas.height = img.height;
                            // Show the canvas-container
                            document.getElementById('canvas-container').style.display = 'block';
                            setUpDrawing();
                        }}
                    }})
                    .catch((error) => {{
                        console.error('Error:', error);
                    }});
                }}

                function setUpDrawing() {{
                    const canvas = document.getElementById('drawingCanvas');
                    const context = canvas.getContext('2d');
                    let drawing = false;
                    let erasing = false;
                    let brushSize = parseInt(document.getElementById('brushSize').value);

                    canvas.onmousedown = function(e) {{
                        drawing = true;
                        context.beginPath();
                        context.moveTo(e.offsetX, e.offsetY);
                    }};

                    canvas.onmousemove = function(e) {{
                        if (drawing) {{
                            context.lineTo(e.offsetX, e.offsetY);
                            context.lineWidth = brushSize;
                            context.lineCap = 'round';
                            if (erasing) {{
                                context.globalCompositeOperation = 'destination-out';
                                context.strokeStyle = 'rgba(0,0,0,1)';
                            }} else {{
                                context.globalCompositeOperation = 'source-over';
                                context.strokeStyle = 'red';
                            }}
                            context.stroke();
                        }}
                    }};

                    canvas.onmouseup = function(e) {{
                        if (drawing) {{
                            context.lineTo(e.offsetX, e.offsetY);
                            context.stroke();
                            context.closePath();
                            drawing = false;
                        }}
                    }};

                    canvas.onmouseout = function(e) {{
                        if (drawing) {{
                            context.closePath();
                            drawing = false;
                        }}
                    }};

                    document.getElementById('brushSize').onchange = function(e) {{
                        brushSize = parseInt(e.target.value);
                    }};

                    document.getElementById('eraseToggle').onclick = function(e) {{
                        erasing = !erasing;
                        e.target.textContent = erasing ? 'Erase Mode' : 'Draw Mode';
                    }};
                }}

                function saveMask() {{
                    const canvas = document.getElementById('drawingCanvas');
                    const maskData = canvas.toDataURL('image/png');

                    fetch('/save_mask', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify({{'mask': maskData}})
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        alert(data.message);
                        // Hide the canvas-container
                        document.getElementById('canvas-container').style.display = 'none';
                        // Reload the page to update the UI
                        window.location.reload();
                    }})
                    .catch((error) => {{
                        console.error('Error:', error);
                    }});
                }}
            </script>
        </head>
        <body>
            <h1>YOLO Detection Control Panel</h1>
            {video_feed}
            <button onclick="sendRequest('/start_detection', {{}})" {'disabled' if detection_enabled else ''}>Start Detection</button>
            <button onclick="sendRequest('/stop_detection', {{}})" {'disabled' if not detection_enabled else ''}>Stop Detection</button>
            <button onclick="sendRequest('/test_alarm', {{}})">Test Alarm</button>
            <button onclick="defineDetectionZone()">Define Detection Zone</button>
            <br>
            <label for="confidence">Confidence Threshold:</label>
            <input type="number" id="confidence" step="0.1" min="0" max="1" value="{CONFIDENCE_THRESHOLD}" {'disabled' if detection_enabled else ''}>
            <button onclick="sendRequest('/set_confidence', {{'confidence': document.getElementById('confidence').value}})" {'disabled' if detection_enabled else ''}>Set Confidence Threshold</button>
            <br>
            <label for="detectionClass">Detection Class:</label>
            <select id="detectionClass">
                {options}
            </select>
            <button onclick="sendRequest('/set_detection_class', {{'class_id': document.getElementById('detectionClass').value}})">Set Detection Class</button>
            <br>
            <div id="canvas-container">
                <img id="snapshot">
                <canvas id="drawingCanvas"></canvas>
                <br>
                <label for="brushSize">Brush Size:</label>
                <input type="number" id="brushSize" value="5" min="1" max="50">
                <button id="eraseToggle">Draw Mode</button>
                <button onclick="saveMask()">Save Zone</button>
            </div>
        </body>
    </html>
    '''

# Route to start detection
@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_enabled
    detection_enabled = True
    return jsonify({'message': 'Detection started.'})

# Route to stop detection
@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_enabled
    detection_enabled = False
    return jsonify({'message': 'Detection stopped.'})

# Route to test alarm
@app.route('/test_alarm', methods=['POST'])
def test_alarm():
    threading.Thread(target=alarm_thread).start()
    return jsonify({'message': 'Alarm tested.'})

def alarm_thread():
    relay.turn_on_relay()
    time.sleep(2)  # Keep the relay on for 2 seconds
    relay.turn_off_relay()

# Route to set confidence threshold
@app.route('/set_confidence', methods=['POST'])
def set_confidence():
    global CONFIDENCE_THRESHOLD
    data = request.get_json()
    confidence = data.get('confidence')
    try:
        confidence_value = float(confidence)
        if 0 <= confidence_value <= 1:
            CONFIDENCE_THRESHOLD = confidence_value
            return jsonify({'message': f'Confidence threshold set to {CONFIDENCE_THRESHOLD}.'})
        else:
            return jsonify({'message': 'Invalid confidence value. Must be between 0 and 1.'}), 400
    except (ValueError, TypeError):
        return jsonify({'message': 'Invalid confidence value. Must be a number.'}), 400

# Route to set detection class ID
@app.route('/set_detection_class', methods=['POST'])
def set_detection_class():
    global detection_class_id
    data = request.get_json()
    class_id_str = data.get('class_id')
    try:
        class_id = int(class_id_str)
        if class_id in class_labels:
            detection_class_id = class_id
            label = class_labels[class_id]
            return jsonify({'message': f'Detection class set to {label} (ID: {class_id}).'})
        else:
            return jsonify({'message': 'Invalid detection class ID.'}), 400
    except (ValueError, TypeError):
        return jsonify({'message': 'Invalid class ID format. Must be an integer.'}), 400

# Start Flask web server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)