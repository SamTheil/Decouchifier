import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
from flask import Flask, send_file, request
from io import BytesIO
from threading import Thread, Lock
from PIL import Image

# Load the TensorFlow Lite model and allocate tensors
interpreter = tflite.Interpreter(model_path='dog_detector.tflite')
interpreter.allocate_tensors()

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Flask setup
app = Flask(__name__)
camera = Picamera2()

# Initialize camera with default settings
camera_config = camera.create_still_configuration(main={"size": (3280, 2464)})
camera.configure(camera_config)
camera.start()

latest_image = BytesIO()
settings_lock = Lock()  # Lock for thread-safe settings update

# Default camera settings
camera_settings = {
    "ExposureTime": None,
    "AnalogueGain": None,
    "Brightness": 0.0,
    "Contrast": 1.0,
    "AwbEnable": True,
    "AeEnable": True
}

def detect_dogs(image):
    # Preprocess the image for the model
    img_resized = cv2.resize(image, (300, 300))
    img_resized = np.expand_dims(img_resized, axis=0).astype(np.float32)

    # Set the tensor to the interpreter
    interpreter.set_tensor(input_details[0]['index'], img_resized)
    interpreter.invoke()

    # Get the detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index (e.g., 'dog')
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores

    height, width, _ = image.shape
    dog_detections = []

    for i in range(len(scores)):
        if scores[i] > 0.5 and int(classes[i]) == 12:  # Class ID 12 is for dogs in COCO dataset
            ymin, xmin, ymax, xmax = boxes[i]
            # Calculate the center of the dog bounding box
            center_x = int((xmin + xmax) * width / 2)
            center_y = int((ymin + ymax) * height / 2)
            dog_detections.append((center_x, center_y))

    return dog_detections

def capture_image():
    global latest_image
    while True:
        stream = BytesIO()
        with settings_lock:
            controls = {
                "Brightness": camera_settings["Brightness"],
                "Contrast": camera_settings["Contrast"],
                "AwbEnable": camera_settings["AwbEnable"],
                "AeEnable": camera_settings["AeEnable"]
            }
            if camera_settings["ExposureTime"] is not None:
                controls["ExposureTime"] = camera_settings["ExposureTime"]
            if camera_settings["AnalogueGain"] is not None:
                controls["AnalogueGain"] = camera_settings["AnalogueGain"]
            camera.set_controls(controls)

        # Capture the image
        image = camera.capture_array()

        # Detect dogs and draw circles
        dog_detections = detect_dogs(image)
        for (x, y) in dog_detections:
            cv2.circle(image, (x, y), 10, (0, 255, 0), thickness=-1)  # Green circle

        img = Image.fromarray(image)
        img.save(stream, format='jpeg', quality=85)
        stream.seek(0)
        latest_image = stream
        sleep(3)

@app.route('/', methods=['GET', 'POST'])
def index():
    """Webpage that displays the camera image and settings form."""
    # HTML rendering code here, similar to your original code...
    pass

@app.route('/image.jpg')
def image():
    """Returns the latest image captured."""
    image_stream = BytesIO(latest_image.getvalue())
    return send_file(image_stream, mimetype='image/jpeg')

if __name__ == '__main__':
    # Start the image capture thread
    capture_thread = Thread(target=capture_image)
    capture_thread.daemon = True
    capture_thread.start()

    # Run the Flask web server
    app.run(host='0.0.0.0', port=5000)
