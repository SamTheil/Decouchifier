from picamera2 import Picamera2, Preview
from flask import Flask, send_file, request
from io import BytesIO
from time import sleep
from threading import Thread, Lock
from PIL import Image

app = Flask(__name__)
camera = Picamera2()

# Initialize camera with default settings
camera_config = camera.create_still_configuration(main={"size": (1920, 1080)})
camera.configure(camera_config)
camera.start()

latest_image = BytesIO()
settings_lock = Lock()  # Lock for thread-safe settings update

# Default camera settings
camera_settings = {
    "ExposureTime": None,  # None for auto
    "AnalogueGain": None,  # None for auto
    "Brightness": 0.0,     # Range: -1.0 to 1.0
    "Contrast": 1.0,       # Range: 0.0 to 2.0
    "AwbEnable": True,
    "AeEnable": True
}

def capture_image():
    global latest_image
    while True:
        stream = BytesIO()
        with settings_lock:
            # Build the controls dict, excluding None values
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
        img = Image.fromarray(image)
        # Save the image with higher JPEG quality
        img.save(stream, format='jpeg', quality=85)
        stream.seek(0)
        latest_image = stream
        sleep(3)  # Wait 1 second before capturing the next image

@app.route('/', methods=['GET', 'POST'])
def index():
    """Webpage that displays the camera image and settings form."""
    if request.method == 'POST':
        # Update camera settings based on form input
        with settings_lock:
            # Update auto white balance
            awb = request.form.get('awb')
            camera_settings["AwbEnable"] = awb == 'on'

            # Update auto exposure
            ae = request.form.get('ae')
            camera_settings["AeEnable"] = ae == 'on'

            # Update exposure time
            exposure = request.form.get('exposure')
            if exposure and not camera_settings["AeEnable"]:
                camera_settings["ExposureTime"] = int(exposure)
            else:
                camera_settings["ExposureTime"] = None

            # Update gain
            gain = request.form.get('gain')
            if gain and not camera_settings["AeEnable"]:
                camera_settings["AnalogueGain"] = float(gain)
            else:
                camera_settings["AnalogueGain"] = None

            # Update brightness
            brightness = request.form.get('brightness')
            if brightness is not None:
                camera_settings["Brightness"] = float(brightness)

            # Update contrast
            contrast = request.form.get('contrast')
            if contrast is not None:
                camera_settings["Contrast"] = float(contrast)

    # Generate the HTML page with the current settings
    return f'''
    <html>
        <head>
            <title>Raspberry Pi Camera</title>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .container {{ display: flex; }}
                .image {{ flex: 1; }}
                .settings {{ flex: 1; padding-left: 20px; }}
                input[type="range"] {{ width: 100%; }}
            </style>
        </head>
        <body>
            <h1>Live Camera Feed</h1>
            <div class="container">
                <div class="image">
                    <!-- Scaled down image display -->
                    <img src="/image.jpg" width="640" height="480"/>
                </div>
                <div class="settings">
                    <h2>Camera Settings</h2>
                    <form method="post">
                        <label for="exposure">Exposure Time (µs):</label><br>
                        <input type="number" id="exposure" name="exposure" min="1" max="1000000" value="{camera_settings['ExposureTime'] if camera_settings['ExposureTime'] else ''}"><br><br>

                        <label for="gain">Analogue Gain:</label><br>
                        <input type="number" step="0.1" id="gain" name="gain" min="1.0" max="16.0" value="{camera_settings['AnalogueGain'] if camera_settings['AnalogueGain'] else ''}"><br><br>

                        <label for="brightness">Brightness (-1.0 to 1.0):</label><br>
                        <input type="range" id="brightness" name="brightness" min="-1.0" max="1.0" step="0.1" value="{camera_settings['Brightness']}"><br><br>

                        <label for="contrast">Contrast (0.0 to 2.0):</label><br>
                        <input type="range" id="contrast" name="contrast" min="0.0" max="2.0" step="0.1" value="{camera_settings['Contrast']}"><br><br>

                        <label for="awb">Auto White Balance:</label>
                        <input type="checkbox" id="awb" name="awb" {'checked' if camera_settings['AwbEnable'] else ''}><br><br>

                        <label for="ae">Auto Exposure:</label>
                        <input type="checkbox" id="ae" name="ae" {'checked' if camera_settings['AeEnable'] else ''}><br><br>

                        <input type="submit" value="Update Settings">
                    </form>
                </div>
            </div>
            <script>
                // Auto-refresh the image every second
                setInterval(() => {{
                    const img = document.querySelector('img');
                    img.src = '/image.jpg?' + new Date().getTime();
                }}, 1000);
            </script>
        </body>
    </html>
    '''

@app.route('/image.jpg')
def image():
    """Returns the latest image captured."""
    # Ensure the image file is open and valid
    image_stream = BytesIO(latest_image.getvalue())
    return send_file(image_stream, mimetype='image/jpeg')

if __name__ == '__main__':
    # Start the image capture thread
    capture_thread = Thread(target=capture_image)
    capture_thread.daemon = True
    capture_thread.start()

    # Run the Flask web server
    app.run(host='0.0.0.0', port=5000)
