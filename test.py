from picamera2 import Picamera2, Preview
from flask import Flask, send_file
from io import BytesIO
from time import sleep
from threading import Thread
from PIL import Image

app = Flask(__name__)
camera = Picamera2()
# Increase the resolution to 1920x1080
camera_config = camera.create_still_configuration(main={"size": (1920, 1080)})
camera.configure(camera_config)
camera.start()

latest_image = BytesIO()

camera.set_controls({
    "ExposureTime": 10000,  # Adjust as needed
    "AnalogueGain": 1.0,
    "AwbEnable": True,
    "AeEnable": True
})

def capture_image():
    global latest_image
    while True:
        stream = BytesIO()
        image = camera.capture_array()
        img = Image.fromarray(image)
        # Save the image with higher JPEG quality
        img.save(stream, format='jpeg', quality=95)
        stream.seek(0)
        latest_image = stream
        sleep(1)  # Wait 1 second before capturing the next image

@app.route('/')
def index():
    """Simple webpage that displays the camera image."""
    return '''
    <html>
        <head><title>Raspberry Pi Camera</title></head>
        <body>
            <h1>Live Camera Feed</h1>
            <img src="/image.jpg" width="1920" height="1080"/>
        </body>
    </html>
    '''

@app.route('/image.jpg')
def image():
    """Returns the latest image captured."""
    return send_file(latest_image, mimetype='image/jpeg')

if __name__ == '__main__':
    # Start the image capture thread
    capture_thread = Thread(target=capture_image)
    capture_thread.daemon = True
    capture_thread.start()

    # Run the Flask web server
    app.run(host='0.0.0.0', port=5000)