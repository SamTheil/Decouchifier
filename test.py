from flask import Flask, Response, send_file
from time import sleep
from io import BytesIO
from picamera import PiCamera
from threading import Thread

app = Flask(__name__)
camera = PiCamera()
camera.resolution = (640, 480)

# Variable to store the latest image in memory
latest_image = BytesIO()

def capture_image():
    global latest_image
    while True:
        stream = BytesIO()
        camera.capture(stream, format='jpeg')
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
            <img src="/image.jpg" width="640" height="480"/>
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