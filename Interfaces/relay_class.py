import RPi.GPIO as GPIO

class relay_class():
    def __init__(self):
        # Set the GPIO mode
        GPIO.setmode(GPIO.BCM)
        # Set GPIO 23 as an output pin
        GPIO.setup(23, GPIO.OUT)
    
    def turn_on_relay(self):
        # Set GPIO 23 high to turn the relay on
        GPIO.output(23, GPIO.HIGH)
        print("Relay turned ON")
    
    def turn_off_relay(self):
        # Set GPIO 23 low to turn the relay off
        GPIO.output(23, GPIO.LOW)
        print("Relay turned OFF")
    
    def cleanup(self):
        # Clean up GPIO settings when done
        GPIO.cleanup()