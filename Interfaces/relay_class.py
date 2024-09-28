import RPi.GPIO as GPIO
import time

class RelayClass():
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

if __name__ == "__main__":
    # Create an instance of the relay class
    relay = RelayClass()

    try:
        # Loop indefinitely, turning the relay on and off every 2 seconds
        while True:
            relay.turn_on_relay()
            time.sleep(2)  # Wait for 2 seconds
            relay.turn_off_relay()
            time.sleep(2)  # Wait for 2 seconds
    except KeyboardInterrupt:
        # Catch the KeyboardInterrupt (Ctrl+C) to safely exit
        print("Exiting and cleaning up GPIO settings")
    finally:
        # Ensure cleanup is always called on exit
        relay.cleanup()