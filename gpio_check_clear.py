import Jetson.GPIO as GPIO
import os

def check_gpio_state(pin):
    gpio_path = f"/sys/class/gpio/gpio{pin}"
    if os.path.exists(gpio_path):
        with open(f"{gpio_path}/direction", "r") as f:
            direction = f.read().strip()
        print(f"GPIO {pin} is exported and set as {direction}")
        return True
    else:
        print(f"GPIO {pin} is not exported")
        return False

def clear_gpio(pin):
    try:
        with open("/sys/class/gpio/unexport", "w") as f:
            f.write(str(pin))
        print(f"GPIO {pin} has been unexported")
    except Exception as e:
        print(f"Error unexporting GPIO {pin}: {e}")

def main():
    # List of GPIO pins you want to check (adjust as needed)
    pins_to_check = [12, 16]  # Add more pin numbers as needed

    for pin in pins_to_check:
        if check_gpio_state(pin):
            clear_gpio(pin)

    # Final cleanup
    GPIO.cleanup()
    print("GPIO cleanup completed")

if __name__ == "__main__":
    main()