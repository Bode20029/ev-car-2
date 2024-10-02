import serial
import time

# Configure the serial port
ser = serial.Serial(
    port='/dev/ttyUSB0',  # This may vary depending on your Jetson Nano setup
    baudrate=9600,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=1
)

def read_pzem_004t():
    # PZEM-004T command to read all electrical parameters
    command = bytes([0xF8, 0x04, 0x00, 0x00, 0x00, 0x0A, 0x64, 0x64])
    
    # Send command
    ser.write(command)
    
    # Read response
    response = ser.read(25)
    
    # Print raw bytes for debugging
    print("Raw response:", " ".join([f"{b:02X}" for b in response]))
    
    if len(response) != 25:
        print(f"Error: Received {len(response)} bytes instead of 25")
        return None
    
    if response[0] != 0xF8 or response[1] != 0x04:
        print(f"Error: Invalid response header: {response[0]:02X} {response[1]:02X}")
        return None
    
    voltage = int.from_bytes(response[3:5], byteorder='big') / 10.0  # V
    current = int.from_bytes(response[5:7], byteorder='big') / 1000.0  # A
    power = int.from_bytes(response[7:11], byteorder='big') / 10.0  # W
    energy = int.from_bytes(response[11:15], byteorder='big')  # Wh
    frequency_raw = int.from_bytes(response[15:17], byteorder='big')
    frequency = frequency_raw / 10.0  # Hz
    pf = int.from_bytes(response[17:19], byteorder='big') / 1000.0
    
    print(f"Frequency raw value: {frequency_raw}")
    
    return {
        'voltage': voltage,
        'current': current,
        'power': power,
        'energy': energy,
        'frequency': frequency,
        'power_factor': pf
    }

# Main execution
reading_count = 0
try:
    while True:
        reading_count += 1
        print(f"\nReading #{reading_count}")
        print("-------------------------")
        
        data = read_pzem_004t()
        if data:
            print(f"Voltage: {data['voltage']:.1f} V")
            print(f"Current: {data['current']:.3f} A")
            print(f"Power: {data['power']:.1f} W")
            print(f"Energy: {data['energy']} Wh")
            print(f"Frequency: {data['frequency']:.1f} Hz")
            print(f"Power Factor: {data['power_factor']:.3f}")
        else:
            print("Failed to read data")
        
        print("-------------------------")
        time.sleep(5)  # Wait for 5 seconds before next reading
except KeyboardInterrupt:
    print("\nProgram terminated by user")
except serial.SerialException as e:
    print(f"\nSerial port error: {e}")
finally:
    ser.close()
    print("Serial port closed")