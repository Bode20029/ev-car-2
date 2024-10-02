import serial
import time
import binascii

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
    command = bytes([0xF8, 0x04, 0x00, 0x00, 0x00, 0x0A, 0x64, 0x64])
    
    ser.write(command)
    response = ser.read(25)
    
    print(f"Full response hex: {binascii.hexlify(response).decode()}")
    
    if len(response) == 25 and response[0] == 0xF8 and response[1] == 0x04:
        voltage = int.from_bytes(response[3:5], byteorder='big') / 10.0
        current = int.from_bytes(response[5:7], byteorder='big') / 1000.0
        power = int.from_bytes(response[7:11], byteorder='big') / 10.0
        energy = int.from_bytes(response[11:15], byteorder='big')
        
        # Debugging frequency
        freq_bytes = response[15:17]
        freq_raw = int.from_bytes(freq_bytes, byteorder='big')
        frequency = freq_raw / 10.0
        print(f"Raw frequency bytes: {freq_bytes.hex()}, Raw value: {freq_raw}")
        
        # Debugging power factor
        pf_bytes = response[17:19]
        pf_raw = int.from_bytes(pf_bytes, byteorder='big')
        print(f"Raw PF bytes: {pf_bytes.hex()}, Raw PF value: {pf_raw}")
        
        pf_1000 = pf_raw / 1000.0
        pf_100 = pf_raw / 100.0
        
        return {
            'voltage': voltage,
            'current': current,
            'power': power,
            'energy': energy,
            'frequency': frequency,
            'frequency_raw': freq_raw,
            'power_factor_1000': pf_1000,
            'power_factor_100': pf_100,
            'power_factor_raw': pf_raw
        }
    else:
        print(f"Unexpected response length: {len(response)}")
        return None

def main():
    try:
        while True:
            data = read_pzem_004t()
            if data:
                print(f"Voltage: {data['voltage']:.1f} V")
                print(f"Current: {data['current']:.3f} A")
                print(f"Power: {data['power']:.1f} W")
                print(f"Energy: {data['energy']} Wh")
                print(f"Frequency: {data['frequency']:.1f} Hz (Raw: {data['frequency_raw']})")
                print(f"Power Factor (/1000): {data['power_factor_1000']:.3f}")
                print(f"Power Factor (/100): {data['power_factor_100']:.3f}")
                print(f"Power Factor (raw): {data['power_factor_raw']}")
                
                # Calculate PF from P, V, I
                calculated_pf = data['power'] / (data['voltage'] * data['current']) if data['voltage'] * data['current'] != 0 else 0
                print(f"Calculated PF: {calculated_pf:.3f}")
                
                print("-------------------------")
            else:
                print("Failed to read data")
            
            time.sleep(5)

    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        ser.close()
        print("Serial port closed")

if __name__ == '__main__':
    main()