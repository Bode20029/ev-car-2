import serial
import struct
import time

serial_port = '/dev/ttyUSB0'
baud_rate = 9600

def read_pzem_v1(serial_connection):
    request_packet = b'\xB0\xC0\xA8\x01\x01\x00\x01\x5A'
    serial_connection.write(request_packet)
    print(f"Sent V1 request packet: {request_packet.hex()}")
    
    response = serial_connection.read(7)
    print(f"Received V1 response: {response.hex()}")
    
    if len(response) == 7:
        data = struct.unpack('>HBBH', response[1:])
        return {
            'voltage': data[0] / 10.0,
            'current': (data[1] << 8 | data[2]) / 100.0,
            'power': data[3] / 10.0
        }
    return None

def read_pzem_v3(serial_connection):
    request_packet = bytes([0xF8, 0x04, 0x00, 0x00, 0x00, 0x0A, 0x64, 0x64])
    serial_connection.write(request_packet)
    print(f"Sent V3 request packet: {request_packet.hex()}")
    
    response = serial_connection.read(25)
    print(f"Received V3 response: {response.hex()}")
    
    if len(response) == 25:
        voltage = int.from_bytes(response[3:5], byteorder='big') / 10.0
        current = int.from_bytes(response[5:7], byteorder='big') / 1000.0
        power = int.from_bytes(response[7:11], byteorder='big') / 10.0
        energy = int.from_bytes(response[11:15], byteorder='big')
        frequency = int.from_bytes(response[15:17], byteorder='big') / 10.0
        pf = int.from_bytes(response[17:19], byteorder='big') / 1000.0
        return {
            'voltage': voltage,
            'current': current,
            'power': power,
            'energy': energy,
            'frequency': frequency,
            'power_factor': pf
        }
    return None

def main():
    try:
        with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
            while True:
                print("\nTrying V1 protocol...")
                data_v1 = read_pzem_v1(ser)
                if data_v1:
                    print("V1 Protocol Succeeded:")
                    print(f"Voltage: {data_v1['voltage']} V")
                    print(f"Current: {data_v1['current']} A")
                    print(f"Power: {data_v1['power']} W")
                else:
                    print("V1 Protocol failed. Trying V3 protocol...")
                    data_v3 = read_pzem_v3(ser)
                    if data_v3:
                        print("V3 Protocol Succeeded:")
                        print(f"Voltage: {data_v3['voltage']} V")
                        print(f"Current: {data_v3['current']} A")
                        print(f"Power: {data_v3['power']} W")
                        print(f"Energy: {data_v3['energy']} Wh")
                        print(f"Frequency: {data_v3['frequency']} Hz")
                        print(f"Power Factor: {data_v3['power_factor']}")
                    else:
                        print("Both protocols failed to read data")
                
                time.sleep(5)
                
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("Program terminated by user")

if __name__ == '__main__':
    main()