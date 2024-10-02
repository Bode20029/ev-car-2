import time
import json
import serial
import logging
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable to store the last energy reading
last_energy_reading = 0

def connect_to_sensor():
    while True:
        try:
            ser = serial.Serial(
                port='/dev/ttyUSB0',
                baudrate=9600,
                bytesize=8,
                parity='N',
                stopbits=1,
                xonxoff=0
            )
            master = modbus_rtu.RtuMaster(ser)
            master.set_timeout(2.0)
            master.set_verbose(True)
            logger.info("Successfully connected to the sensor")
            return master
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            time.sleep(5)  # Wait before retrying

def read_sensor_data(master):
    global last_energy_reading
    data = master.execute(1, cst.READ_INPUT_REGISTERS, 0, 10)
    voltage = data[0] / 10.0
    current = (data[1] + (data[2] << 16)) / 1000.0
    power = (data[3] + (data[4] << 16)) / 10.0
    
    # Check if current is close to zero (allowing for small measurement errors)
    if abs(current) == 0:  # Threshold of 0.01A
        power = 0  # Set power to 0 if current is effectively zero
    
    current_energy = data[5] + (data[6] << 16)
    energy_delta = current_energy - last_energy_reading if last_energy_reading != 0 else 0
    last_energy_reading = current_energy
    
    dict_payload = {
        "voltage": voltage,
        "current_A": current,
        "power_W": power,
        "energy_Wh": current_energy,
        "energy_delta_Wh": energy_delta,
        "frequency_Hz": data[7] / 10.0,
        "power_factor": data[8] / 100.0,
        "alarm": data[9],
        "calculated_power_W": voltage * current  # Add calculated power for comparison
    }
    return dict_payload

def main():
    master = None
    reading_number = 1
    try:
        master = connect_to_sensor()
        while True:
            try:
                dict_payload = read_sensor_data(master)
                dict_payload["reading_number"] = reading_number
                str_payload = json.dumps(dict_payload, indent=2)
                print(f"Reading #{reading_number}")
                print(str_payload)
                reading_number += 1
            except Exception as e:
                logger.error(f"Error reading data: {e}")
                master.close()
                master = connect_to_sensor()  # Reconnect
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info('Exiting PZEM script')
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
    finally:
        if master:
            master.close()

if __name__ == "__main__":
    main()