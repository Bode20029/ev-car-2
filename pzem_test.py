import time
import logging
from Updated_PZEM_Sensor_Reader_Script import connect_to_sensor, read_sensor_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    pzem_master = connect_to_sensor()
    try:
        while True:
            try:
                pzem_data = read_sensor_data(pzem_master)
                logger.info(f"Current: {pzem_data['current_A']:.2f} A, Voltage: {pzem_data['voltage']:.1f} V")
            except Exception as e:
                logger.error(f"Error reading PZEM data: {e}")
                pzem_master = connect_to_sensor()  # Reconnect
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Measurement stopped by user")
    finally:
        if pzem_master:
            pzem_master.close()

if __name__ == "__main__":
    main()