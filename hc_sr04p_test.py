import time
import logging
from hc_sr04p_distance import get_distance, GPIO_setup, GPIO_cleanup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    GPIO_setup()
    try:
        while True:
            distance = get_distance()
            if distance is not None:
                logger.info(f"Measured Distance = {distance:.2f} cm")
            else:
                logger.warning("Failed to get distance measurement")
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Measurement stopped by user")
    finally:
        GPIO_cleanup()

if __name__ == "__main__":
    main()