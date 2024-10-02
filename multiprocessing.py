import multiprocessing
import time
from hc_sr04p_distance import get_distance, GPIO_setup, GPIO_cleanup
from Updated_PZEM_Sensor_Reader_Script import connect_to_sensor, read_sensor_data

def distance_process():
    GPIO_setup()
    try:
        while True:
            distance = get_distance()
            if distance is not None:
                print(f"Distance: {distance:.2f} cm")
            time.sleep(1)
    finally:
        GPIO_cleanup()

def pzem_process():
    pzem_master = connect_to_sensor()
    try:
        while True:
            try:
                pzem_data = read_sensor_data(pzem_master)
                print(f"Current: {pzem_data['current_A']:.2f} A")
            except Exception as e:
                print(f"PZEM Error: {e}")
                pzem_master = connect_to_sensor()
            time.sleep(1)
    finally:
        if pzem_master:
            pzem_master.close()

if __name__ == "__main__":
    p1 = multiprocessing.Process(target=distance_process)
    p2 = multiprocessing.Process(target=pzem_process)

    p1.start()
    p2.start()

    try:
        p1.join()
        p2.join()
    except KeyboardInterrupt:
        print("Stopping processes...")
        p1.terminate()
        p2.terminate()
        p1.join()
        p2.join()
    print("Processes stopped")