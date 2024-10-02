import asyncio
import time
from hc_sr04p_distance import get_distance, GPIO_setup, GPIO_cleanup
from Updated_PZEM_Sensor_Reader_Script import connect_to_sensor, read_sensor_data

async def distance_coroutine():
    GPIO_setup()
    try:
        while True:
            distance = get_distance()
            if distance is not None:
                print(f"Distance: {distance:.2f} cm")
            await asyncio.sleep(1)
    finally:
        GPIO_cleanup()

async def pzem_coroutine():
    pzem_master = connect_to_sensor()
    try:
        while True:
            try:
                pzem_data = read_sensor_data(pzem_master)
                print(f"Current: {pzem_data['current_A']:.2f} A")
            except Exception as e:
                print(f"PZEM Error: {e}")
                pzem_master = connect_to_sensor()
            await asyncio.sleep(1)
    finally:
        if pzem_master:
            pzem_master.close()

async def main():
    task1 = asyncio.create_task(distance_coroutine())
    task2 = asyncio.create_task(pzem_coroutine())
    await asyncio.gather(task1, task2)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopping coroutines...")
    print("Coroutines stopped")