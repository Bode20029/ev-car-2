import serial
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu
import time

# Connect to the sensor
sensor = serial.Serial(
    port='/dev/ttyUSB0',
    baudrate=9600,
    bytesize=8,
    parity='N',
    stopbits=1,
    xonxoff=0
)

master = modbus_rtu.RtuMaster(sensor)
master.set_timeout(2.0)
master.set_verbose(True)

# Counter to indicate reading number
counter = 1

try:
    while True:
        # Get data from the sensor
        data = master.execute(1, cst.READ_INPUT_REGISTERS, 0, 10)

        voltage = data[0] / 10.0  # [V]
        current = (data[1] + (data[2] << 16)) / 1000.0  # [A]
        power = (data[3] + (data[4] << 16)) / 10.0  # [W]
        energy = data[5] + (data[6] << 16)  # [Wh]
        frequency = data[7] / 10.0  # [Hz]
        powerFactor = data[8] / 100.0
        alarm = data[9]  # 0 = no alarm

        # Print readings along with the counter
        print(f'Reading {counter}:')
        print('Voltage [V]: ', voltage)
        print('Current [A]: ', current)
        print('Power [W]: ', power)  # active power (V * I * power factor)
        print('Energy [Wh]: ', energy)
        print('Frequency [Hz]: ', frequency)
        print('Power factor []: ', powerFactor)
        print('Alarm : ', alarm)
        print('------------------------')

        # Increment the counter and wait for 5 seconds before the next reading
        counter += 1
        time.sleep(5)

except KeyboardInterrupt:
    print('Stopping sensor readings.')

finally:
    # Close connections gracefully
    try:
        master.close()
        if sensor.is_open:
            sensor.close()
    except:
        pass
