import bluetooth

def find_bluetooth_speaker():
    nearby_devices = bluetooth.discover_devices(lookup_names=True)
    for addr, name in nearby_devices:
        if "Speaker" in name:  # Adjust based on your speaker's name
            return addr
    return None

def connect_to_speaker(address):
    sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    sock.connect((address, 1))
    return sock