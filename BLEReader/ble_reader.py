import asyncio
from bleak import BleakScanner, BleakClient, BleakError

UNUSED = '00001801-0000-1000-8000-00805f9b34fb'

DEFAULT_UUID = '-0000-1000-8000-00805f9b34fb'

DEVICE_NAME = '00002a00' + DEFAULT_UUID  # READ

DEVICE_MAN = '00002a29' + DEFAULT_UUID  # READ
MODEL_N = '00002a24' + DEFAULT_UUID  # READ
SERIAL_N = '00002a25' + DEFAULT_UUID  # READ
HW_VERSION = '00002a27' + DEFAULT_UUID  # READ
FW_VERSION = '00002a26' + DEFAULT_UUID  # READ

BATTERY_LEVEL = '00002a19' + DEFAULT_UUID  # READ

ELECTRIC_FIELD = 'b92bcb8e-61a1-4059-9997-c915674ecda9'  # Status READ/NOTIFY

HUMIDITY = '5830147d-92d3-4dab-b1c6-f2452cc8a517'  # READ
ENV_NOISE = '9552b902-ff70-48ad-9ae0-fc4d8fae0c31'  # READ

ELEVATION = '44722079-6b5e-4c32-af7a-15536be7f3b3'  # READ
PRESSURE = 'd469a9f2-2952-4396-8655-6e8c828caaf9'  # READ
TEMPERATURE = '00002a6e' + DEFAULT_UUID  # READ

ACTIVITY = '656d0f29-09f7-4481-ad36-efb098ebd979'  # READ
POSE_ESTIM = '656d0f29-09f7-4481-ad36-efb098ebd982'  # READ
VERT_MOV = '656d0f29-09f7-4481-ad36-efb098ebd984'  # READ
FALL_DETECT = '04efc999-83d7-467b-836a-d5ea1e6c4447'  # READ


VALID_UUIDS = [DEVICE_NAME, DEVICE_MAN, MODEL_N, SERIAL_N, HW_VERSION,
               FW_VERSION, BATTERY_LEVEL, ELECTRIC_FIELD, HUMIDITY, ENV_NOISE,
               ELEVATION, PRESSURE, TEMPERATURE, ACTIVITY, POSE_ESTIM,
               VERT_MOV, FALL_DETECT]

UUIDS_STR = ['Device Name', 'Device Manufacturer', 'Model Number',
             'Serial Number', 'Hardware Revision', 'Firmware Revision',
             'Battery Level', 'Electric Field', 'Humidity',
             'Environment Noise', 'Elevation', 'Pressure',
             'Temperature', 'Activity', 'Pose Estimation',
             'Vertical Movement', 'Fall Detection']

CONVERSION = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 100, 1, 1, 1, 1, 1]


class Sefield():
    """
    Dispositivo BLE
    """

    def __init__(self):
        self.name = "SEFIELD"
        self.uuids = []
        self.address = ""
        self.found = False


async def discover_devices(desired_device: Sefield):
    """
    Encontra dispositivos disponíveis
    """
    devices = await BleakScanner.discover()
    print("Devices available:")
    for dev in devices:
        try:
            print(dev.address, dev.details['props']['Name'])

            if dev.details['props']['Name'] == desired_device.name:
                print("Found {0}\nProps. {1}: ".format(desired_device.name,
                                                       dev.details['props']))
                desired_device.uuids.extend(dev.details['props']['UUIDs'])
                desired_device.address = dev.address
                desired_device.found = True
                break

        except KeyError as key_error:
            print(key_error, "Not found", dev)


async def read_characteristics(device: Sefield):
    """
    Lê as características do dispositivo
    """
    async with BleakClient(device.address) as client:
        for uuid_name, uuid, conv in zip(UUIDS_STR, VALID_UUIDS, CONVERSION):
            try:
                raw_value = await client.read_gatt_char(uuid)
                value = "".join(map(chr, raw_value))

                if conv >= 1:
                    value = int.from_bytes(raw_value, "big")/conv
                print("{0}: {1}".format(uuid_name, value))

            except BleakError as ble_error:
                print("Error reading {0} = {1}\nError: {2}".format(uuid_name,
                                                                   uuid,
                                                                   ble_error))

if __name__ == "__main__":
    sefield_device = Sefield()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(discover_devices(sefield_device))

    cmd = ['', '1']
    user_cmd = '1'
    if sefield_device.found:
        while True:
            if user_cmd in cmd:
                print("")
                loop.run_until_complete(read_characteristics(sefield_device))
            else:
                break
            user_cmd = input("\n0) Exit\n1) Continue [enter]: ")
    else:
        print("{} not found".format(sefield_device.name))
