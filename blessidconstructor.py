import time
import asyncio
import bleak

async def scan_ble_devices():
    scanner = bleak.BleakScanner()
    await scanner.start()
    devices = await scanner.discover()
    await scanner.stop()
    return devices

def construct_text_output(devices):
    output = ""
    for device in devices:
        output += f"Address: {device.address}\n"
        output += f"Name: {device.name if device.name else 'Unknown'}\n"
        advertisement_data = device.metadata.get("advertisement_data", {})
        output += f"RSSI: {advertisement_data.get('rssi', 'Unknown')} dBm\n"
        for key, value in advertisement_data.items():
            if key != "rssi":
                output += f"{key}: {value}\n"
        output += "\n"
    return output

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    while True:
        devices = loop.run_until_complete(scan_ble_devices())
        text_output = construct_text_output(devices)
        print(text_output)
        time.sleep(1)
