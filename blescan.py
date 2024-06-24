import asyncio
from datetime import datetime
import bleak

async def scan_and_log():
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Perform BLE scan
    scanner = bleak.BleakScanner()
    devices = await scanner.discover()

    # Log the scan details into a text file
    with open('ble_scan_log.txt', 'a') as log_file:
        log_file.write(f"Scan Time: {timestamp}\n")
        log_file.write("---------------------------------------------------\n")
        for device in devices:
            log_file.write(f"Device Name: {device.name}\n")
            log_file.write(f"Address: {device.address}\n")
            log_file.write("---------------------------------------------------\n")
        log_file.write("\n")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(scan_and_log())
