import time
from scapy.all import *

def beacon(ssid):
    # Create a basic beacon frame with the specified SSID
    beacon_frame = RadioTap() / Dot11(type=0, subtype=8, addr1='ff:ff:ff:ff:ff:ff', addr2=RandMAC(), addr3=RandMAC()) / Dot11Beacon(cap='ESS') / Dot11Elt(ID='SSID', info=ssid, len=len(ssid))

    # Send the beacon frame
    sendp(beacon_frame, iface='Wi-Fi', loop=0, verbose=False)

def main():
    # Array of SSIDs
    ssids = ["Hi, with, this, you, can, send, data, through, wifi, without, connecting, as, a, raw, text, if, my, receiver, is, running, "]  # Add more SSIDs if needed

    # Loop through each SSID indefinitely
    while True:
        for ssid in ssids:
            print(f"Broadcasting beacon for SSID: {ssid}")
            beacon(ssid)
            time.sleep(2)  # Broadcast beacon for 5 seconds

if __name__ == "__main__":
    main()
