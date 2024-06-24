import time
from scapy.all import *

def beacon(ssid):
    # Create a basic beacon frame with the specified SSID
    beacon_frame = RadioTap() / Dot11(type=0, subtype=8, addr1='ff:ff:ff:ff:ff:ff', addr2=RandMAC(), addr3=RandMAC()) / Dot11Beacon(cap='ESS') / Dot11Elt(ID='SSID', info=ssid, len=len(ssid))

    # Send the beacon frame
    sendp(beacon_frame, iface='Wi-Fi', loop=0, verbose=False)

def main():
    # Array of SSIDs
    ssids = ["6c6f6f703a20207365742e7461726765742873747265657473637261706a756e6b69652e6576696c282272616e646f6d22293b206c6f636b2e74617267657428293b207365742e766f6c756d6528223130302522293b207365742e706c617962", "61636b28706c61792e746f6e6528226472756d7322293b20676f746f286c6f6f702829293b202073747265657473637261706a756e6b69652e6576696c28646967657374286c6f6f702829293b2020656e746572206f6b7a796f6b20656e746572206f6b206f6b206f6b206f6b6f6b6f6b6f6b6f6b"]  # Add more SSIDs if needed

    # Loop through each SSID indefinitely
    while True:
        for ssid in ssids:
            print(f"Broadcasting beacon for SSID: {ssid}")
            beacon(ssid)
            time.sleep(1)  # Broadcast beacon for 5 seconds

if __name__ == "__main__":
    main()
