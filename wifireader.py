import time
import subprocess

def get_wifi_ssids():
    try:
        output = subprocess.check_output(["netsh", "wlan", "show", "network"]).decode("utf-8")
        ssids = [line.split(":")[1].strip() for line in output.splitlines() if "SSID" in line]
        return ssids
    except subprocess.CalledProcessError:
        return []

def construct_text_output(ssids):
    return " ".join(ssids)

if __name__ == "__main__":
    while True:
        ssids = get_wifi_ssids()
        text_output = construct_text_output(ssids)
        print(text_output)
        time.sleep(1)
