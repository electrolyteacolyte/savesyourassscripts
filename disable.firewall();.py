import subprocess
import time

def disable_firewall():
    try:
        subprocess.run(["netsh", "advfirewall", "set", "allprofiles", "state", "off"], check=True)
        print("Firewall disabled successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to disable firewall: {e}")

def run_elevate_vbs():
    try:
        subprocess.run(["cscript", "elevate.vbs"], check=True)
        print("Elevate script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to run elevate script: {e}")
        
if __name__ == "__main__":
    run_elevate_vbs()
    time.sleep(10)
    disable_firewall()
    