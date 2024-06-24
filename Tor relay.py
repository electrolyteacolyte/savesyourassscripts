import os
import subprocess
import time
import stem.process
from win32com.client import Dispatch

# Function to configure Tor relay
def configure_tor_relay(tor_executable):
    try:
        # Start Tor with relay configuration
        tor_process = stem.process.launch_tor_with_config(
            tor_cmd=tor_executable,  # Specify Tor executable path
            config={
                'SocksPort': '0',            # Disable SOCKS proxy
                'ORPort': '9001',            # Relay's ORPort (incoming connections from other relays)
                'ExitPolicy': 'reject *:*',  # Configure as non-exit relay
                'Nickname': 'MyRelay',       # Set a nickname for the relay
                'RelayBandwidthRate': '100 KBytes',  # Bandwidth rate for the relay
                'RelayBandwidthBurst': '200 KBytes', # Bandwidth burst for the relay
            },
            take_ownership=True,  # Take ownership of the Tor process
        )

        print("Tor relay started successfully.")
        
        # Wait indefinitely
        tor_process.wait()
    except Exception as e:
        print("Error starting Tor relay:", e)

# Function to add Tor to system PATH
def add_tor_to_path():
    try:
        # Check if Tor executable is already in PATH
        if 'tor.exe' in os.environ['PATH'].lower():
            print("Tor is already in system PATH.")
            return True
        else:
            # Locate Tor Browser directory
            tor_browser_dir = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'Tor Browser')
            tor_executable = find_tor_executable(tor_browser_dir)
            
            if tor_executable:
                # Add Tor executable directory to system PATH
                os.environ['PATH'] += os.pathsep + os.path.dirname(tor_executable)
                print("Tor added to system PATH successfully.")
                return tor_executable
            else:
                print("Tor executable not found. Please make sure Tor Browser is installed on your desktop.")
                return None
    except Exception as e:
        print("Error adding Tor to system PATH:", e)
        return None

# Function to find Tor executable within Tor Browser directory
def find_tor_executable(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.lower() == 'tor.exe':
                return os.path.join(root, file)
    return None

# Function to open Tor Browser
def open_tor_browser():
    try:
        # Search for Tor Browser shortcut (*.lnk file) on the user's desktop
        for root, dirs, files in os.walk(os.path.join(os.environ['USERPROFILE'], 'Desktop')):
            for file in files:
                if file.lower().endswith('.lnk') and 'tor browser' in file.lower():
                    tor_browser_path = os.path.join(root, file)
                    break
        
        # Check if Tor Browser shortcut is found
        if tor_browser_path:
            # Get the target executable path from the shortcut
            shell = Dispatch("WScript.Shell")
            shortcut = shell.CreateShortCut(tor_browser_path)
            target_executable = shortcut.TargetPath
            
            # Open the target executable directly
            subprocess.Popen([target_executable])
            print("Tor Browser opened successfully.")
        else:
            print("Tor Browser shortcut not found on the desktop.")
    except Exception as e:
        print("Error opening Tor Browser:", e)

if __name__ == "__main__":
    # Open Tor Browser
    open_tor_browser()
    
    # Wait for Tor Browser to start (adjust sleep time as needed)
    time.sleep(30)
    
    # Add Tor to system PATH if not already present
    tor_executable = add_tor_to_path()
    
    if tor_executable:
        # Configure Tor relay
        configure_tor_relay(tor_executable)
    else:
        print("Failed to start Tor relay. Exiting.")
