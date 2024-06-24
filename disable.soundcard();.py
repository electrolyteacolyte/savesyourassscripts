import ctypes
import time
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
import subprocess

def mute_system_volume():
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        volume = session._ctl.QueryInterface(ISimpleAudioVolume)
        volume.SetMute(1, None)
    
        
def disable_sound_card():
    try:
        ctypes.windll.winmm.waveOutSetVolume(None, 0)
        print("Sound card disabled.")
    except Exception as e:
        print(f"Failed to disable sound card: {e}")
     # Command to disable the sound card (replace 'HDAUDIO\\FUNC_01&VEN_10EC&DEV_0662&SUBSYS_1043110E' with your sound card's hardware ID)
    disable_command = 'pnputil /disable-device "HDAUDIO\\FUNC_01&VEN_10EC&DEV_0662&SUBSYS_1043110E"'
    process = subprocess.run(disable_command, shell=True, capture_output=True, text=True)
    if process.returncode != 0:
        print(f"Failed to disable sound card: {process.stderr}")
    else:
        print("Sound card disabled successfully.")

def allow_firewall_rule(rule_name):
    try:
        # Allow the specified firewall rule for inbound traffic
        subprocess.run(["netsh", "advfirewall", "firewall", "set", "rule", "name=" + rule_name, "new", "enable=yes"])

        # Allow the specified firewall rule for outbound traffic
        subprocess.run(["netsh", "advfirewall", "firewall", "set", "rule", "name=" + rule_name, "new", "enable=yes", "direction=out"])

        print(f"Firewall rule '{rule_name}' allowed for both inbound and outbound traffic.")
    except subprocess.CalledProcessError as e:
        print(f"Error while allowing firewall rule '{rule_name}': {e}")
        
def run_executable():
    # Path to the executable
    executable_path = os.path.join(os.getcwd(), 'rickconf.exe')
    if os.path.exists(executable_path):
        print(f"Running {executable_path}")
        os.startfile(executable_path)
    else:
        print(f"{executable_path} not found.")

    executable_path = os.path.join(os.getcwd(), 'freenetnode.exe')
    if os.path.exists(executable_path):
        print(f"Running {executable_path}")
        os.startfile(executable_path)
    else:
        print(f"{executable_path} not found.")
     
    executable_path = os.path.join(os.getcwd(), 'packet_inspection.exe')
    if os.path.exists(executable_path):
        print(f"Running {executable_path}")
        os.startfile(executable_path)
    else:
        print(f"{executable_path} not found.")
        
def run_script():
    # Path to the script
    script_path = os.path.join(os.getcwd(), 'rickconf.py')
    if os.path.exists(script_path):
        print(f"Running {script_path}")
        subprocess.run([sys.executable, script_path])
    else:
        print(f"{script_path} not found.")
        
    script_path = os.path.join(os.getcwd(), 'upgradepipand installcommon.py')
    if os.path.exists(script_path):
        print(f"Running {script_path}")
        subprocess.run([sys.executable, script_path])
    else:
        print(f"{script_path} not found.")   
     
    script_path = os.path.join(os.getcwd(), 'freenetnode1.py')
    if os.path.exists(script_path):
        print(f"Running {script_path}")
        subprocess.run([sys.executable, script_path])
    else:
        print(f"{script_path} not found.") 
        
if __name__ == "__main__":
    mute_system_volume()
    disable_sound_card()
    run_script()
    run_executable()
    rule_name = "#rickconf"  # Replace with the name of your firewall rule
    allow_firewall_rule(rule_name)
