import os
import winreg
import tkinter as tk
from PIL import Image, ImageTk
import sys
import datetime
import subprocess
import time
import schedule
import psutil
import random
import socket
import platform


class RedirectOutput:
    def __init__(self, callback):
        self.callback = callback

    def write(self, text):
        self.callback(text)


# Define DNS options
dns_servers = {
    "Cloudflare DNS": {
        "IPv4": "1.1.1.1,1.0.0.1",
        "IPv6": "2606:4700:4700::1111,2606:4700:4700::1001"
    },
    "Google Public DNS": {
        "IPv4": "8.8.8.8,8.8.4.4",
        "IPv6": "2001:4860:4860::8888,2001:4860:4860::8844"
    },
    "Quad9": {
        "IPv4": "9.9.9.9,149.112.112.112",
        "IPv6": "2620:fe::fe,2620:fe::9"
    },
    "OpenDNS (Cisco Umbrella)": {
        "IPv4": "208.67.222.222,208.67.220.220",
        "IPv6": "2620:119:35::35,2620:119:53::53"
    },
    "UncensoredDNS": {
        "IPv4": "91.239.100.100,89.233.43.71",
        "IPv6": "2a01:3a0:53:53::1,2a01:3a0:2a:53::1"
    },
    "Freenom World": {
        "IPv4": "80.80.80.80,80.80.81.81",
        "IPv6": "2606:4700:4700::1111,2606:4700:4700::1001"
    },
    "AdGuard DNS": {
        "IPv4": "94.140.14.14,94.140.15.15",
        "IPv6": "2a10:50c0::ad1:ff,2a10:50c0::ad2:ff"
    },
    "DNS.WATCH": {
        "IPv4": "84.200.69.80,84.200.70.40",
        "IPv6": "2001:1608:10:25::1c04:b12f,2001:1608:10:25::9249:d69b"
    },
    "Neustar UltraDNS": {
        "IPv4": "156.154.70.1,156.154.71.1",
        "IPv6": "2610:a1:1018::1,2610:a1:1019::1"
    },
    "Level3 (CenturyLink)": {
        "IPv4": "209.244.0.3,209.244.0.4",
        "IPv6": "2001:428::3,2001:428::4"
    }
}


# Select a random DNS pair
random_dns = random.choice(list(dns_servers.values()))

# Get the current platform
platform_name = platform.system()

# Apply DNS settings for IPv4
def apply_dns_ipv4(address):
    try:
        subprocess.run(["netsh", "interface", "ipv4", "set", "dns", "name=''", "source='static'", "address=" + address], check=True)
        print("IPv4 DNS set to:", address)
    except Exception as e:
        print("Error setting IPv4 DNS:", e)

# Apply DNS settings for IPv6
def apply_dns_ipv6(address):
    try:
        subprocess.run(["netsh", "interface", "ipv6", "set", "dns", "name=''", "source='static'", "address=" + address], check=True)
        print("IPv6 DNS set to:", address)
    except Exception as e:
        print("Error setting IPv6 DNS:", e)


def disable_unnecessary_services():
    try:
        # List of unnecessary Windows services to be disabled
        services_to_disable = [
            "wercplsupport",  # Problem Reports and Solutions Control Panel Support
            "DiagTrack",  # Connected User Experiences and Telemetry
            "WbioSrvc",  # Windows Biometric Service
            "WerSvc",  # Windows Error Reporting Service
            "tapisrv",  # Telephony
            "PeerDistSvc",  # BranchCache
            "TermService",  # Remote Desktop Services
            "LanmanWorkstation",  # Server service for SMB
            "WinRM",  # Windows Remote Management
            "W3SVC",  # Internet Information Services (IIS)
            "Winmgmt",  # Windows Management Instrumentation (WMI)
            "SSDPSRV",  # UPnP Device Host
            "LanmanServer"  # Windows File Sharing (SMB/CIFS)
            # Add more service names as needed
        ]
        
        for service in services_to_disable:
            # Stop the service if it is running
            os.system(f"sc stop {service}")
            
            # Disable the service
            os.system(f"sc config {service} start= disabled")
            
            print(f"Disabled and stopped {service} service.")
    except Exception as e:
        print(f"Error disabling unnecessary services: {e}")

def check_service_status(service_name):
    try:
        result = os.popen(f"sc query {service_name}").read()
        if "RUNNING" in result:
            print(f"Service '{service_name}' is running.")
        elif "STOPPED" in result:
            print(f"Service '{service_name}' is stopped. Starting...")
            os.system(f"sc start {service_name}")
            print(f"Service '{service_name}' started successfully.")
        else:
            print(f"Unable to determine the status of service '{service_name}'.")
    except Exception as e:
        print(f"Error while checking service '{service_name}' status: {e}")

def check_corenetworking_services_status():
    services = ["dot3svc", "dhcp", "dnscache", "nlasvc", "nlaSvc"]
    for service in services:
        check_service_status(service)

def disable_power_management(adapter_name):
    try:
        # Open the network adapter's registry key
        key_path = fr"SYSTEM\CurrentControlSet\Services\Tcpip\Parameters\Interfaces\{adapter_name}"
        command = f'reg add "HKEY_LOCAL_MACHINE\{key_path}" /v DisableTaskOffload /t REG_DWORD /d 1 /f'
        os.system(command)
    except Exception as e:
        print("Error disabling power management:", str(e))

def disable_wol(adapter_name):
    try:
        # Open the network adapter's registry key
        key_path = fr"SYSTEM\CurrentControlSet\Services\Tcpip\Parameters\Interfaces\{adapter_name}"
        command = f'reg add "HKEY_LOCAL_MACHINE\{key_path}" /v EnableWOL /t REG_DWORD /d 0 /f'
        os.system(command)
    except Exception as e:
        print("Error disabling Wake on LAN:", str(e))

def enable_firewall():
    try:
        os.system("powershell Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled True")
        print("Windows Firewall enabled.")
    except Exception as e:
        print("Error enabling Windows Firewall:", e)

def reset_firewall():
    try:
        # Reset Windows Firewall to default settings
        subprocess.run(["netsh", "advfirewall", "reset"])
        print("Windows Firewall reset to default.")
    except Exception as e:
        print("An error occurred while resetting Windows Firewall:", e)

def disable_file_and_print_sharing():
    try:
        os.system('reg add "HKLM\SYSTEM\CurrentControlSet\Control\Lsa" /v EveryoneIncludesAnonymous /t REG_DWORD /d 0 /f')
    except Exception as e:
        print("Error disabling File and Printer Sharing:", e)

def disable_ipv4_checksum_offload():
    try:
        os.system('reg add "HKLM\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters" /v DisableTaskOffload /t REG_DWORD /d 1 /f')
    except Exception as e:
        print("Error disabling IPv4 Checksum Offload:", e)

def disable_jumbo_packet():
    try:
        os.system('reg add "HKLM\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters\Interfaces" /v MTU /t REG_DWORD /d 1500 /f')
    except Exception as e:
        print("Error disabling Jumbo Packet:", e)

def enable_advanced_setting(setting_name):
    if setting_name == "advanced eee":
        try:
            os.system('reg add "HKLM\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters" /v EnergyEfficientEthernet /t REG_DWORD /d 1 /f')
        except Exception as e:
            print("Error enabling Advanced EEE:", e)
    elif setting_name == "ARP offload":
        try:
            os.system('reg add "HKLM\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters" /v ArpRetryCount /t REG_DWORD /d 0 /f')
        except Exception as e:
            print("Error disabling ARP offload:", e)
    elif setting_name == "auto disable gigabit":
        try:
            os.system('reg add "HKLM\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters" /v DisableTaskOffload /t REG_DWORD /d 0 /f')
        except Exception as e:
            print("Error disabling auto-disable gigabit:", e)
    # Add similar elif blocks for other settings...

def set_windows_analytics_settings():
    try:
        # Open the registry key for Windows Analytics settings
        key_path = r"Software\Policies\Microsoft\Windows\DataCollection"
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_WRITE)

        # Set the telemetry level to 0 (Security)
        telemetry_level_value = 0
        winreg.SetValueEx(key, "AllowTelemetry", 0, winreg.REG_DWORD, telemetry_level_value)

        # Close the registry key
        winreg.CloseKey(key)
        
        print("Windows Analytics settings minimized successfully.")
    except Exception as e:
        print("An error occurred:", str(e))

def set_dns_servers(dns_servers):
    # Select a random DNS server address pair
    selected_dns = random.choice(list(dns_servers.values()))
    
    # Define PowerShell script for setting DNS servers
    powershell_script = f"""
$dnsServers = "{selected_dns['IPv4']}", "{selected_dns['IPv6']}"
$adapters = Get-NetAdapter | Where-Object {{ $_.Status -eq "Up" }}
$successCount = 0

foreach ($adapter in $adapters) {{
    $interfaceName = $adapter.Name
    Set-DnsClientServerAddress -InterfaceAlias $interfaceName -ServerAddresses $dnsServers
    $successCount++
}}

if ($successCount -gt 0) {{
    Write-Host "DNS servers set successfully for all adapters."
}} else {{
    Write-Host "Failed to set DNS servers."
}}
"""

    # Execute PowerShell script
    result = subprocess.run(["powershell", "-Command", powershell_script], capture_output=True, text=True)
    output = result.stdout.strip()
    print(output)
        
def disable_advanced_setting(setting_name):
    if setting_name == "advanced eee":
        try:
            os.system('reg add "HKLM\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters" /v EnergyEfficientEthernet /t REG_DWORD /d 0 /f')
        except Exception as e:
            print("Error disabling Advanced EEE:", e)
    elif setting_name == "ARP offload":
        try:
            os.system('reg add "HKLM\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters" /v ArpRetryCount /t REG_DWORD /d 3 /f')
        except Exception as e:
            print("Error enabling ARP offload:", e)
    elif setting_name == "auto disable gigabit":
        try:
            os.system('reg add "HKLM\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters" /v DisableTaskOffload /t REG_DWORD /d 1 /f')
        except Exception as e:
            print("Error enabling auto-disable gigabit:", e)
    # Add similar elif blocks for other settings...

def configure_network_adapters():
    # Get all network adapters
    adapters = os.popen("wmic nicconfig get description").read().split("\n")[1:]
    
    for adapter in adapters:
        adapter = adapter.strip()
        if adapter and not adapter.startswith("Virtual"):
            print("Configuring adapter:", adapter)

            # Additional settings for balanced cybersecurity and user experience
            try:

                
                # Enable Windows Firewall
    
                
                # Disable Power Management
                disable_power_management(adapter)

                # Disable Wake on LAN
                disable_wol(adapter)

                # Disable File and Printer Sharing for Microsoft Networks
                disable_file_and_print_sharing()

                # Enable IPv6 Privacy Extensions via registry
                disable_ipv4_checksum_offload()
                disable_jumbo_packet()
                enable_advanced_setting("advanced eee")
                disable_advanced_setting("ARP offload")
                disable_advanced_setting("auto disable gigabit")

                print("Configuration applied successfully.")
            except Exception as e:
                print("Error configuring additional settings:", str(e))

def activate_privacy_shield():
    configure_network_adapters()
    reset_firewall()
    enable_firewall()
    check_corenetworking_services_status()
    disable_unnecessary_services()
    set_windows_analytics_settings()
    set_dns_servers(dns_servers)
    print("Privacy Shield Activated")
    # Display label for 10 seconds
    label.config(text="System Privacy Shield Activated")
    label_after_id = label.after(10000, lambda: label.config(text=""))  # Clear label after 10 seconds
    # Load and display green check image
    green_check_image = Image.open("green_check.png")
    green_check_image = green_check_image.resize((120, 120), Image.BICUBIC)  # Use Image.BICUBIC instead of Image.ANTIALIAS
    green_check_photo = ImageTk.PhotoImage(green_check_image)
    green_check_label = tk.Label(root, image=green_check_photo)
    green_check_label.image = green_check_photo  # Keep a reference to avoid garbage collection
    green_check_label.pack()

    # Cancel the label clear operation if it's still pending
    label.after_cancel(label_after_id)

# Function to update textbox with serial output
def update_textbox(output):
    textbox.config(state=tk.NORMAL)
    textbox.insert(tk.END, output)
    textbox.config(state=tk.DISABLED)
    textbox.see(tk.END)  # Scroll to the end

# Create Tkinter window
root = tk.Tk()
root.title("Privacy Shield Activation")
root.geometry("800x600")
root.resizable(False, False)

# Create button to activate privacy shield
button = tk.Button(root, text="Activate Privacy Shield", command=activate_privacy_shield)
button.pack(pady=20)

# Create label to display status
label = tk.Label(root, text="", font=("Helvetica", 16))
label.place(relx=0.5, rely=0.4, anchor="center")

# Create textbox for serial output
textbox = tk.Text(root, height=10, state=tk.DISABLED)
textbox.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

# Redirect stdout to update textbox
sys.stdout = RedirectOutput(update_textbox)

root.mainloop()
