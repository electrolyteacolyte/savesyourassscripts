import subprocess
import wmi

def disable_power_management(adapter_name):
    try:
        # Execute netsh command to disable power management
        subprocess.run(["netsh", "interface", "ipv4", "set", "interface", adapter_name, "taskoffload=disabled"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error disabling power management:", e)

def disable_wol(adapter_name):
    try:
        # Execute netsh command to disable Wake on LAN
        subprocess.run(["netsh", "int", "ipv4", "set", "interface", adapter_name, "wol=disabled"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error disabling Wake on LAN:", e)

def enable_windows_firewall():
    try:
        # Enable Windows Firewall
        subprocess.run(["netsh", "advfirewall", "set", "allprofiles", "state", "on"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error enabling Windows Firewall:", e)

def enable_remote_desktop():
    try:
        # Enable Remote Desktop
        subprocess.run(["reg", "add", "HKLM\SYSTEM\CurrentControlSet\Control\Terminal Server", "/v", "fDenyTSConnections", "/t", "REG_DWORD", "/d", "0", "/f"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error enabling Remote Desktop:", e)

def disable_file_and_print_sharing():
    try:
        # Execute netsh command to disable file and print sharing
        subprocess.run(["netsh", "advfirewall", "firewall", "set", "rule", "group=\"File and Printer Sharing\"", "new", "enable=no"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error disabling File and Printer Sharing:", e)

def configure_network_adapters():
    # Initialize WMI
    c = wmi.WMI()

    # Get all network adapters
    adapters = c.Win32_NetworkAdapterConfiguration()

    for adapter in adapters:
        # Check if the adapter is enabled and is not a virtual adapter
        if adapter.IPEnabled and not adapter.Description.startswith("Virtual"):
            print("Configuring adapter:", adapter.Description)

            # Additional settings for balanced cybersecurity and user experience
            try:
                # Disable Power Management
                disable_power_management(adapter.Description)

                # Disable Wake on LAN
                disable_wol(adapter.Description)

                # Enable Windows Firewall
                enable_windows_firewall()

                # Enable Remote Desktop
                enable_remote_desktop()

                # Disable File and Printer Sharing for Microsoft Networks
                disable_file_and_print_sharing()

                print("Configuration applied successfully.")
            except Exception as e:
                print("Error configuring additional settings:", str(e))

if __name__ == "__main__":
    configure_network_adapters()
