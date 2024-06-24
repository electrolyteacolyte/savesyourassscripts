import subprocess
import wmi
import winreg

def disable_power_management(adapter_name):
    try:
        # Open the network adapter's registry key
        key_path = fr"SYSTEM\CurrentControlSet\Services\Tcpip\Parameters\Interfaces\{adapter_name}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_WRITE) as key:
            # Set the DisableTaskOffload registry value to 1 (disabled)
            winreg.SetValueEx(key, "DisableTaskOffload", 0, winreg.REG_DWORD, 1)
    except FileNotFoundError:
        print("Registry key not found. Power management may already be disabled.")
    except Exception as e:
        print("Error disabling power management:", str(e))

def disable_wol(adapter_name):
    try:
        # Open the network adapter's registry key
        key_path = fr"SYSTEM\CurrentControlSet\Services\Tcpip\Parameters\Interfaces\{adapter_name}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_WRITE) as key:
            # Set the EnableWOL registry value to 0 (disabled)
            winreg.SetValueEx(key, "EnableWOL", 0, winreg.REG_DWORD, 0)
    except FileNotFoundError:
        print("Registry key not found. Wake on LAN may already be disabled.")
    except Exception as e:
        print("Error disabling Wake on LAN:", str(e))

def enable_firewall():
    try:
        # Enable Windows Firewall using PowerShell
        subprocess.run(["powershell", "Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled True"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error enabling Windows Firewall:", e)

def enable_remote_desktop():
    try:
        # Enable Remote Desktop using PowerShell
        subprocess.run(["powershell", "Set-ItemProperty -Path 'HKLM:\System\CurrentControlSet\Control\Terminal Server' -Name 'fDenyTSConnections' -Value 0"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error enabling Remote Desktop:", e)

def disable_file_and_print_sharing():
    try:
        # Disable File and Printer Sharing using registry modification
        subprocess.run(["reg", "add", "HKLM\SYSTEM\CurrentControlSet\Control\Lsa", "/v", "EveryoneIncludesAnonymous", "/t", "REG_DWORD", "/d", "0", "/f"], check=True)
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
                enable_firewall()

                # Enable Remote Desktop
                enable_remote_desktop()

                # Disable File and Printer Sharing for Microsoft Networks
                disable_file_and_print_sharing()

                # Enable IPv6 Privacy Extensions via registry
                subprocess.run(["reg", "add", fr"HKLM\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters\Interfaces\{adapter.SettingID}", "/v", "UseTemporaryAddresses", "/t", "REG_DWORD", "/d", "1", "/f"], check=True)

                print("Configuration applied successfully.")
            except Exception as e:
                print("Error configuring additional settings:", str(e))

if __name__ == "__main__":
    configure_network_adapters()
