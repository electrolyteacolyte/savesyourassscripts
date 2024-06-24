import wmi
import requests
import subprocess
import os
import winreg
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# SQLAlchemy setup
Base = declarative_base()

class Device(Base):
    __tablename__ = 'devices'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    manufacturer = Column(String)
    hardware_id = Column(String, unique=True)

class InstalledDriver(Base):
    __tablename__ = 'installed_drivers'

    id = Column(Integer, primary_key=True)
    hardware_id = Column(String, unique=True)
    version = Column(String)
    latest = Column(Boolean)

class DriverInstallationReceipt:
    def __init__(self, filename='driver_installation_receipt.txt'):
        self.filename = filename

    def add_device_info(self, device_info):
        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(f"Device Name: {device_info['Name']}\n")
            f.write(f"Manufacturer: {device_info['Manufacturer']}\n")
            hardware_ids = device_info['HardwareID']
            if hardware_ids:
                f.write(f"Hardware ID: {', '.join([str(id) for id in hardware_ids])}\n")
            else:
                f.write("No Hardware ID available\n")
            f.write("\n")

    def add_driver_installation(self, driver_url):
        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(f"Driver installed: {driver_url}\n")

class DriverInstaller:
    def __init__(self, engine):
        self.engine = engine
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def get_device_info(self):
        print("Getting device information...")
        c = wmi.WMI()
        devices = c.Win32_PnPEntity()
        device_info = []
        for device in devices:
            hardware_ids = device.HardwareID if device.HardwareID else []
            device_info.append({
                "Name": device.Name,
                "Manufacturer": device.Manufacturer,
                "HardwareID": hardware_ids,
            })
        print("Device information retrieved.")
        return device_info

    def search_for_driver(self, hardware_id):
        # Simulated driver URLs, replace with real sources
        driver_urls = {
            'PCI\\VEN_10DE&DEV_1C03&SUBSYS_859C1043&REV_A1': 'https://example.com/nvidia_driver',
            'PCI\\VEN_8086&DEV_10EA&SUBSYS_00011179&REV_05': 'https://example.com/intel_driver',
        }
        return driver_urls.get(hardware_id)

    def download_driver(self, url, destination):
        print(f"Downloading driver from {url}...")
        response = requests.get(url)
        with open(destination, 'wb') as f:
            f.write(response.content)
        print("Driver downloaded.")

    def install_driver(self, driver_path):
        print(f"Installing driver from {driver_path}...")
        subprocess.call(driver_path, shell=True)
        print("Driver installed.")

    def is_latest_version(self, hardware_id, installed_version):
        latest_version = driver_versions.get(hardware_id)
        if latest_version and installed_version == latest_version:
            print("Driver is already the latest version.")
            return True
        return False

    def install_drivers(self):
        print("Starting driver installation process...\n")
        device_info = self.get_device_info()
        receipt = DriverInstallationReceipt()
        for device in device_info:
            receipt.add_device_info(device)
            hardware_ids = device['HardwareID']
            for hardware_id in hardware_ids:
                print(f"\nSearching for driver for hardware ID: {hardware_id}")
                driver_url = self.search_for_driver(hardware_id)
                if driver_url:
                    installed_version = driver_versions.get(hardware_id)
                    if not self.is_latest_version(hardware_id, installed_version):
                        temp_file = 'temp_driver.exe'
                        self.download_driver(driver_url, temp_file)
                        self.install_driver(temp_file)
                        os.remove(temp_file)
                        receipt.add_driver_installation(driver_url)
                else:
                    print("No driver found for this hardware ID.")
        self.session.commit()

def get_access_dsn():
    try:
        # Open the ODBC Data Sources key in the Windows Registry
        reg = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
        key = winreg.OpenKey(reg, r"SOFTWARE\ODBC\ODBC.INI\ODBC Data Sources")

        # Iterate over subkeys to find the one that ends with "Microsoft Access Driver"
        dsn_name = None
        for i in range(1024):
            try:
                subkey_name = winreg.EnumKey(key, i)
                subkey = winreg.OpenKey(reg, fr"SOFTWARE\ODBC\ODBC.INI\{subkey_name}")
                driver = winreg.QueryValueEx(subkey, "Driver")
                if "Microsoft Access Driver" in driver[0]:
                    dsn_name = subkey_name
                    break
            except OSError:
                break

        # Close the registry keys
        winreg.CloseKey(key)
        winreg.CloseKey(reg)

        return dsn_name
    except Exception as e:
        print(f"Error accessing registry: {e}")
        return None

def main():
    dsn_name = get_access_dsn()
    if dsn_name:
        # Modify the connection string to use the retrieved DSN name
        # Example: 'microsoft+pyodbc://DSN_name'
        engine = create_engine(f'microsoft+pyodbc://{dsn_name}', echo=True)
        installer = DriverInstaller(engine)
        installer.install_drivers()
    else:
        print("Microsoft Access DSN not found.")

if __name__ == "__main__":
    main()
