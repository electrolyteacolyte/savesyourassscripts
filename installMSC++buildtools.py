import os
import subprocess
import requests

# URL to download Microsoft Visual C++ Build Tools
build_tools_url = "https://aka.ms/buildtools"

# Function to download file from URL
def download_file(url, destination):
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

# Function to install Microsoft Visual C++ Build Tools
def install_build_tools(installer_path):
    print("Starting installation of Microsoft Visual C++ Build Tools...")
    os.system(installer_path)

if __name__ == "__main__":
    # Define the path to save the installer
    installer_path = "vc_build_tools.exe"

    # Download the installer
    download_file(build_tools_url, installer_path)

    # Install Microsoft Visual C++ Build Tools
    install_build_tools(installer_path)

    # Clean up: Delete the installer file
    os.remove(installer_path)

    print("Microsoft Visual C++ Build Tools installation completed.")
