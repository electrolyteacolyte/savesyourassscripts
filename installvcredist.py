import os
import subprocess
import shutil
import requests

# List of download links for Visual C++ Redistributable packages
vcredist_links = [
    "https://aka.ms/vs/16/release/vc_redist.x86.exe",  # Visual C++ 2010
    "https://aka.ms/vs/16/release/vc_redist.x64.exe",  # Visual C++ 2010
    "https://aka.ms/vs/16/release/vc_redist.x86.exe",  # Visual C++ 2012
    "https://aka.ms/vs/16/release/vc_redist.x64.exe",  # Visual C++ 2012
    "https://aka.ms/vs/16/release/vc_redist.x86.exe",  # Visual C++ 2013
    "https://aka.ms/vs/16/release/vc_redist.x64.exe",  # Visual C++ 2013
    "https://aka.ms/vs/16/release/vc_redist.x86.exe",  # Visual C++ 2015-2019
    "https://aka.ms/vs/16/release/vc_redist.x64.exe",  # Visual C++ 2015-2019
]

def download_and_install_vcredist():
    for link in vcredist_links:
        filename = os.path.basename(link)
        print(f"Downloading {filename}...")
        with requests.get(link, stream=True) as r:
            with open(filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        print(f"{filename} downloaded.")

        print(f"Installing {filename}...")
        subprocess.call([filename, "/quiet", "/norestart"])
        print(f"{filename} installed.")

    print("All Visual C++ Redistributable packages installed. Restarting the computer...")
    os.system("shutdown /r /t 0")

if __name__ == "__main__":
    download_and_install_vcredist()
