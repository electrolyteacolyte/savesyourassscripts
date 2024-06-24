import subprocess
import os

def open_google_earth():
    try:
        # Path to the Google Earth executable
        google_earth_path = os.path.join(os.getenv("PROGRAMFILES"), "Google", "Google Earth Pro", "client", "googleearth.exe")

        # Check if the Google Earth executable exists
        if os.path.exists(google_earth_path):
            subprocess.Popen(google_earth_path)
            print("Google Earth opened successfully.")
        else:
            print("Google Earth executable not found.")
    except Exception as e:
        print(f"Error while opening Google Earth: {e}")

if __name__ == "__main__":
    open_google_earth()
