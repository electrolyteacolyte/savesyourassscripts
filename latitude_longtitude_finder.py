import tkinter as tk
import webbrowser
import requests

def get_location_from_ip():
    try:
        # Use a free IP geolocation API to get the user's location based on their IP
        response = requests.get('https://ipinfo.io/json')
        data = response.json()
        location = data.get('loc')
        return location.split(',')
    except Exception as e:
        print("Error fetching location from IP:", e)
        return None, None

def open_location_in_maps():
    latitude = latitude_entry.get()
    longitude = longitude_entry.get()

    try:
        if not latitude and not longitude:
            # If latitude and longitude are not provided, get the location from IP
            latitude, longitude = get_location_from_ip()

        if latitude and longitude:
            # Construct the Google Maps URL with the latitude and longitude
            url = f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"

            # Open the URL in the default web browser
            webbrowser.open(url)
        else:
            result_label.config(text="Error: Unable to retrieve location")
    except Exception as e:
        result_label.config(text="Error: Invalid latitude/longitude values")

# Create main window
root = tk.Tk()
root.title("Google Maps Locator")

# Information label
info_label = tk.Label(root, text="Note: If latitude and longitude are not provided, your location will be fetched from your IP address.")
info_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

# Latitude entry
latitude_label = tk.Label(root, text="Latitude:")
latitude_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
latitude_entry = tk.Entry(root)
latitude_entry.grid(row=1, column=1, padx=5, pady=5)

# Longitude entry
longitude_label = tk.Label(root, text="Longitude:")
longitude_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
longitude_entry = tk.Entry(root)
longitude_entry.grid(row=2, column=1, padx=5, pady=5)

# Button to open location
locate_button = tk.Button(root, text="Locate", command=open_location_in_maps)
locate_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

# Result label
result_label = tk.Label(root, text="")
result_label.grid(row=4, column=0, columnspan=2)

root.mainloop()
