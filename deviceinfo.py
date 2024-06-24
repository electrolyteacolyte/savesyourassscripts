import tkinter as tk
from tkinter import messagebox
import wmi

def get_hardware_info():
    c = wmi.WMI()
    devices = []
    for item in c.Win32_PnPEntity():
        device = {
            "name": item.Name,
            "type": item.Description,
            "status": item.Status,
            "location": getattr(item, "Location", "N/A")
        }
        devices.append(device)
    return devices

def populate_device_info():
    devices = get_hardware_info()
    for i, device in enumerate(devices):
        device_frame = tk.Frame(device_info_frame, relief=tk.RIDGE, borderwidth=1)
        device_frame.grid(row=i, column=0, sticky="nsew")

        tk.Label(device_frame, text=device["name"]).grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        tk.Label(device_frame, text=device["type"]).grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        tk.Label(device_frame, text=device["status"]).grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        tk.Label(device_frame, text=device["location"]).grid(row=0, column=3, padx=5, pady=5, sticky="nsew")

def save_to_file():
    devices = get_hardware_info()
    with open("hardware_info.txt", "w", encoding="utf-8") as f:
        for device in devices:
            f.write(f"Device Name: {device['name']}\n")
            f.write(f"Device Type: {device['type']}\n")
            f.write(f"Status: {device['status']}\n")
            f.write(f"Location: {device['location']}\n")
            f.write("\n")
    messagebox.showinfo("Saved", "Hardware info has been saved to hardware_info.txt")

def _on_mouse_wheel(event):
    canvas.yview_scroll(-1 * int((event.delta / 120)), "units")

# Create main window
root = tk.Tk()
root.title("Device Info Board")

# Create a frame for device information and add scrollbars
canvas = tk.Canvas(root)
canvas.pack(side="left", fill="both", expand=True)

scrollbar = tk.Scrollbar(root, command=canvas.yview)
scrollbar.pack(side="right", fill="y", before=canvas)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind_all("<MouseWheel>", _on_mouse_wheel)

device_info_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=device_info_frame, anchor="nw")

# Populate device information
populate_device_info()

# Create a button to save hardware info
save_button = tk.Button(root, text="Save to File", command=save_to_file)
save_button.pack()

root.mainloop()
