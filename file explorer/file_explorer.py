import os
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog

def list_files(folder_path):
    current_path.set(folder_path)
    refresh_ui()

def go_back(event=None):
    onclick_parent_folder()  # Call onclick_parent_folder function to go back
    refresh_ui()

def refresh_ui():
    for widget in frame.winfo_children():
        widget.destroy()
    
    folder_path = current_path.get()
    try:
        files = os.listdir(folder_path)
    except PermissionError as e:
        print(f"Permission denied to access {folder_path}: {e}")
        return
    
    back_label = ttk.Label(frame, text="Go Back", cursor="hand2", foreground="blue", font="TkDefaultFont 9 underline", anchor="w")
    back_label.pack(fill='x')
    back_label.bind("<Button-1>", go_back)
    
    for file_name in files:
        full_path = os.path.join(folder_path, file_name)
        if os.path.isdir(full_path):
            label = ttk.Label(frame, text=file_name, cursor="hand2", foreground="black", font="TkDefaultFont 9", anchor="w")
            label.pack(fill='x')
            label.bind("<Button-1>", lambda event, path=full_path: list_files(path))
        else:
            label = ttk.Label(frame, text=file_name, cursor="hand2", foreground="black", font="TkDefaultFont 9", anchor="w")
            label.pack(fill='x')
            label.bind("<Button-1>", lambda event, path=full_path: open_default_application(path))

def open_default_application(file_path):
    if os.access(file_path, os.R_OK):
        try:
            os.system(f'powershell Start-Process "{file_path}" -Verb RunAs')
        except PermissionError as e:
            print(f"Permission denied to access {file_path}: {e}")
    else:
        print(f"File {file_path} is not accessible.")


def onclick_parent_folder():
    current_folder = current_path.get()
    parent_folder = os.path.dirname(current_folder)
    list_files(parent_folder)

def list_files_in_drive(event=None):
    drive = drive_combobox.get()
    if drive:
        list_files(drive)

def get_available_drives():
    drives = []
    for drive in range(ord('A'), ord('Z')+1):
        drive = chr(drive) + ':/'
        if os.path.exists(drive):
            drives.append(drive)
    return drives

root = tk.Tk()
root.title("Basic File Explorer")

current_path = tk.StringVar()

# Drive selection
ttk.Label(root, text="Select Drive:").pack(pady=5)
drive_combobox = ttk.Combobox(root, values=get_available_drives(), width=30)
drive_combobox.pack(pady=5)
drive_combobox.bind("<<ComboboxSelected>>", lambda event: list_files_in_drive())

# Entry for folder path
folder_path_entry = ttk.Entry(root, textvariable=current_path, width=40)
folder_path_entry.pack(pady=5)

# Frame to display files
frame = ttk.Frame(root)
frame.pack(padx=10, pady=5, fill='both', expand=True)

# Create "Go Back" label/button and bind go_back function
back_label = ttk.Label(frame, text="Go Back", cursor="hand2", foreground="blue", font="TkDefaultFont 9 underline", anchor="w")
back_label.pack(fill='x')
back_label.bind("<Button-1>", go_back)

root.mainloop()
