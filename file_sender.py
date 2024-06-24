import os
import tkinter as tk
from tkinter import filedialog, messagebox
import socket
import threading
import random
import requests

# List of DNS providers
DNS_PROVIDERS = [
    "1.1.1.1",
    "8.8.8.8",
    "208.67.222.222",
    "9.9.9.9",
    "64.6.64.6",
    "8.26.56.26",
    "84.200.69.80",
    "8.8.4.4",
    "208.67.220.220",
    "64.6.65.6"
]

# Unique UUID for identification/authentication
UUID = "9128fd2e-b2d5-4350-b81d-73701190a81d"

class FileSharerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("File Sharer")
        
        self.download_folder = os.path.join(os.path.expanduser("~"), "Desktop")
        
        # Select DNS server
        self.selected_dns = random.choice(DNS_PROVIDERS)
        
        # Get public and local IP addresses
        self.public_ip = self.get_public_ip()
        self.local_ip = self.get_local_ip()
        
        # Main frame
        self.main_frame = tk.Frame(root, padx=10, pady=10)
        self.main_frame.pack()
        
        # IP Label
        self.ip_label = tk.Label(self.main_frame, text=f"Public IP Address: {self.public_ip}\nLocal IP Address: {self.local_ip}")
        self.ip_label.pack()
        
        # DNS Server Label
        self.dns_label = tk.Label(self.main_frame, text=f"Using DNS Server: {self.selected_dns}")
        self.dns_label.pack()
        
        # IP Address Entry
        self.ip_label = tk.Label(self.main_frame, text="Enter IP Address:")
        self.ip_label.pack()
        self.ip_entry = tk.Entry(self.main_frame)
        self.ip_entry.pack()
        
        # Select File Button
        self.select_file_button = tk.Button(self.main_frame, text="Select File", command=self.select_file)
        self.select_file_button.pack()
        
        # Send File Button
        self.send_file_button = tk.Button(self.main_frame, text="Send File", command=self.send_file)
        self.send_file_button.pack()
        
        # Thread to start listening for incoming connections
        self.listen_thread = threading.Thread(target=self.listen_for_files)
        self.listen_thread.start()
        
        # Received File Label
        self.received_file_label = tk.Label(self.main_frame, text="Received File: ")
        self.received_file_label.pack()
        
    def get_public_ip(self):
        try:
            response = requests.get('https://api.ipify.org').text
            return response.strip()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch public IP address: {e}")
            return None
        
    def get_local_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))  # Connect to Google DNS server
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch local IP address: {e}")
            return None
        
    def select_file(self):
        self.file_path = filedialog.askopenfilename(initialdir=self.download_folder, title="Select File to Send")
        
    # Send File Button
    def send_file(self):
        ip_address = self.ip_entry.get()
        if not ip_address:
            messagebox.showerror("Error", "Please enter an IP address.")
            return
    
        if not os.path.exists(self.file_path):
            messagebox.showerror("Error", "Please select a file to send.")
            return
    
        try:
        # Create socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
            s.connect((ip_address, 1))  # Change port as necessary
        
        # Send UUID for authentication
            s.sendall(UUID.encode())
        
        # Send file path
            s.sendall(self.file_path.encode())
        
        # Send file
            with open(self.file_path, "rb") as f:
                while True:
                    data = f.read(1024)
                    if not data:
                        break
                    s.sendall(data)
        
            s.close()
            messagebox.showinfo("Success", "File sent successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

        
    def listen_for_files(self):
        try:
            # Create socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('0.0.0.0', 1))  # Listening on all interfaces
            s.listen(5)
            
            while True:
                conn, addr = s.accept()
                threading.Thread(target=self.handle_incoming_connection, args=(conn, addr)).start()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            
    def handle_incoming_connection(self, conn, addr):
        try:
        # Receive UUID for authentication
            received_uuid = conn.recv(1024).decode()
            if received_uuid != UUID:
                messagebox.showerror("Error", "Authentication failed.")
                conn.close()
                return
        
        # Prompt user to accept or reject incoming file transfer
            response = messagebox.askyesno("Incoming File Transfer", f"Do you want to accept a file from {addr[0]}?")
            if response:
            # Receive file path
                file_path = conn.recv(1024).decode()
            
            # Save received file
                filename = os.path.basename(file_path)
                save_path = os.path.join(self.download_folder, filename)
            
            # Receive file data and write to file
                with open(save_path, "wb") as f:
                    while True:
                        data = conn.recv(1024)
                        if not data:
                            break
                        f.write(data)
            
                self.received_file_label.config(text=f"Received File: {filename}")
                messagebox.showinfo("Success", f"File received successfully from {addr[0]}!")
            else:
                messagebox.showinfo("Info", "File transfer rejected.")
        
            conn.close()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

        

if __name__ == "__main__":
    root = tk.Tk()
    app = FileSharerApp(root)
    root.mainloop()
