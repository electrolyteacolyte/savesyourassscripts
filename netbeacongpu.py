import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc
import socket
import subprocess
import time
import random
from multiprocessing.pool import ThreadPool
import requests
import os
import threading
import tkinter as tk
from tkinter import filedialog

# Check if CUDA is available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f'Using device: {device}')
gc.enable()

# Step 1: Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 100)  # Output size to match the payload length

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Using sigmoid to get output between 0 and 1
        return x

# Initialize the neural network
model = SimpleNet().to(device)

# Dummy training data
X_train = torch.randn(1000, 10).to(device)
y_train = torch.randn(1000, 100).to(device)

# Create a DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (shortened for demonstration purposes)
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Function to perform calculations on GPU using the trained network
def calculate_on_gpu(task_id):
    try:
        model.eval()
        with torch.no_grad():
            task_on_gpu(task_id)
            task_on_gpu1(task_id)
            task_on_gpu2(task_id)
            task_on_gpu3(task_id)
        return task_id
    finally:
        torch.cuda.empty_cache()  # Clear GPU cache after each calculation
        gc.collect()

# Function to monitor memory usage for garbage collection purposes
def monitor_memory():
    mem = psutil.virtual_memory()
    # Log memory usage
    print(f"Total memory: {mem.total}, Available memory: {mem.available}, Used memory: {mem.used}")
    if mem.available < mem.total * 0.1:
        print("Low memory detected, performing garbage collection.")

# Function to execute task on GPU
def task_on_gpu(task_id):
    torch.cuda.set_device(0)  # Set GPU device index
    while True:
        start_heartbeat(self)
        print(f"Task {task_id} executed on GPU.")
        monitor_memory()  # Monitor memory usage
        # Add a delay or other tasks as needed

# Function to execute task on GPU
def task_on_gpu1(task_id):
    torch.cuda.set_device(0)  # Set GPU device index
    while True:
        start_heartbeat(self)
        print(f"Task {task_id} executed on GPU.")
        monitor_memory()  # Monitor memory usage
        # Add a delay or other tasks as needed

# Function to execute task on GPU
def task_on_gpu2(task_id):
    torch.cuda.set_device(0)  # Set GPU device index
    while True:
        start_heartbeat(self)
        print(f"Task {task_id} executed on GPU.")
        monitor_memory()  # Monitor memory usage
        # Add a delay or other tasks as needed

# Function to execute task on GPU
def task_on_gpu3(task_id):
    torch.cuda.set_device(0)  # Set GPU device index
    while True:
        start_heartbeat(self)
        print(f"Task {task_id} executed on GPU.")
        monitor_memory()  # Monitor memory usage
        # Add a delay or other tasks as needed



# Create ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4096) as executor:
        executor.submit(calculate_on_gpu)

class NetworkScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Network Beacon")

        self.processes_label = tk.Label(root, text="Processes Running:")
        self.processes_label.pack()

        self.processes_text = tk.Text(root, height=10, width=50)
        self.processes_text.pack()

        self.ip_label = tk.Label(root, text="Local IP:")
        self.ip_label.pack()

        self.ip_text = tk.Text(root, height=1, width=50)
        self.ip_text.pack()

        self.global_ip_label = tk.Label(root, text="Global IP:")
        self.global_ip_label.pack()

        self.global_ip_text = tk.Text(root, height=1, width=50)
        self.global_ip_text.pack()

        self.scan_button = tk.Button(root, text="Start NetworkScan(smallest subnetwork)", command=self.start_scan)
        self.scan_button.pack()

        self.sos_button = tk.Button(root, text="Send SOS Signal(NASA)", command=self.send_sos_signal)
        self.sos_button.pack()

        self.load_button = tk.Button(root, text="Load IP List from Text File to add it to the heartbeat", command=self.load_ip_list)
        self.load_button.pack()

        self.load_button1 = tk.Button(root, text="Load IP List1 from Text File to add it to the heartbeat", command=self.load_ip_list1)
        self.load_button1.pack()

        self.load_button1 = tk.Button(root, text="Load IP List2 from Text File to add it to the heartbeat", command=self.load_ip_list1)
        self.load_button1.pack()

        self.load_button1 = tk.Button(root, text="Load IP List3 from Text File to add it to the heartbeat", command=self.load_ip_list1)
        self.load_button1.pack()

        self.start_heartbeat_button = tk.Button(root, text="Start Heartbeat", command=self.start_heartbeat)
        self.start_heartbeat_button.pack()

        self.stop_heartbeat_button = tk.Button(root, text="Stop Heartbeat", command=self.stop_heartbeat)
        self.stop_heartbeat_button.pack()

        self.loaded_ips = []
        self.loaded_ips1 = []
        self.heartbeat_pool = ThreadPool(processes=1)
        self.heartbeat_running = False

        self.run_startup()

    def run_startup(self):
        local_ip = self.get_local_ip()
        global_ip = self.get_global_ip()

        self.ip_text.insert(tk.END, local_ip)
        self.global_ip_text.insert(tk.END, global_ip)

    def start_scan(self):
        self.processes_text.delete(1.0, tk.END)
        self.ip_text.delete(1.0, tk.END)
        self.global_ip_text.delete(1.0, tk.END)

        local_ip = self.get_local_ip()
        self.ip_text.insert(tk.END, local_ip)

        global_ip = self.get_global_ip()
        self.global_ip_text.insert(tk.END, global_ip)

        # Run the scan in a background process
        scan_process = threading.Thread(target=self.run_scan, args=(global_ip,))
        scan_process.daemon = True
        scan_process.start()

    def run_scan(self, global_ip):
        subnet_range = self.get_subnet_range(global_ip)
        results = self.perform_scan(subnet_range)
        self.log_results(results)

    def perform_scan(self, subnet):
        results = []
        for i in range(1, 255):
            ip = subnet[:-4] + str(i)
            result = subprocess.call(['ping', '-c', '1', ip])
            if result == 0:
                results.append(f"Host {ip} is up\n")
            else:
                results.append(f"Host {ip} is down\n")

        return results

    def log_results(self, results):
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        log_file = os.path.join(desktop_path, "network_scan_results.txt")
        with open(log_file, "w") as file:
            for result in results:
                file.write(result)

    def get_local_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip

    def get_global_ip(self):
        try:
            response = requests.get("https://api.ipify.org?format=text")
            return response.text
        except Exception as e:
            self.processes_text.insert(tk.END, f"Error getting global IP: {e}\n")
            return "127.0.0.1"

    def get_subnet_range(self, ip):
        parts = ip.split('.')
        subnet = '.'.join(parts[:3]) + '.0/24'
        return subnet

    def send_sos_signal(self):
        ip_address = "192.0.66.108"
        morse_sos = [
            '. . .   - - -   . . .',
            '. . .   - - -   . . .',
            '. . .   - - -   . . .'
        ]
        for char in morse_sos:
            for symbol in char.split('   '):
                for signal in symbol.split():
                    if signal == '.':
                        subprocess.call(['ping', '-c', '1', ip_address])
                        time.sleep(0.1)
                    elif signal == '-':
                        subprocess.call(['ping', '-c', '1', ip_address])
                        time.sleep(0.3)
                time.sleep(0.1)
            time.sleep(0.3)

    def log_handshake(self, ip, local_ip):
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        log_file = os.path.join(desktop_path, "successful_handshakes.txt")
        with open(log_file, "a") as file:
            file.write(f"Successful handshake with {ip} using local IP: {local_ip}\n")

    def load_ip_list(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            with open(file_path, 'r') as file:
                self.loaded_ips = file.read().splitlines()
            gc.get_objects().extend(self.loaded_ips)

    def load_ip_list1(self):
        file_path1 = filedialog.askopenfilename()
        if file_path1:
            with open(file_path1, 'r') as file:
                self.loaded_ips1 = file.read().splitlines()
            gc.get_objects().extend(self.loaded_ips1)

    def load_ip_list2(self):
        file_path2 = filedialog.askopenfilename()
        if file_path2:
            with open(file_path2, 'r') as file:
                self.loaded_ips2 = file.read().splitlines()
            gc.get_objects().extend(self.loaded_ips2)

    def load_ip_list3(self):
        file_path3 = filedialog.askopenfilename()
        if file_path3:
            with open(file_path3, 'r') as file:
                self.loaded_ips3 = file.read().splitlines()
            gc.get_objects().extend(self.loaded_ips3)
            
    def send_heartbeat(self):
        handshake_methods = {
            "TCP": [80, 443, 25, 110, 143, 22, 21, 389, 636, 23, 3389, 5900],
            "UDP": [53, 161, 162, 123, 514],
            # Add more handshake methods here
        }

        local_ip = self.get_local_ip()
        global_ip = self.get_global_ip()
        all_ips = self.loaded_ips
        for ip in all_ips:
            try:
                for method_name, ports in handshake_methods.items():
                    for port in ports:
                        try:
                            handshake_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            handshake_socket.connect((ip, port))
                            handshake_socket.sendall(global_ip.encode())
                            handshake_socket.close()

                            # Print successful handshake
                            result = f"Heartbeat sent to {ip} using {method_name} ({port})\n"
                            self.processes_text.insert(tk.END, result)
                            self.root.update_idletasks()

                            # Log successful handshake
                            self.log_handshake(ip, global_ip)
                        except Exception as e:
                            # Print failed handshake
                            result = f"Failed to send heartbeat to {ip} using {method_name} ({port}): {str(e)}\n"
                            self.processes_text.insert(tk.END, result)
                            self.root.update_idletasks()
            except Exception as e:
                # Print failed handshake
                result = f"Failed to send heartbeat to {ip}: {str(e)}\n"
                self.processes_text.insert(tk.END, result)
                self.root.update_idletasks()

        gc.collect()

    def send_heartbeat1(self):
        handshake_methods = {
            "TCP": [80, 443, 25, 110, 143, 22, 21, 389, 636, 23, 3389, 5900],
            "UDP": [53, 161, 162, 123, 514],
            # Add more handshake methods here
        }

        local_ip = self.get_local_ip()
        global_ip = self.get_global_ip()
        all_ips1 = self.loaded_ips1
        for ip in all_ips1:
            try:
                for method_name, ports in handshake_methods.items():
                    for port in ports:
                        try:
                            handshake_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            handshake_socket.connect((ip, port))
                            handshake_socket.sendall(global_ip.encode())
                            handshake_socket.close()

                            # Print successful handshake
                            result = f"Heartbeat sent to {ip} using {method_name} ({port})\n"
                            self.processes_text.insert(tk.END, result)
                            self.root.update_idletasks()

                            # Log successful handshake
                            self.log_handshake(ip, global_ip)
                        except Exception as e:
                            # Print failed handshake
                            result = f"Failed to send heartbeat to {ip} using {method_name} ({port}): {str(e)}\n"
                            self.processes_text.insert(tk.END, result)
                            self.root.update_idletasks()
            except Exception as e:
                # Print failed handshake
                result = f"Failed to send heartbeat to {ip}: {str(e)}\n"
                self.processes_text.insert(tk.END, result)
                self.root.update_idletasks()

        gc.collect()

        def send_heartbeat2(self):
            handshake_methods = {
            "TCP": [80, 443, 25, 110, 143, 22, 21, 389, 636, 23, 3389, 5900],
            "UDP": [53, 161, 162, 123, 514],
            # Add more handshake methods here
        }
            
        local_ip = self.get_local_ip()
        global_ip = self.get_global_ip()
        all_ips2 = self.loaded_ips2
        for ip in all_ips2:
            try:
                for method_name, ports in handshake_methods.items():
                    for port in ports:
                        try:
                            handshake_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            handshake_socket.connect((ip, port))
                            handshake_socket.sendall(global_ip.encode())
                            handshake_socket.close()

                            # Print successful handshake
                            result = f"Heartbeat sent to {ip} using {method_name} ({port})\n"
                            self.processes_text.insert(tk.END, result)
                            self.root.update_idletasks()

                            # Log successful handshake
                            self.log_handshake(ip, global_ip)
                        except Exception as e:
                            # Print failed handshake
                            result = f"Failed to send heartbeat to {ip} using {method_name} ({port}): {str(e)}\n"
                            self.processes_text.insert(tk.END, result)
                            self.root.update_idletasks()
            except Exception as e:
                # Print failed handshake
                result = f"Failed to send heartbeat to {ip}: {str(e)}\n"
                self.processes_text.insert(tk.END, result)
                self.root.update_idletasks()

        gc.collect()


        def send_heartbeat3(self):
            handshake_methods = {
            "TCP": [80, 443, 25, 110, 143, 22, 21, 389, 636, 23, 3389, 5900],
            "UDP": [53, 161, 162, 123, 514],
            # Add more handshake methods here
        }

        local_ip = self.get_local_ip()
        global_ip = self.get_global_ip()
        all_ips3 = self.loaded_ips3
        for ip in all_ips3:
            try:
                for method_name, ports in handshake_methods.items():
                    for port in ports:
                        try:
                            handshake_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            handshake_socket.connect((ip, port))
                            handshake_socket.sendall(global_ip.encode())
                            handshake_socket.close()

                            # Print successful handshake
                            result = f"Heartbeat sent to {ip} using {method_name} ({port})\n"
                            self.processes_text.insert(tk.END, result)
                            self.root.update_idletasks()

                            # Log successful handshake
                            self.log_handshake(ip, global_ip)
                        except Exception as e:
                            # Print failed handshake
                            result = f"Failed to send heartbeat to {ip} using {method_name} ({port}): {str(e)}\n"
                            self.processes_text.insert(tk.END, result)
                            self.root.update_idletasks()
            except Exception as e:
                # Print failed handshake
                result = f"Failed to send heartbeat to {ip}: {str(e)}\n"
                self.processes_text.insert(tk.END, result)
                self.root.update_idletasks()

        gc.collect()

    def start_heartbeat(self):
        self.heartbeat_running = True
        heartbeat_thread = threading.Thread(target=self.send_heartbeat_loop)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()

    def stop_heartbeat(self):
        self.heartbeat_running = False

    def send_heartbeat_loop(self):
        while self.heartbeat_running:
            self.send_heartbeat()
            self.send_heartbeat1()
            self.send_heartbeat2()
            self.send_heartbeat3()
            time.sleep(2)  # Send heartbeat every 60 seconds

def main():
    root = tk.Tk()
    app = NetworkScannerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
