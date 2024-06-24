import socket
import subprocess
import time
import random
import tkinter as tk
import threading
import gc
import requests
import os
from tkinter import filedialog
from multiprocessing.pool import ThreadPool

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
        
        self.start_heartbeat_button = tk.Button(root, text="Start Heartbeat", command=self.start_heartbeat)
        self.start_heartbeat_button.pack()
        
        self.stop_heartbeat_button = tk.Button(root, text="Stop Heartbeat", command=self.stop_heartbeat)
        self.stop_heartbeat_button.pack()
        
        self.loaded_ips = []
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
    
    def send_heartbeat(self):
        handshake_methods = {
            "TCP": [80, 443, 25, 110, 143, 22, 21, 389, 636, 23, 3389, 5900],
            "UDP": [53, 161, 162, 123, 514],
            # Add more handshake methods here
        }
    
        dns_servers = [
            "8.8.8.8", "8.8.4.4", "1.1.1.1", "9.9.9.9",
            "208.67.222.222", "208.67.220.220", "185.228.168.9", "185.228.169.9"
        ]
        local_ip = self.get_local_ip()
        global_ip = self.get_global_ip()
        all_ips = dns_servers + self.loaded_ips
        for dns_server in random.sample(all_ips, 8):
            try:
                for method_name, ports in handshake_methods.items():
                    for port in ports:
                        try:
                            handshake_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            handshake_socket.connect((dns_server, port))
                            handshake_socket.sendall(global_ip.encode())
                            handshake_socket.close()

                            # Print successful handshake
                            result = f"Heartbeat sent to {dns_server} using {method_name} ({port})\n"
                            self.processes_text.insert(tk.END, result)
                            self.root.update_idletasks()

                            # Log successful handshake
                            self.log_handshake(dns_server, global_ip)
                        except Exception as e:
                            # Print failed handshake
                            result = f"Failed to send heartbeat to {dns_server} using {method_name} ({port}): {str(e)}\n"
                            self.processes_text.insert(tk.END, result)
                            self.root.update_idletasks()
            except Exception as e:
                # Print failed handshake
                result = f"Failed to send heartbeat to {dns_server}: {str(e)}\n"
                self.processes_text.insert(tk.END, result)
                self.root.update_idletasks()
        
        gc.collect()
        
    def start_heartbeat(self):
        if not self.heartbeat_running:
            self.heartbeat_running = True
            self.heartbeat_thread = threading.Thread(target=self.send_heartbeat_loop)
            self.heartbeat_thread.daemon = True
            self.heartbeat_thread.start()
    
    def stop_heartbeat(self):
        self.heartbeat_running = False

    def send_heartbeat_loop(self):
        while self.heartbeat_running:
            self.send_heartbeat()
            time.sleep(60)  # Send heartbeat every 60 seconds
    
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

    def log_handshake(self, dns_server, local_ip):
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        log_file = os.path.join(desktop_path, "successful_handshakes.txt")
        with open(log_file, "a") as file:
            file.write(f"Successful handshake with {dns_server} using local IP: {local_ip}\n")

    def load_ip_list(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            with open(file_path, 'r') as file:
                self.loaded_ips = file.read().splitlines()
            gc.get_objects().extend(self.loaded_ips)

def main():
    root = tk.Tk()
    app = NetworkScannerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

