import tkinter as tk
from tkinter import simpledialog, ttk, messagebox
import threading
import socket
import time
import os
import requests
from ncclient import manager

class NetcatMeshNetwork:
    def __init__(self, root):
        self.root = root
        self.root.title("Netcat Mesh Network")
        
        # Show startup message box
        self.show_startup_message()

        # GUI elements
        self.scan_button = tk.Button(root, text="Scan Network", command=self.scan_network)
        self.scan_button.pack(pady=10)

        self.query_button = tk.Button(root, text="Query DNS", command=self.query_dns)
        self.query_button.pack(pady=10)

        self.communicate_button = tk.Button(root, text="Communicate", command=self.show_communicate_dialog)
        self.communicate_button.pack(pady=10)

        self.text_box = tk.Text(root, width=80, height=20)
        self.text_box.pack(pady=10)

        self.domain_entry = tk.Entry(root)
        self.domain_entry.pack(pady=10)
        self.domain_entry.insert(0, "google.com")

        # Network properties
        self.global_ip = self.get_global_ip()
        self.ip_range = self.get_ip_range(self.global_ip)
        self.connections_file = "netcat_connections.txt"
        self.dns_server = "1.1.1.1"
        self.heartbeat_port = 9999
        self.heartbeat_message = f"Hi I'm using netcat: {self.global_ip}"

        # Common ports to check
        self.common_ports = [20, 21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 3389]

        # Start DNS query thread
        self.dns_thread = threading.Thread(target=self.dns_query_loop)
        self.dns_thread.daemon = True
        self.dns_thread.start()

        # Start Heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self.send_heartbeat)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()

    def show_startup_message(self):
        message = (
            "Welcome to Netcat Mesh Network!\n\n"
            "Features and Functions:\n"
            "1. Scan Network: Scan the local network for open ports and save results.\n"
            "2. Query DNS: Query DNS server for a domain's details every minute.\n"
            "3. Communicate: Perform various actions with a specified IP:\n"
            "   - Chat: Chat with another IP on a specified port.\n"
            "   - File Transfer: Transfer files to/from another IP on a specified port.\n"
            "   - Port Scan: Scan specified ports on the target IP.\n"
            "   - Remote Shell: Open a remote shell on the target IP using a specified port.\n"
            "4. Heartbeat: Broadcast a heartbeat message every 60 seconds.\n\n"
            "Enjoy using the application!"
        )
        messagebox.showinfo("Netcat Mesh Network", message)

    def get_global_ip(self):
        try:
            response = requests.get("https://api.ipify.org?format=text")
            return response.text
        except Exception as e:
            self.text_box.insert(tk.END, f"Error getting global IP: {e}\n")
            return "127.0.0.1"

    def get_ip_range(self, ip):
        parts = ip.split('.')
        base_ip = '.'.join(parts[:3]) + '.'
        return [f"{base_ip}{i}" for i in range(1, 255)]
    
    def scan_network(self):
        self.text_box.insert(tk.END, "Scanning network...\n")
        open_ips = self.check_open_ports(self.ip_range)
        with open(self.connections_file, 'w') as file:
            for ip in open_ips:
                file.write(ip + "\n")
        self.text_box.insert(tk.END, f"Scan complete. Results saved in {self.connections_file}\n")

    def check_open_ports(self, ips):
        open_ips = []
        for ip in ips:
            for port in self.common_ports:
                try:
                    with manager.connect(host=ip, port=port) as m:
                        open_ips.append(f"{ip}:{port}")
                        self.text_box.insert(tk.END, f"Found open IP: {ip} on port {port}\n")
                except Exception as e:
                    pass
        return open_ips

    def query_dns(self):
        domain = self.domain_entry.get()
        self.text_box.insert(tk.END, f"Querying DNS server {self.dns_server} for domain {domain}...\n")
        try:
            result = socket.gethostbyname_ex(domain)
            self.text_box.insert(tk.END, f"Domain: {domain}\n")
            self.text_box.insert(tk.END, f"Hostname: {result[0]}\n")
            self.text_box.insert(tk.END, f"Aliases: {result[1]}\n")
            self.text_box.insert(tk.END, f"IP addresses: {result[2]}\n")
        except Exception as e:
            self.text_box.insert(tk.END, f"Error querying domain {domain}: {e}\n")

    def dns_query_loop(self):
        while True:
            self.query_dns()
            time.sleep(60)

    def send_heartbeat(self):
        while True:
            self.broadcast_message(self.heartbeat_message, self.heartbeat_port)
            time.sleep(60)

    def broadcast_message(self, message, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.sendto(message.encode('utf-8'), ('<broadcast>', port))
        s.close()
        self.text_box.insert(tk.END, f"Heartbeat broadcasted: {message}\n")

    def show_communicate_dialog(self):
        self.communicate_dialog = tk.Toplevel(self.root)
        self.communicate_dialog.title("Communicate")

        tk.Label(self.communicate_dialog, text="Enter IP:").pack(pady=5)
        self.ip_entry = tk.Entry(self.communicate_dialog)
        self.ip_entry.pack(pady=5)

        tk.Label(self.communicate_dialog, text="Select Action:").pack(pady=5)
        self.action_var = tk.StringVar()
        self.action_menu = ttk.Combobox(self.communicate_dialog, textvariable=self.action_var)
        self.action_menu['values'] = ('Chat', 'File Transfer', 'Port Scan', 'Remote Shell')
        self.action_menu.pack(pady=5)
        self.action_menu.bind("<<ComboboxSelected>>", self.show_action_options)

        self.action_options_frame = tk.Frame(self.communicate_dialog)
        self.action_options_frame.pack(pady=5)

        self.execute_button = tk.Button(self.communicate_dialog, text="Execute", command=self.execute_action)
        self.execute_button.pack(pady=10)

    def show_action_options(self, event):
        for widget in self.action_options_frame.winfo_children():
            widget.destroy()

        action = self.action_var.get()

        if action == "Chat":
            tk.Label(self.action_options_frame, text="Port:").pack(pady=5)
            self.port_entry = tk.Entry(self.action_options_frame)
            self.port_entry.pack(pady=5)
        elif action == "File Transfer":
            tk.Label(self.action_options_frame, text="Port:").pack(pady=5)
            self.port_entry = tk.Entry(self.action_options_frame)
            self.port_entry.pack(pady=5)
            tk.Label(self.action_options_frame, text="File Path:").pack(pady=5)
            self.file_path_entry = tk.Entry(self.action_options_frame)
            self.file_path_entry.pack(pady=5)
        elif action == "Port Scan":
            tk.Label(self.action_options_frame, text="Ports (comma-separated):").pack(pady=5)
            self.ports_entry = tk.Entry(self.action_options_frame)
            self.ports_entry.pack(pady=5)
        elif action == "Remote Shell":
            tk.Label(self.action_options_frame, text="Port:").pack(pady=5)
            self.port_entry = tk.Entry(self.action_options_frame)
            self.port_entry.pack(pady=5)

    def execute_action(self):
        ip = self.ip_entry.get()
        action = self.action_var.get()

        if action == "Chat":
            port = self.port_entry.get()
            threading.Thread(target=self.start_chat, args=(ip, port)).start()
        elif action == "File Transfer":
            port = self.port_entry.get()
            file_path = self.file_path_entry.get()
            threading.Thread(target=self.start_file_transfer, args=(ip, port, file_path)).start()
        elif action == "Port Scan":
            ports = self.ports_entry.get()
            port_list = ports.split(',')
            for port in port_list:
                threading.Thread(target=self.start_port_scan, args=(ip, port)).start()
        elif action == "Remote Shell":
            port = self.port_entry.get()
            threading.Thread(target=self.start_remote_shell, args=(ip, port)).start()

    def start_chat(self, ip, port):
        with manager.connect(host=ip, port=port) as m:
            m.on_disconnect = self.on_disconnect
            m.close_session()

    def start_file_transfer(self, ip, port, file_path):
        with manager.connect(host=ip, port=port) as m:
            m.on_disconnect = self.on_disconnect
            with open(file_path, 'rb') as file:
                m.copy_config(source=file, target="running")

    def start_port_scan(self, ip, port):
        try:
            with manager.connect(host=ip, port=port, timeout=1) as m:
                self.text_box.insert(tk.END, f"Port {port} is open on {ip}\n")
        except:
            pass

    def start_remote_shell(self, ip, port):
        with manager.connect(host=ip, port=port) as m:
            m.on_disconnect = self.on_disconnect
            m.interact()

    def on_disconnect(self, session, channel):
        self.text_box.insert(tk.END, "Disconnected from session\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = NetcatMeshNetwork(root)
    root.mainloop()

