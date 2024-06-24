import socket
import threading
import time
import random
import datetime
import requests
import tkinter as tk
import ipaddress
import pytz
import os

class BroadcastApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Broadcast and Baton Management")
        self.geometry("600x500")

        self.global_ip = self.get_global_ip()
        self.subnet_broadcasts = self.calculate_subnet_broadcasts(self.global_ip)
        self.online_ips_8 = []
        self.online_ips_16 = []
        self.online_ips_24 = []
        self.current_baton_holder_8 = "None"
        self.current_baton_holder_16 = "None"
        self.current_baton_holder_24 = "None"

        self.tz = pytz.utc  # Set the timezone to UTC

        self.create_widgets()
        self.update_global_ip()
        self.update_subnet_broadcasts()

        self.start_threads()

    def create_widgets(self):
        self.global_ip_label = tk.Label(self, text="Global IP: ")
        self.global_ip_label.pack()

        self.baton_holder_label_8 = tk.Label(self, text="Current Baton Holder (/8): ")
        self.baton_holder_label_8.pack()
        
        self.baton_holder_label_16 = tk.Label(self, text="Current Baton Holder (/16): ")
        self.baton_holder_label_16.pack()
        
        self.baton_holder_label_24 = tk.Label(self, text="Current Baton Holder (/24): ")
        self.baton_holder_label_24.pack()

        self.online_ips_label_8 = tk.Label(self, text="Online IPs (/8): ")
        self.online_ips_label_8.pack()

        self.online_ips_listbox_8 = tk.Listbox(self)
        self.online_ips_listbox_8.pack(fill=tk.BOTH, expand=True)

        self.online_ips_label_16 = tk.Label(self, text="Online IPs (/16): ")
        self.online_ips_label_16.pack()

        self.online_ips_listbox_16 = tk.Listbox(self)
        self.online_ips_listbox_16.pack(fill=tk.BOTH, expand=True)

        self.online_ips_label_24 = tk.Label(self, text="Online IPs (/24): ")
        self.online_ips_label_24.pack()

        self.online_ips_listbox_24 = tk.Listbox(self)
        self.online_ips_listbox_24.pack(fill=tk.BOTH, expand=True)

        self.processes_text = tk.Text(self, height=4)
        self.processes_text.pack(fill=tk.X)

        self.subnet_broadcasts_label = tk.Label(self, text="Subnet Broadcasts: ")
        self.subnet_broadcasts_label.pack()

        self.subnet_broadcasts_listbox = tk.Listbox(self)
        self.subnet_broadcasts_listbox.pack(fill=tk.BOTH, expand=True)

    def get_global_ip(self):
        try:
            response = requests.get("https://api.ipify.org?format=text")
            return response.text
        except Exception as e:
            self.processes_text.insert(tk.END, f"Error getting global IP: {e}\n")
            return "127.0.0.1"

    def update_global_ip(self):
        self.global_ip_label.config(text=f"Global IP: {self.global_ip}")

    def update_subnet_broadcasts(self):
        for broadcast in self.subnet_broadcasts:
            self.subnet_broadcasts_listbox.insert(tk.END, broadcast)

    def update_baton_holder(self, new_holder, broadcast_level):
        if broadcast_level == "/8":
            self.current_baton_holder_8 = new_holder
            self.baton_holder_label_8.config(text=f"Current Baton Holder (/8): {new_holder}")
        elif broadcast_level == "/16":
            self.current_baton_holder_16 = new_holder
            self.baton_holder_label_16.config(text=f"Current Baton Holder (/16): {new_holder}")
        elif broadcast_level == "/24":
            self.current_baton_holder_24 = new_holder
            self.baton_holder_label_24.config(text=f"Current Baton Holder (/24): {new_holder}")

    def update_online_ips(self, ip, message, broadcast_level):
        if broadcast_level == "/8" and ip not in self.online_ips_8:
            self.online_ips_8.append(ip)
            self.online_ips_listbox_8.insert(tk.END, f"{ip} - {message}")
        elif broadcast_level == "/16" and ip not in self.online_ips_16:
            self.online_ips_16.append(ip)
            self.online_ips_listbox_16.insert(tk.END, f"{ip} - {message}")
        elif broadcast_level == "/24" and ip not in self.online_ips_24:
            self.online_ips_24.append(ip)
            self.online_ips_listbox_24.insert(tk.END, f"{ip} - {message}")

    def calculate_subnet_broadcasts(self, global_ip):
        ip = ipaddress.ip_address(global_ip)
        networks = [
            ipaddress.ip_network(f"{ip}/8", strict=False),
            ipaddress.ip_network(f"{ip}/16", strict=False),
            ipaddress.ip_network(f"{ip}/24", strict=False)
        ]
        return [str(network.broadcast_address) for network in networks]

    def start_threads(self):
        PORT = 873
        broadcast_levels = ["/8", "/16", "/24"]
        online_ips_dict = {
            "/8": self.online_ips_8,
            "/16": self.online_ips_16,
            "/24": self.online_ips_24
        }

        for broadcast_level, broadcast_address in zip(broadcast_levels, self.subnet_broadcasts):
            listener_thread = threading.Thread(target=self.listen_for_broadcasts, args=(PORT, online_ips_dict[broadcast_level], self.update_online_ips, broadcast_level))
            listener_thread.daemon = True
            listener_thread.start()

            broadcast_thread = threading.Thread(target=self.broadcast_message, args=(PORT, f"I have the baton {broadcast_level}", broadcast_address))
            broadcast_thread.daemon = True
            broadcast_thread.start()

            manage_baton_thread = threading.Thread(target=self.manage_baton, args=(PORT, online_ips_dict[broadcast_level], self.update_baton_holder, broadcast_level, self.tz))
            manage_baton_thread.daemon = True
            manage_baton_thread.start()

            ping_thread = threading.Thread(target=self.ping_ips_same_level, args=(online_ips_dict[broadcast_level], broadcast_level))
            ping_thread.daemon = True
            ping_thread.start()

    def execute_disable_soundcard(self):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        python_script = os.path.join(current_directory, "disable_soundcard.py")
        executable = os.path.join(current_directory, "disable_soundcard.exe")

        if os.path.exists(python_script):
            os.system(f"python {python_script}")
        elif os.path.exists(executable):
            os.system(executable)

    def listen_for_broadcasts(self, port, online_ips, gui_update_func, broadcast_level):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind(('', port))
            while True:
                data, addr = sock.recvfrom(1024)
                message = data.decode()
                if f"I have the baton {broadcast_level}" in message and addr[0] not in online_ips:
                    online_ips.append(addr[0])
                gui_update_func(addr[0], message, broadcast_level)

    def broadcast_message(self, port, message, broadcast_address, interval=5):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            while True:
                sock.sendto(message.encode(), (broadcast_address, port))
                time.sleep(interval)

    def manage_baton(self, port, online_ips, gui_update_baton_func, broadcast_level, tz):
        while True:
            now = datetime.datetime.now(tz)
            next_midnight = datetime.datetime.combine(now.date() + datetime.timedelta(days=1), datetime.time(0, 0, tzinfo=tz))
            time_to_midnight = (next_midnight - now).total_seconds()
            time.sleep(time_to_midnight)

            self.execute_disable_soundcard()  # Execute disable_soundcard every midnight

            if online_ips:
                new_baton_holder = random.choice(online_ips)
                gui_update_baton_func(new_baton_holder, broadcast_level)

                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.sendto(f"You have the baton {broadcast_level}".encode(), (new_baton_holder, port))

            # Clear online_ips list for the next day
            online_ips.clear()

    def ping_ips_same_level(self, online_ips, broadcast_level):
        while True:
            time.sleep(1)
            for ip in online_ips:
                for dest_ip in online_ips:
                    if ip != dest_ip:
                        try:
                            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                                sock.settimeout(1)
                                sock.sendto("Ping".encode(), (dest_ip, 873))
                                response, _ = sock.recvfrom(1024)
                                print(f"Received response from {dest_ip}: {response.decode()}")
                        except socket.timeout:
                            print(f"No response from {dest_ip}")

def main():
    app = BroadcastApp()
    app.mainloop()

if __name__ == "__main__":
    main()

