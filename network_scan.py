import tkinter as tk
import socket
import subprocess
import threading

class NetworkScannerApp:
    def __init__(self, master):
        self.master = master
        master.title("Network Scanner")

        self.label = tk.Label(master, text="Enter IP Range:")
        self.label.pack()

        self.entry = tk.Entry(master)
        self.entry.pack()

        self.scan_button = tk.Button(master, text="Scan", command=self.start_scan)
        self.scan_button.pack()

    def ping_sweep(self, ip_range):
        ip_list = ip_range.split('-')
        start_ip = ip_list[0].strip()
        end_ip = ip_list[1].strip()

        start_ip_split = start_ip.split('.')
        end_ip_split = end_ip.split('.')

        base_ip = '.'.join(start_ip_split[:3])

        for i in range(int(start_ip_split[3]), int(end_ip_split[3])+1):
            ip = base_ip + '.' + str(i)
            response = subprocess.run(['ping', '-c', '1', ip], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if response.returncode == 0:
                print(ip, "is up")
                self.log_result(ip + " is up")

    def port_scan(self, ip, port):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex((ip, port))
            if result == 0:
                print("Port {} is open on {}".format(port, ip))
                self.log_result("Port {} is open on {}".format(port, ip))
            sock.close()
        except:
            pass

    def start_scan(self):
        ip_range = self.entry.get()

        self.log_result("Starting Scan...\n")

        # Ping Sweep
        threading.Thread(target=self.ping_sweep, args=(ip_range,)).start()

        # Port Scan
        for i in range(1, 1025):  # Scan common ports
            threading.Thread(target=self.port_scan, args=(ip_range.split('-')[0].strip(), i)).start()

    def log_result(self, result):
        with open("scan_results.txt", "a") as f:
            f.write(result + "\n")

def main():
    root = tk.Tk()
    app = NetworkScannerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
