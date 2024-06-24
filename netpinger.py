import tkinter as tk
from tkinter import filedialog
import subprocess
import threading

class PingApp:
    def __init__(self, master):
        self.master = master
        master.title("Ping Tool")

        self.label_ip_file = tk.Label(master, text="IP Address List File:")
        self.label_ip_file.pack()

        self.ip_file_entry = tk.Entry(master, state='disabled')
        self.ip_file_entry.pack()

        self.load_button = tk.Button(master, text="Select IP List", command=self.select_ip_list)
        self.load_button.pack()

        self.label_packet_count = tk.Label(master, text="Packet Count:")
        self.label_packet_count.pack()

        self.packet_count_entry = tk.Entry(master)
        self.packet_count_entry.pack()

        self.label_packet_size = tk.Label(master, text="Packet Size:")
        self.label_packet_size.pack()

        self.packet_size_entry = tk.Entry(master)
        self.packet_size_entry.pack()

        self.label_message = tk.Label(master, text="Custom Message:")
        self.label_message.pack()

        self.message_entry = tk.Entry(master)
        self.message_entry.pack()

        self.start_button = tk.Button(master, text="Start Pinging", command=self.start_pinging, state=tk.DISABLED)
        self.start_button.pack()

        self.stop_button = tk.Button(master, text="Stop Pinging", command=self.stop_pinging, state=tk.DISABLED)
        self.stop_button.pack()

        self.ping_process = None
        self.ping_thread = None
        self.running = False

    def select_ip_list(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            self.ip_file_entry.config(state='normal')
            self.ip_file_entry.delete(0, tk.END)
            self.ip_file_entry.insert(0, file_path)
            self.start_button.config(state=tk.NORMAL)

    def start_pinging(self):
        ip_file = self.ip_file_entry.get()
        packet_count = self.packet_count_entry.get() or 5
        packet_size = self.packet_size_entry.get() or 64
        custom_message = self.message_entry.get() or "Ping"

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.running = True

        def ping():
            with open(ip_file, 'r') as file:
                ip_addresses = file.read().splitlines()
            while self.running:
                for ip in ip_addresses:
                    command = ['ping', '-c', str(packet_count), '-s', str(packet_size), '-W', '1', ip]
                    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    output = result.stdout.decode('utf-8')
                    print(output)
                    # You can process the output further if needed

        self.ping_thread = threading.Thread(target=ping)
        self.ping_thread.start()

    def stop_pinging(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

root = tk.Tk()
app = PingApp(root)
root.mainloop()
