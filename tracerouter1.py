import tkinter as tk
from tkinter import filedialog
from scapy.all import traceroute
import socket
import os

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    entry_file_path.delete(0, tk.END)
    entry_file_path.insert(0, file_path)

def scan_ports(ip, ports=[80, 443, 22, 21, 25, 8080]):
    open_ports = []
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((ip, port))
            if result == 0:
                open_ports.append(port)
    return open_ports

def perform_traceroute():
    file_path = entry_file_path.get()
    if not file_path:
        text_output.insert(tk.END, "Please select a file.\n")
        return
    
    desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    output_file_path = os.path.join(desktop_path, 'traceroute_and_port_scan_results.txt')
    
    with open(file_path, 'r') as file, open(output_file_path, 'w') as output_file:
        ips = [line.strip() for line in file.readlines()]
        for ip in ips:
            output_file.write(f"Traceroute results for {ip}:\n")
            text_output.insert(tk.END, f"Traceroute results for {ip}:\n")
            
            try:
                res, _ = traceroute(ip, verbose=0)
                for snd, rcv in res:
                    output_file.write(f"{snd.ttl} {rcv.src}\n")
                    text_output.insert(tk.END, f"{snd.ttl} {rcv.src}\n")
            except Exception as e:
                output_file.write(f"Traceroute error: {e}\n")
                text_output.insert(tk.END, f"Traceroute error: {e}\n")
            
            output_file.write("\nPort scan results:\n")
            text_output.insert(tk.END, "\nPort scan results:\n")
            
            try:
                open_ports = scan_ports(ip)
                if open_ports:
                    for port in open_ports:
                        output_file.write(f"Port {port} is open\n")
                        text_output.insert(tk.END, f"Port {port} is open\n")
                else:
                    output_file.write("No open ports found.\n")
                    text_output.insert(tk.END, "No open ports found.\n")
            except Exception as e:
                output_file.write(f"Port scan error: {e}\n")
                text_output.insert(tk.END, f"Port scan error: {e}\n")
            
            output_file.write("\n")
            text_output.insert(tk.END, "\n")

    text_output.insert(tk.END, f"Results written to {output_file_path}\n")

app = tk.Tk()
app.title("Traceroute and Port Scan")

frame = tk.Frame(app)
frame.pack(pady=10)

label_file_path = tk.Label(frame, text="Select IP List File:")
label_file_path.grid(row=0, column=0, padx=5)

entry_file_path = tk.Entry(frame, width=50)
entry_file_path.grid(row=0, column=1, padx=5)

button_browse = tk.Button(frame, text="Browse", command=select_file)
button_browse.grid(row=0, column=2, padx=5)

button_start = tk.Button(app, text="Start", command=perform_traceroute)
button_start.pack(pady=10)

text_output = tk.Text(app, height=20, width=80)
text_output.pack(pady=10)

app.mainloop()
