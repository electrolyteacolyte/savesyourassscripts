from tkinter import Tk, Label, Button, filedialog, Listbox, END, Text
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc
import socket
import os
import uuid
import subprocess
import platform
from scapy.layers.inet import IP, ICMP, traceroute
from scapy.all import srp, Ether, IP, ICMP
import json
from pathlib import Path
import requests
import multiprocessing as mp
import queue

# Preallocate 8 GB of RAM
fixed_memory = np.zeros((4 * 1024 * 1024 * 1024 // np.dtype(np.uint8).itemsize,), dtype=np.uint8)  # 8 GB

gc.enable()
# List to store loaded IPs and their filenames
ips_dict = {}
# Get the user's Documents folder
documents_folder = Path(os.path.expanduser('~')) / 'Documents'

# Create a folder for measurements inside Documents
measurements_folder = documents_folder / 'measurements'
measurements_folder.mkdir(parents=True, exist_ok=True)

# Define the file path for handshake results
handshake_file_path = measurements_folder / 'handshake_results.json'
portscan_file_path = measurements_folder/ 'portscan_results.json'
pingsweep_file_path = measurements_folder/ 'pingsweep_results.json'
traceroute_file_path = measurements_folder/ 'traceroute_results.json'
broadcast_file_path = measurements_folder/ 'broadcast_results.json'
query_file_path = measurements_folder/ 'query_results.json'

def scan_ports(ip, ports=[80, 443, 22, 21, 25, 8080]):
    open_ports = []

    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((ip, port))
            if result == 0:
                open_ports.append(port)

    # Save results to JSON file
    save_results_to_json0(ip, open_ports)

def pingsweep():
    pingsweep_results = []
    
    # Craft the ICMP echo request packet
    icmp_request = IP(dst=str(ip)) / ICMP()

    # Send the ICMP echo request and receive response
    reply = sr1(icmp_request, timeout=1, verbose=False)

    # Check if the host is reachable (received ICMP echo reply)
    if reply and reply.haslayer(ICMP) and reply.getlayer(ICMP).type == 0:
        pingsweep_results.append({"ip": str(ip), "status": "reachable"})
    else:
        pingsweep_results.append({"ip": str(ip), "status": "unreachable"})

# Save results to JSON file
    save_results_to_json1(pingsweep_results)

def perform_traceroute():
    try:
        traceroute_results = []

        # Perform traceroute
        hops = traceroute(ip)

        # Store traceroute results
        for hop in hops:
            traceroute_results.append({"ip": hop[1].src})

    except Exception as e:
        print(f"Error saving traceroute results: {e}")

    # Append traceroute results to JSON file
        save_results_to_json2(traceroute_results)

def send_heartbeat():
    handshake_methods = {
        "TCP": [80, 443, 25, 110, 143, 22, 21, 389, 636, 23, 3389, 5900],
        "UDP": [53, 161, 162, 123, 514],
        # Add more handshake methods here
    }

    global_ip = get_global_ip()  # Fetch global IP dynamically

    handshake_results = []

    try:
        for method_name, ports in handshake_methods.items():
            for port in ports:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handshake_socket:
                        handshake_socket.settimeout(1)  # Set a timeout of 1 second
                        result = handshake_socket.connect_ex((ip, port))
                        if result == 0:  # If the connection is successful
                            handshake_socket.sendall(global_ip.encode())
                            # Log successful handshake
                            handshake_results.append({
                                "ip": ip,
                                "method": method_name,
                                "port": port,
                                "status": "Success"
                            })
                            print(f"Heartbeat sent to {ip} using {method_name} ({port})")
                            log_handshake(ip, global_ip)  # Assuming log_handshake is defined elsewhere
                        else:
                            # Print failed handshake
                            handshake_results.append({
                                "ip": ip,
                                "method": method_name,
                                "port": port,
                                "status": "Failed: Connection refused"
                            })
                            print(f"Failed to send heartbeat to {ip} using {method_name} ({port}): Connection refused")
                except Exception as e:
                    # Print failed handshake
                    handshake_results.append({
                        "ip": ip,
                        "method": method_name,
                        "port": port,
                        "status": f"Failed: {str(e)}"
                    })
                    print(f"Failed to send heartbeat to {ip} using {method_name} ({port}): {str(e)}")
    except Exception as e:
        # Print failed handshake
        print(f"Failed to send heartbeat to {ip}: {str(e)}")

    # Save handshake results to JSON file
    save_handshake_results(handshake_results)
        


def broadcast_ip():
    broadcast_results = []  # Initialize an empty list to store broadcast results
    try:
        # Get the hostname of the local machine
        hostname = socket.gethostname()
        
        # Get the IP address associated with the hostname
        ip_address = socket.gethostbyname(hostname)
        
        global_ip1 = get_global_ip()  # Fetch global IP dynamically
        
        # Predefined ports for broadcasting
        handshake_methods = {
            "TCP": [80, 443, 25, 110, 143, 22, 21, 389, 636, 23, 3389, 5900],
            "UDP": [53, 161, 162, 123, 514],
            # Add more handshake methods here
        }
        
        # Log successful or failed broadcast for each protocol and port
        for protocol, ports in handshake_methods.items():
            for port in ports:
                try:
                    # Construct broadcast message
                    broadcast_message = f"My IP address is {ip_address} on port {port} using {protocol} protocol"
                    
                    # Create a UDP socket
                    if protocol == 'UDP':
                        sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        sock2.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                    # Create a TCP socket
                    elif protocol == 'TCP':
                        sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock2.bind(('', 0))  # Bind to any available port
                        
                    # Broadcast the message on the specified port
                    sock2.sendto(broadcast_message.encode(), (global_ip1, port))
                    sock2.close()
                    
                    print(f"Broadcasted IP address successfully on port {port} using {protocol} protocol")
                    # Log successful broadcast
                    broadcast_results.append({
                        "ip": ip_address,
                        "port": port,
                        "protocol": protocol,
                        "status": "Success"
                    })
                except Exception as e:
                    print(f"Failed to broadcast IP address on port {port} using {protocol} protocol: {str(e)}")
                    # Log failed broadcast
                    broadcast_results.append({
                        "ip": ip_address,
                        "port": port,
                        "protocol": protocol,
                        "status": f"Failed: {str(e)}"
                    })
        
    except Exception as e:
        print(f"Error broadcasting IP address: {e}")
        
    save_results_to_json3(broadcast_results)

def query_ip():
    query_results = []  # Initialize an empty list to store query results
    try:
        # Get the hostname of the local machine
        hostname = socket.gethostname()
        
        # Get the IP address associated with the hostname
        ip_address = socket.gethostbyname(hostname)
        
        # Predefined ports for querying
        query_ports = {
            "TCP": [80, 443, 25, 110, 143, 22, 21, 389, 636, 23, 3389, 5900],
            "UDP": [53, 161, 162, 123, 514],
            # Add more handshake methods here
        }
        
        # Log successful or failed query for each protocol and port
        for protocol, ports in query_ports.items():
            for port in ports:
                try:
                    # Create a TCP socket
                    sock1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock1.settimeout(1)  # Set timeout for connection attempt
                    
                    # Attempt to connect to the IP address on the specified port
                    result = sock1.connect_ex((ip_address, port))
                    
                    if result == 0:  # If the connection is successful
                        print(f"Connection established to {ip_address} on port {port} using {protocol} protocol")
                        # Log successful query
                        query_results.append({
                            "ip": ip_address,
                            "port": port,
                            "protocol": protocol,
                            "status": "Success"
                        })
                    else:
                        print(f"Failed to connect to {ip_address} on port {port} using {protocol} protocol: Connection refused")
                        # Log failed query
                        query_results.append({
                            "ip": ip_address,
                            "port": port,
                            "protocol": protocol,
                            "status": "Failed: Connection refused"
                        })
                    
                    sock1.close()
                    
                except Exception as e:
                    print(f"Failed to connect to {ip_address} on port {port} using {protocol} protocol: {str(e)}")
                    # Log failed query
                    query_results.append({
                        "ip": ip_address,
                        "port": port,
                        "protocol": protocol,
                        "status": f"Failed: {str(e)}"
                    })
        
    except Exception as e:
        print(f"Error querying IP address: {e}")
        
    save_results_to_json3(query_results)
    
def save_results_to_json4(query_results):
    try:
        with open(query_file_path, 'a') as query_file:
            json.dump({"results": query_results}, query_file, indent=4)
            query_file.write('\n')
    except Exception as e:
        print(f"Error saving query results: {e}")
        
def save_results_to_json3(broadcast_results):
    try:
        with open(broadcast_file_path, 'a') as broadcast_file:
            json.dump({"ip": ip, "results": broadcast_results}, broadcast_file, indent=4)
            broadcast_file.write('\n')
    except Exception as e:
        print(f"Error saving broadcast results: {e}")
        
def save_results_to_json0(portscan_results):
    
    try:
        with open(portscan_file_path, 'a') as portscan_file:
            json.dump({"ip": ip, "open_ports": open_ports}, portscan_file, indent=4)
            portscan_file.write('\n')


    except Exception as e:
        print(f"Error saving portscan results: {e}")


def save_results_to_json1(pingsweep_results):
    try:
        with open(pingsweep_file_path, 'a') as pingsweep_file:
            json.dump({"results": pingsweep_results}, pingsweep_file, indent=4)
            pingsweep_file.write('\n')

    except Exception as e:
        print(f"Error saving handshake results: {e}") 
        

def save_results_to_json2(traceroute_results):
    try:
        with open(traceroute_file_path, 'a') as traceroute_file:
            json.dump({"results": traceroute_results}, traceroute_file, indent=4)
            traceroute_file.write('\n')

    except Exception as e:
        print(f"Error saving handshake results: {e}")    
             

def save_handshake_results(handshake_results):

    try:
        with open(handshake_file_path, 'a') as handshake_file:
            json.dump(handshake_results, handshake_file, indent=4)
            handshake_file.write('\n')

    except Exception as e:
        print(f"Error saving handshake results: {e}")

def get_global_ip():
    try:
        response = requests.get("https://api.ipify.org?format=text")
        return response.text.strip()  # Remove any leading or trailing whitespace
    except Exception as e:
        print(f"Error getting global IP: {e}")
        return "127.0.0.1"



def load_ip_file():
    global ips_dict
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'r') as file:
            new_ips = file.readlines()
        new_ips = [ip.strip() for ip in new_ips if ip.strip()]
        if file_path not in ips_dict:
            ips_dict[file_path] = new_ips
            loaded_files_box.insert(END, file_path)
            print(f"Loaded IPs from file {file_path}: {new_ips}")
        else:
            print(f"Ignore file {file_path} as it's already loaded.")
    else:
        print("No file selected.")

def initiate_tasks(tasks):
    max_workers = 128
    with ThreadPoolExecutor(max_workers) as executor:
        futures = [executor.submit(task) for task in tasks]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Thread generated an exception: {e}")

def initiate_start():
    tasks = []
    for file_path, ips in ips_dict.items():
        for ip in ips:
            tasks.append(lambda ip=ip: perform_traceroute(ip))
            tasks.append(lambda ip=ip: pingsweep(ip))
            tasks.append(lambda ip=ip: scan_ports(ip))
            tasks.append(lambda ip=ip: send_heartbeat(ip))
            tasks.append(lambda ip=ip: query_ip(ip))
            tasks.append(lambda ip=ip: broadcast_ip(ip))
    initiate_tasks(tasks)
                
root = Tk()
root.title("networkmap")

# Define the Text widget to display processes
processes_text = Text(root, height=20, width=50)
processes_text.grid(row=6, columnspan=2)

load_file_button = Button(root, text="Load IPs from File", command=load_ip_file)
load_file_button.grid(row=3, columnspan=2)

loaded_files_label = Label(root, text="Files Loaded:")
loaded_files_label.grid(row=4, column=0)
loaded_files_box = Listbox(root, width=40)
loaded_files_box.grid(row=4, column=1)

start_traceroute_button = Button(root, text="Start", command=initiate_start)
start_traceroute_button.grid(row=22, columnspan=2)



root.mainloop()

