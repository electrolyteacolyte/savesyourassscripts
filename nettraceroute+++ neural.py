from tkinter import Tk, Label, Button, filedialog, Listbox, END, Text
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
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
import dns.resolver
import requests
from queue import Queue
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Check if CUDA is available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f'Using device: {device}')

# List to store loaded IPs and their filenames
ips_dict = {}

handshake_socket = None
scan_ports_socket = None
broadcast_socket1 = None
broadcast_socket2 = None
query_socket = None
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
dnsrecords_file_path = measurements_folder/ 'dnsrecords_results.json'
http_file_path = measurements_folder/ 'http_results.json'




class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 100)
        global handshake_socket
        global scan_ports_socket
        global broadcast_socket1
        global broadcast_socket2
        global query_socket# Output size to match the payload length

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



def calculate_on_gpu():
    model.eval()
    with torch.no_grad():
        handshake_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        handshake_socket.settimeout(1)  # Set a timeout of 1 second
        result = handshake_socket.connect_ex((ip, port))
    return result
        
def calculate_on_gpu0():
    model.eval()
    with torch.no_grad():
        scan_ports_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
def calculate_on_gpu1():
    model.eval()
    with torch.no_grad():
        broadcast_socket1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        broadcast_socket1.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
def calculate_on_gpu2():
    model.eval()
    with torch.no_grad():  
        broadcast_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        broadcast_socket2.bind(('', 0))  # Bind to any available port
    
def calculate_on_gpu3():
    model.eval()
    with torch.no_grad():  
        query_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        query_socket.settimeout(1)  # Set timeout for connection attempt
        
def scan_ports(ip, ports=[80, 443, 22, 21, 25, 8080]):
    open_ports = []
    global scan_ports_socket
    for port in ports:
        with calculate_on_gpu0():
            sock.settimeout(1)
            result = sock.connect_ex((ip, port))
            if result == 0:
                open_ports.append(port)
     

    # Save results to JSON file
    save_results_to_json0(ip, open_ports)

def pingsweep(ip_list):
    pingsweep_results = []

    for ip in ip_list:
        icmp_request = IP(dst=str(ip)) / ICMP()
        reply = sr1(icmp_request, timeout=1, verbose=False)

        if reply and reply.haslayer(ICMP) and reply.getlayer(ICMP).type == 0:
            pingsweep_results.append({"ip": str(ip), "status": "reachable"})
        else:
            pingsweep_results.append({"ip": str(ip), "status": "unreachable"})
     
    save_results_to_json1(pingsweep_results)

def perform_traceroute(ip):
    try:
        traceroute_results = []
        buffer_ips = []

        # Perform traceroute
        hops = traceroute(ip, maxttl=255)[0]

        # Store traceroute results and prepare IPs for pingsweep
        for i, hop in enumerate(hops, start=1):
            hop_ip = hop[1].src
            traceroute_results.append({"hop": i, "ip": hop_ip})
            buffer_ips.append(hop_ip)
            
            # If the number of hops exceeds 25, pingsweep every 25th hop
            if len(hops) > 25 and i % 25 == 0:
                perform_traceroute(hop_ip)

       

        # Perform pingsweep on all hops
        pingsweep(buffer_ips)
         
    except Exception as e:
        print(f"Error performing traceroute: {e}")
    save_results_to_json2(traceroute_results)
def send_heartbeat(ip):
    handshake_methods = {
        "TCP": [80, 443, 25, 110, 143, 22, 21, 389, 636, 23, 3389, 5900],
        "UDP": [53, 161, 162, 123, 514],
        # Add more handshake methods here
    }

    global_ip = get_global_ip()  # Fetch global IP dynamically

    handshake_results = []
    global handshake_socket
    try:
        for method_name, ports in handshake_methods.items():
            for port in ports:
                try:
                    with calculate_on_gpu():
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
        


def broadcast_ip(ip):
    broadcast_results = []
    global broadcast_socket1
    global broadcast_socket2
    # Initialize an empty list to store broadcast results
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
                        calculate_on_gpu1()
                    # Create a TCP socket
                    elif protocol == 'TCP':
                        calculate_on_gpu2()
                    
                    # Broadcast the message on the specified port
                    broadcast_socket1.sendto(broadcast_message.encode(), (global_ip1, port))
                    broadcast_socket1.close()
                    broadcast_socket2.sendto(broadcast_message.encode(), (global_ip1, port))
                    broadcast_socket2.close()
                
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

def query_ip(ip):
    query_results = []  # Initialize an empty list to store query results
    global query_socket
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
                    calculate_on_gpu3()
                    
                    # Attempt to connect to the IP address on the specified port
                    result = query_socket.connect_ex((ip_address, port))
                    
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
    
      
    
def get_dns_records(ip):
    records = {}
    record_types = ['A', 'AAAA', 'MX', 'NS', 'TXT', 'CNAME', 'SOA']

    for record_type in record_types:
        try:
            answers = dns.resolver.resolve(domain, record_type)
            records[record_type] = [answer.to_text() for answer in answers]
        except dns.resolver.NoAnswer:
            records[record_type] = []
        except dns.resolver.NXDOMAIN:
            records[record_type] = None
            print(f"The domain '{domain}' does not exist.")
            break
        except Exception as e:
            records[record_type] = None
            print(f"An error occurred while fetching {record_type} records: {e}")

    return records
     
    save_results_to_json5(records)


def get_domain_name(ip):
    try:
        return socket.gethostbyaddr(ip)[0]
    except Exception as e:
        print(f"Error getting domain name for {ip}: {e}")
        return None
        
def perform_http_requests(ip):
    global_ip = get_global_ip()
    if not global_ip:
        return
    for ip in ip_list:
        domain_name = get_domain_name(ip)
    if not domain_name:
        print(f"Skipping IP {ip} as no domain name found.")
        return

    http_results = []
    url = f"http://{domain_name}"
    for method in methods:
        try:
            response = requests.request(method, url, json=data, headers=headers, timeout=1)
            http_results.append({
                "global_ip": global_ip,
                "domain_name": domain_name,
                "url": url,
                "method": method,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.text
            })

        except requests.RequestException as e:
            print(f"Error performing HTTP {method} request to {url}: {e}")
     
    save_results_to_json6(http_results)


def save_results_to_json6(http_results):
    try:
        with open(http_file_path, 'a') as http_file:
            json.dump({"results": http_results}, http_file, indent=4)
            http_file.write('\n')
    except Exception as e:
        print(f"Error saving query results: {e}")

        
def save_results_to_json5(records):
    try:
        with open(dnsrecords_file_path, 'a') as dnsrecords_file:
            json.dump({"results": records}, dnsrecords_file, indent=4)
            dnsrecords_file.write('\n')
    except Exception as e:
        print(f"Error saving query results: {e}")

        
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
    max_workers = 512
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
            tasks.append(lambda ip=ip: get_dns_records(ip))
            tasks.append(lambda ip=ip: perform_http_requests(ip))
            
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

