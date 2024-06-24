import socket
import threading
import time
import struct
from tkinter import Tk, Label, Button, messagebox
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 4096)  # Adjusting output size to 4096 to match payload length

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Using sigmoid to get output between 0 and 1
        return x

# Initialize the neural network
model = SimpleNet()

# Configuration
FREENET_PROXY_ADDRESS = '127.0.0.1'
FREENET_PROXY_PORT = 54356  # Default port changed to 1080
FREENET_NODE_ADDRESS = '127.0.0.1'
CLOUDFLARE_DNS = '1.1.1.1'  # Cloudflare DNS

# Dummy training data
X_train = torch.randn(1000, 10)
y_train = torch.randn(1000, 4096)

# Create a DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

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
def calculate_on_gpu(client_data, client_socket):
    try:
        model.eval()
        with torch.no_grad():
            data = torch.tensor(client_data, dtype=torch.float32)
            result = model(data)
            client_socket.sendall(result.numpy().tobytes())
    except Exception as e:
        print(f"[!] Error processing data on GPU: {e}")

# Function to resolve hostname using Cloudflare DNS
def resolve_hostname(hostname):
    resolver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    resolver.connect((CLOUDFLARE_DNS, 53))
    resolver.sendall(b'\x1d\x20\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00' + bytes(hostname, 'utf-8') + b'\x00\x00\x01\x00\x01')
    response = resolver.recv(1024)
    resolver.close()
    return socket.inet_ntoa(response[-4:])

# Function to get global IP address
def get_global_ip():
    try:
        response = requests.get("https://api.ipify.org?format=text")
        return response.text
    except Exception as e:
        return "127.0.0.1"

# Function to send heartbeat to Cloudflare DNS
def send_heartbeat():
    while True:
        try:
            resolver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            resolver.connect((CLOUDFLARE_DNS, 53))
            resolver.sendall(b'HEARTBEAT')
            resolver.close()
            time.sleep(3600)  # Send heartbeat every 60 minutes
        except Exception as e:
            print(f"Error sending heartbeat: {e}")

# Function to handle client connections
def handle_client(client_socket):
    client_socket.settimeout(60)  # Set socket timeout for heartbeat

    while True:
        try:
            # Receive data from client
            client_data = client_socket.recv(4096)
            if not client_data:
                break

            # Process data using the neural network
            calculate_on_gpu(client_data, client_socket)

            # Assuming the first line of the client_data contains the request line
            request_line = client_data.split(b'\r\n')[0].decode('utf-8')
            method, url, _ = request_line.split()
            
            if method == "CONNECT":
                host = url.split(':')[0]
                port = int(url.split(':')[1])
            else:
                host = url.split('/')[2]
                port = 80

            # Resolve hostname using Cloudflare DNS
            resolved_ip = resolve_hostname(host)

            # Connect to the actual destination server
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as internet_socket:
                internet_socket.connect((resolved_ip, port))

                # Send processed data to destination server
                internet_socket.sendall(client_data)

                while True:
                    # Receive data from destination server
                    internet_data = internet_socket.recv(4096)
                    if not internet_data:
                        break

                    # Send data back to client
                    client_socket.sendall(internet_data)
            
            # Heartbeat to keep the connection alive
            while True:
                time.sleep(60)
                client_socket.sendall(b'HEARTBEAT')
                
        except socket.timeout:
            print("[*] Connection timed out")
            break
        except Exception as e:
            print(f"[!] Error handling client: {e}")
            break

    client_socket.close()

# Main function to start the proxy server
def main():
    global_ip = get_global_ip()
    messagebox.showinfo("Proxy Information", f"Proxy is running on IP: {global_ip}, Port: {FREENET_PROXY_PORT}\nGlobal IP: {global_ip}")

    # Start heartbeat thread
    heartbeat_thread = threading.Thread(target=send_heartbeat)
    heartbeat_thread.start()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((FREENET_PROXY_ADDRESS, FREENET_PROXY_PORT))
    server.listen(5)
    print(f'[*] Listening on {FREENET_PROXY_ADDRESS}:{FREENET_PROXY_PORT}')

    while True:
        client_socket, addr = server.accept()
        print(f'[*] Accepted connection from {addr[0]}:{addr[1]}')
        client_handler = threading.Thread(target=handle_client, args=(client_socket,))
        client_handler.start()

if __name__ == '__main__':
    main()
