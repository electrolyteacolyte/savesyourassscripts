import socket
import threading
import time
import struct
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random

# Check if CUDA is available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f'Using device: {device}')

# Step 1: Define a simple neural network
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
model = SimpleNet().to(device)

# Configuration
VPN_SERVER_ADDRESS = '0.0.0.0'  # Listen on all available interfaces
VPN_SERVER_PORT = 1194  # OpenVPN default port
CLOUDFLARE_DNS = '1.1.1.1'  # Cloudflare DNS

# Dummy training data
X_train = torch.randn(1000, 10).to(device)
y_train = torch.randn(1000, 4096).to(device)

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
            data = torch.tensor(client_data, dtype=torch.float32).to(device)
            result = model(data)
            client_socket.sendall(result.cpu().numpy().tobytes())
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

# Function to handle VPN client connections
def handle_vpn_client(client_socket):
    while True:
        try:
            # Receive data from client
            client_data = client_socket.recv(4096)
            if not client_data:
                break

            # Process data using the neural network on GPU
            calculate_on_gpu(client_data, client_socket)
                
        except Exception as e:
            print(f"[!] Error handling VPN client: {e}")
            break

    client_socket.close()

# Main function to start the VPN server
def main():
    # Start VPN server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((VPN_SERVER_ADDRESS, VPN_SERVER_PORT))
    server.listen(5)
    print(f'[*] VPN server listening on {VPN_SERVER_ADDRESS}:{VPN_SERVER_PORT}')

    while True:
        client_socket, addr = server.accept()
        print(f'[*] Accepted VPN connection from {addr[0]}:{addr[1]}')
        client_handler = threading.Thread(target=handle_vpn_client, args=(client_socket,))
        client_handler.start()

if __name__ == '__main__':
    main()
