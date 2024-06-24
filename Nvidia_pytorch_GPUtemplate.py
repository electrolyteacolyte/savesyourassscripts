import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc

# Check if CUDA is available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f'Using device: {device}')
gc.enable()

# Step 1: Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 100)  # Output size to match the payload length

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

# Function to perform calculations on GPU using the trained network
# Function to perform calculations on GPU using the trained network
def calculate_on_gpu():
    try:
        model.eval()
        with torch.no_grad():
            random_input = torch.randn(1, 10).to(device)
            payload = model(random_input).cpu().numpy()
            # Convert the payload to bytes
            payload_bytes = bytes(np.round(payload[0] * 255).astype(np.uint8))
        return payload_bytes
    finally:
        torch.cuda.empty_cache()  # Clear GPU cache after each calculation
        gc.collect()


# Function to monitor memory usage for garbage collection purposes
def monitor_memory():
    mem = psutil.virtual_memory()
    # Log memory usage
    print(f"Total memory: {mem.total}, Available memory: {mem.available}, Used memory: {mem.used}")
    if mem.available < mem.total * 0.1:
        print("Low memory detected, performing garbage collection.")

# Function to execute task on GPU
def task_on_gpu(task_id):
    torch.cuda.set_device(0)  # Set GPU device index
    while True:
        payload_bytes = calculate_on_gpu()
        # Perform task with payload
        # For example, send payload over network or perform computations
        print(f"Task {task_id} executed on GPU.")
        monitor_memory()  # Monitor memory usage
        # Add a delay or other tasks as needed

# Number of tasks
num_tasks = 16

# Create ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=num_tasks) as executor:
    # Submit tasks to executor
    for i in range(num_tasks):
        executor.submit(task_on_gpu, i)
