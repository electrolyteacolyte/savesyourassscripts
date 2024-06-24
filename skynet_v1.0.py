import socket
import threading
import os

# Server configuration
HOST = '0.0.0.0'   # Listen on all available network interfaces
PORT = 1080        # Port to listen on

# Unique UUID for identification/authentication
UUID = "9128fd2e-b2d5-4350-b81d-73701190a81d"

# Function to handle individual client connections
def handle_client(client_socket, client_address):
    print(f"Connected to {client_address}")
    try:
        # Receive UUID from client
        received_uuid = client_socket.recv(1024).decode('utf-8')

        if received_uuid == UUID:
            print("Authentication successful.")
            # Start a new thread to handle sending messages to the client
            send_thread = threading.Thread(target=send_message, args=(client_socket,))
            send_thread.start()
        else:
            print("Authentication failed.")
            return

        while True:
            # Receive message from client
            message = client_socket.recv(1024).decode('utf-8')
            if not message:
                break

            print(f"Received from {client_address}: {message}")
            if message.startswith("run_pyfile:"):
                filename = message.split(":")[1]
                execute_file_from_desktop(filename)

    except Exception as e:
        print(f"Error: {e}")

    print(f"Disconnected from {client_address}")
    client_socket.close()

# Function to execute a Python file from the desktop
def execute_file_from_desktop(filename):
    try:
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_path = os.path.join(desktop_path, filename)
        if os.path.exists(file_path) and file_path.endswith('.py'):
            print(f"Executing {filename}...")
            os.system(f"python \"{file_path}\"")
        else:
            print(f"File '{filename}' does not exist on the desktop or is not a Python file.")
    except Exception as e:
        print(f"Error executing file: {e}")

# Function to start the server and listen for incoming connections
def start_server():
    #print("Hi, I'm a server. With the clone of myself somebody can run a Python file from my computer, and I can do the same with the clone of myself.")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()

        print(f"Server listening on {HOST}:{PORT}")

        while True:
            # Accept incoming connection
            client_socket, client_address = server_socket.accept()

            # Start a new thread to handle client
            client_thread = threading.Thread(
                target=handle_client, args=(client_socket, client_address))
            client_thread.start()

# Function to send message to the connected client
def send_message(client_socket):
    while True:
        try:
            # Input message to send
            message = input("Enter a message to send: ")
            client_socket.send(message.encode('utf-8'))
        except Exception as e:
            print(f"Error: {e}")
            break

# Function to prompt for user input
def prompt_user_input():
    # User input for IP address
    ip_address = input("Enter the IP address of the remote machine: ")

    # User input for filename
    filename = input("Enter the filename of the Python file to execute (with extension): ")

    # Connect as client to another server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            client_socket.connect((ip_address, PORT))
            print("Connected to server.")

            # Send UUID to authenticate
            client_socket.send(UUID.encode('utf-8'))

            # Start a new thread to handle receiving messages from the server
            receive_thread = threading.Thread(target=receive_message, args=(client_socket,))
            receive_thread.start()

            # Send messages to server
            while True:
                message = input("Enter a message to send to the server: ")
                client_socket.send(message.encode('utf-8'))

        except Exception as e:
            print(f"Error: {e}")

# Function to receive messages from the server
def receive_message(client_socket):
    while True:
        try:
            # Receive message from server
            message = client_socket.recv(1024).decode('utf-8')
            if not message:
                break

            print(f"Received from server: {message}")

        except Exception as e:
            print(f"Error: {e}")
            break

# Main function
if __name__ == "__main__":
    # Start the server in a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.start()

    # Prompt for user input in the main thread
    prompt_user_input()
