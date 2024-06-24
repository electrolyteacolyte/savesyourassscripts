import socket
import threading
import time
import tkinter as tk
from tkinter import scrolledtext

# Configuration
FREENET_PROXY_ADDRESS = '127.0.0.1'
FREENET_PROXY_PORT = 8888
FREENET_NODE_ADDRESS = '127.0.0.1'
FREENET_NODE_PORT = 9481  # Default Freenet node port
COMMON_PORTS = [80, 443, 21, 22, 25, 110, 143, 993, 995, 3306, 3389, 5900, 8080]
CLOUDFLARE_DNS = '1.1.1.1'  # Cloudflare DNS

# Function to resolve hostname using Cloudflare DNS
def resolve_hostname(hostname):
    resolver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    resolver.connect((CLOUDFLARE_DNS, 53))
    resolver.sendall(b'\x1d\x20\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00' + bytes(hostname, 'utf-8') + b'\x00\x00\x01\x00\x01')
    response = resolver.recv(1024)
    resolver.close()
    return socket.inet_ntoa(response[-4:])

# Function to handle client connections
def handle_client(client_socket, client_addr, listbox):
    client_socket.settimeout(60)  # Set socket timeout for heartbeat
    listbox.insert(tk.END, f'[*] Client connected: {client_addr[0]}:{client_addr[1]}')
    listbox.yview(tk.END)

    while True:
        try:
            # Receive data from client
            client_data = client_socket.recv(4096)
            if not client_data:
                break

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

                # Send data to destination server
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
            listbox.insert(tk.END, f'[*] Connection timed out: {client_addr[0]}:{client_addr[1]}')
            listbox.yview(tk.END)
            break
        except Exception as e:
            print(f"[!] Error handling client: {e}")
            listbox.insert(tk.END, f'[!] Error: {e}')
            listbox.yview(tk.END)
            break

    client_socket.close()
    listbox.insert(tk.END, f'[*] Client disconnected: {client_addr[0]}:{client_addr[1]}')
    listbox.yview(tk.END)

# Function to start a server on a given port
def start_server(port, listbox):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((FREENET_PROXY_ADDRESS, port))
    server_socket.listen(5)
    listbox.insert(tk.END, f'[*] Listening on {FREENET_PROXY_ADDRESS}:{port}')
    listbox.yview(tk.END)

    while True:
        client_socket, client_addr = server_socket.accept()
        listbox.insert(tk.END, f'[*] Accepted connection from {client_addr[0]}:{client_addr[1]} on port {port}')
        listbox.yview(tk.END)
        client_handler = threading.Thread(target=handle_client, args=(client_socket, client_addr, listbox))
        client_handler.start()

def start_servers(listbox):
    for port in COMMON_PORTS:
        server_thread = threading.Thread(target=start_server, args=(port, listbox))
        server_thread.daemon = True
        server_thread.start()

    start_server(FREENET_PROXY_PORT, listbox)

def create_gui():
    root = tk.Tk()
    root.title("Proxy Server")

    start_button = tk.Button(root, text="Start Server", command=lambda: start_servers(log_listbox))
    start_button.pack()

    stop_button = tk.Button(root, text="Stop Server", state=tk.DISABLED)
    stop_button.pack()

    log_listbox = scrolledtext.ScrolledText(root, width=100, height=30)
    log_listbox.pack()

    root.mainloop()

if __name__ == '__main__':
    create_gui()
