import socket
import ssl
import tkinter as tk
from tkinter import messagebox, scrolledtext
import threading
import requests

class ChatApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Peer-to-Peer Chat")

        self.username = self.get_public_ip()  # Default username is the public IP address
        self.connections = {}  # Dictionary to hold active connections
        self.max_connections = 10  # Maximum allowed connections per user

        print("Your Public IP Address:", self.username)  # Print the public IP address
        self.label = tk.Label(self, text="Your Public IP Address: " + self.username)
        self.label.pack(padx=10, pady=10)

        self.chat_frame = tk.Frame(self)
        self.chat_text = scrolledtext.ScrolledText(self.chat_frame, width=50, height=20)
        self.chat_text.pack(padx=10, pady=10)
        self.entry = tk.Entry(self.chat_frame, width=40)
        self.entry.pack(side=tk.LEFT, padx=10, pady=5)
        self.send_button = tk.Button(self.chat_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.clear_button = tk.Button(self.chat_frame, text="Clear Chat", command=self.clear_chat)
        self.clear_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.chat_frame.pack(pady=10)

        self.connect_frame = tk.Frame(self)
        self.connect_label = tk.Label(self.connect_frame, text="Enter IP Address to Connect:")
        self.connect_label.pack(padx=10, pady=5)
        self.connect_entry = tk.Entry(self.connect_frame, width=30)
        self.connect_entry.pack(pady=5)
        self.connect_button = tk.Button(self.connect_frame, text="Connect", command=self.connect_to_ip)
        self.connect_button.pack(pady=5)
        self.connect_frame.pack(pady=10)

        self.dns_frame = tk.Frame(self)
        self.dns_label = tk.Label(self.dns_frame, text="Enter DNS Server (optional):")
        self.dns_label.pack(padx=10, pady=5)
        self.dns_entry = tk.Entry(self.dns_frame, width=30)
        self.dns_entry.pack(pady=5)
        self.dns_frame.pack(pady=10)

        self.start_button = tk.Button(self, text="Start to listen for connections", command=self.start_server)
        self.start_button.pack(pady=5)

        self.stop_button = tk.Button(self, text="Stop listening for connections", command=self.stop_server)
        self.stop_button.pack(pady=5)
        self.stop_button.config(state=tk.DISABLED)

        self.server_thread = None

    def start_server(self):
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        self.server_thread = threading.Thread(target=self.server_loop)
        self.server_thread.start()

    def stop_server(self):
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.server_thread.join()

    def server_loop(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('127.0.0.1', 1080))
        server_socket.listen(5)
        print("Server is listening...")

        while True:
            client_socket, addr = server_socket.accept()
            if len(self.connections) < self.max_connections:
                response = messagebox.askyesno("Incoming Connection", f"{addr} wants to connect. Accept?")
                if response:
                    self.accept_connection(client_socket, addr)
            else:
                client_socket.close()
                messagebox.showinfo("Connection Rejected", "Maximum connections reached.")
                print("Maximum connections reached.")

    def accept_connection(self, client_socket, addr):
        self.connections[addr] = client_socket
        print(f"Connection accepted from {addr}")
        threading.Thread(target=self.handle_client, args=(client_socket, addr)).start()

    def handle_client(self, client_socket, addr=None):
        try:
            ssl_client_socket = self.create_tls_socket(client_socket, addr)
            while True:
                data = ssl_client_socket.recv(1024)
                if not data:
                    break
                if addr is not None:
                    print(f"Received from {addr}: {data.decode()}")
                else:
                    print(f"Received from server: {data.decode()}")
        finally:
            ssl_client_socket.close() if addr is not None else client_socket.close()
            if addr is not None:
                del self.connections[addr]
                print(f"Connection with {addr} closed.")

    def send_message(self):
        message = self.entry.get()
        self.chat_text.insert(tk.END, f"You: {message}\n")
        self.entry.delete(0, tk.END)

    def connect_to_ip(self):
        ip_address = self.connect_entry.get()
        dns_server = self.dns_entry.get()
        threading.Thread(target=self.connect, args=(ip_address, dns_server)).start()

    def connect(self, ip_address, dns_server):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((ip_address, 1080))
            print(f"Connected to {ip_address}")
            ssl_client_socket = self.create_tls_socket(client_socket, (ip_address, 1080), dns_server)
            self.handle_client(ssl_client_socket)
        except socket.error as e:
            print(f"Failed to connect to {ip_address}: {e}")
        finally:
            client_socket.close()

    def create_tls_socket(self, client_socket, addr, dns_server=None):
        context = ssl.create_default_context()
        if dns_server:
            context.resolver = ssl.RFCOMMANDS["RFC7858_1"](host=dns_server)
        return context.wrap_socket(client_socket, server_hostname=addr[0])

    def clear_chat(self):
        self.chat_text.delete('1.0', tk.END)

    def get_public_ip(self):
        try:
            response = requests.get("https://api.ipify.org")
            return response.text
        except Exception as e:
            print("Failed to fetch public IP:", e)
            return "Unknown"

if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()
