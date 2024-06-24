import tkinter as tk
import threading
import socket
import os

class Tracker:
    def __init__(self):
        self.host = '127.0.0.1'
        self.port = 5555
        self.peers = set()

    def start(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen(5)
        print("Tracker started on {}:{}".format(self.host, self.port))
        while True:
            client, addr = self.server.accept()
            print("Connection from:", addr)
            threading.Thread(target=self.handle_client, args=(client,)).start()

    def handle_client(self, client):
        while True:
            data = client.recv(1024).decode('utf-8')
            if not data:
                break
            if data == "JOIN":
                self.peers.add(client)
                print("Peer {} joined".format(client.getpeername()))
                client.send("ACK".encode('utf-8'))
            elif data == "LIST":
                client.send(str(self.peers).encode('utf-8'))
            elif data.startswith("GET"):
                filename = data.split()[1]
                if os.path.exists(filename):
                    client.send("FOUND".encode('utf-8'))
                    with open(filename, 'rb') as file:
                        while True:
                            chunk = file.read(1024)
                            if not chunk:
                                break
                            client.send(chunk)
                    client.send("DONE".encode('utf-8'))
                else:
                    client.send("NOTFOUND".encode('utf-8'))
        client.close()

class Peer:
    def __init__(self, tracker_host, tracker_port):
        self.host = '127.0.0.1'
        self.port = 5556
        self.tracker_host = tracker_host
        self.tracker_port = tracker_port

    def start(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen(5)
        print("Peer started on {}:{}".format(self.host, self.port))
        threading.Thread(target=self.listen_to_peers).start()
        self.register_with_tracker()

    def listen_to_peers(self):
        while True:
            client, addr = self.server.accept()
            print("Connection from:", addr)
            threading.Thread(target=self.handle_peer_requests, args=(client,)).start()

    def handle_peer_requests(self, client):
        while True:
            data = client.recv(1024).decode('utf-8')
            if not data:
                break
            if data == "FOUND":
                print("File found, receiving...")
                with open('received_file', 'wb') as file:
                    while True:
                        data = client.recv(1024)
                        if data == b'DONE':
                            break
                        file.write(data)
                print("File received successfully")
            elif data == "NOTFOUND":
                print("File not found on any peer")
        client.close()

    def register_with_tracker(self):
        try:
            tracker_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tracker_socket.connect((self.tracker_host, self.tracker_port))
            tracker_socket.send("JOIN".encode('utf-8'))
            response = tracker_socket.recv(1024).decode('utf-8')
            if response == "ACK":
                print("Registered with tracker successfully")
            else:
                print("Failed to register with tracker")
        except Exception as e:
            print("Error connecting to tracker:", e)

    def search_file(self, filename):
        try:
            tracker_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tracker_socket.connect((self.tracker_host, self.tracker_port))
            tracker_socket.send("GET {}".format(filename).encode('utf-8'))
            response = tracker_socket.recv(1024).decode('utf-8')
            if response == "FOUND":
                print("File found, downloading...")
                with open(filename, 'wb') as file:
                    while True:
                        data = tracker_socket.recv(1024)
                        if data == b'DONE':
                            break
                        file.write(data)
                print("File downloaded successfully")
            else:
                print("File not found")
        except Exception as e:
            print("Error searching for file:", e)

def start_tracker():
    tracker = Tracker()
    tracker.start()

def start_peer():
    peer = Peer(tracker_host_entry.get(), int(tracker_port_entry.get()))
    peer.start()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Decentralized File Sharing")

    tracker_host_label = tk.Label(root, text="Tracker Host:")
    tracker_host_label.grid(row=0, column=0, padx=5, pady=5)
    tracker_host_entry = tk.Entry(root)
    tracker_host_entry.grid(row=0, column=1, padx=5, pady=5)

    tracker_port_label = tk.Label(root, text="Tracker Port:")
    tracker_port_label.grid(row=1, column=0, padx=5, pady=5)
    tracker_port_entry = tk.Entry(root)
    tracker_port_entry.grid(row=1, column=1, padx=5, pady=5)

    tracker_button = tk.Button(root, text="Start Tracker", command=start_tracker)
    tracker_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

    peer_button = tk.Button(root, text="Start Peer", command=start_peer)
    peer_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

    root.mainloop()
