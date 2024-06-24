import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from scapy.all import sniff
from threading import Thread

class PacketCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Packet Capture App")
        
        self.start_button = tk.Button(root, text="Start Capturing", command=self.start_capture)
        self.start_button.pack(pady=5)
        
        self.stop_button = tk.Button(root, text="Stop Capturing", command=self.stop_capture, state=tk.DISABLED)
        self.stop_button.pack(pady=5)
        
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_text)
        self.clear_button.pack(pady=5)
        
        self.text_box = ScrolledText(root, wrap=tk.WORD, width=80, height=20)
        self.text_box.pack(padx=5, pady=5)
        
        self.sniff_thread = None
        self.stop_sniffing = False
    
    def start_capture(self):
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.sniff_thread = Thread(target=self.capture_packets)
        self.sniff_thread.start()
    
    def stop_capture(self):
        self.stop_sniffing = True
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
    
    def clear_text(self):
        self.text_box.delete('1.0', tk.END)
    
    def capture_packets(self):
        self.stop_sniffing = False
        self.text_box.insert(tk.END, "Capturing packets...\n")
        sniff(prn=self.packet_handler)
        self.text_box.insert(tk.END, "Packet capture stopped.\n")
    
    def packet_handler(self, packet):
        if not self.stop_sniffing:
            self.text_box.insert(tk.END, packet.summary() + "\n")
            self.text_box.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = PacketCaptureApp(root)
    root.mainloop()
