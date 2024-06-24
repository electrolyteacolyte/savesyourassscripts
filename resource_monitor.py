import psutil
import os
import subprocess
import gc  # Import gc module
import tkinter as tk

class ResourceMonitor(tk.Tk):
    update_interval = 1500  # Update interval in milliseconds
    
    def __init__(self):
        super().__init__()
        self.title("Resource Monitor")
        self.set_window_size()
        
        self.cpu_label = tk.Label(self, text="CPU Usage: ")
        self.cpu_label.pack()
        
        self.memory_label = tk.Label(self, text="Memory Usage: ")
        self.memory_label.pack()
        
        self.gpu_label = tk.Label(self, text="GPU Workload: ")
        self.gpu_label.pack()
        
        self.gpu_temp_label = tk.Label(self, text="GPU Temperature: ")
        self.gpu_temp_label.pack()
        
        self.clean_button = tk.Button(self, text="Clean RAM", command=self.clean_ram)
        self.clean_button.pack()
        
        self.update_labels()
    
    def set_window_size(self):
        title_length = len(self.title())
        self.geometry(f"{title_length * 20}x150")  # Set fixed window size
        
    def update_labels(self):
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        gpu_info, gpu_temp = self.get_gpu_info_and_temp()
        
        self.cpu_label.config(text=f"CPU Usage: {cpu_percent:.2f}%")
        self.memory_label.config(text=f"Memory Usage: {memory_percent:.2f}%")
        self.gpu_label.config(text=f"GPU Workload: {gpu_info}%")
        self.gpu_temp_label.config(text=f"GPU Temperature: {gpu_temp}°C")
        
        self.after(self.update_interval, self.update_labels)
    
    def get_gpu_info_and_temp(self):
        try:
            output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'], creationflags=subprocess.CREATE_NO_WINDOW)
            gpu_info = output.decode('utf-8').strip().split(', ')
            gpu_utilization = gpu_info[0] + "%"  # Adding '%' character
            gpu_temp = gpu_info[1]
            return gpu_utilization, gpu_temp
        except subprocess.CalledProcessError:
            return "N/A (NVIDIA GPU not found)", "N/A"
    
    def clean_ram(self):
        gc.collect()  # Invoke the garbage collector
        print("Garbage Collected!")
        
        # Clean OS RAM by clearing disk cache
        os.system("sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches")
        print("OS RAM Cleaned!")
        
        # Clean physical RAM
        self.clear_physical_ram()
        print("RAM Cleaned!")  
    
    def clear_physical_ram(self):
        # Get total RAM and used RAM
        total_ram = psutil.virtual_memory().total
        used_ram = psutil.virtual_memory().used
        
        # Calculate available RAM with a safety margin of 128 MB
        available_ram = total_ram - used_ram - (128 * 1024**2)
        
        # Allocate a large chunk of memory to fill up the remaining available RAM
        large_list_size = available_ram // 8  # Each element in the list occupies 8 bytes
        large_list = [0] * large_list_size
        
        # Clear the list to deallocate memory
        del large_list

if __name__ == "__main__":
    app = ResourceMonitor()
    app.mainloop()
