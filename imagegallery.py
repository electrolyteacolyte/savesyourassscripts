import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import screeninfo

class ImageGallery:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Gallery")

        # Initialize variables
        self.images = []
        self.current_image_index = 0
        self.fullscreen = False
        self.display_modes = ["Fit to Screen", "Original Size"]
        self.selected_display_mode = tk.StringVar(value=self.display_modes[0])

        # Get screen information
        self.screen_info = screeninfo.get_monitors()

        # Create main frame for buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side="top", fill="x")

        # Create Select Folder button
        self.select_folder_button = tk.Button(self.button_frame, text="Select Folder", command=self.select_image_folder)
        self.select_folder_button.pack(side="left")

        # Create Previous button
        self.prev_button = tk.Button(self.button_frame, text="Previous", command=self.show_previous_image)
        self.prev_button.pack(side="left")

        # Create Next button
        self.next_button = tk.Button(self.button_frame, text="Next", command=self.show_next_image)
        self.next_button.pack(side="left")

        # Create Display Selector
        self.display_selector = tk.OptionMenu(self.button_frame, self.selected_display_mode, *self.display_modes, command=self.change_display_mode)
        self.display_selector.pack(side="left")

        # Create Select Monitor Selector
        self.monitor_options = [f"Monitor {i+1}" for i in range(len(self.screen_info))]
        self.selected_monitor = tk.StringVar(value=self.monitor_options[0])
        self.monitor_selector = tk.OptionMenu(self.button_frame, self.selected_monitor, *self.monitor_options, command=self.select_monitor)
        self.monitor_selector.pack(side="left")

        # Create main frame for image display
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        # Create label for image display
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

    def select_image_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            if folder_path != getattr(self, 'selected_folder', None):
                self.selected_folder = folder_path
                self.load_images(folder_path)
                self.show_image()

    def load_images(self, folder_path):
        self.images.clear()
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(folder_path, file_name)
                self.images.append(Image.open(image_path))

    def show_image(self):
        if not self.images:
            return

        if 0 <= self.current_image_index < len(self.images):
            self.resize_image()
            self.update_navigation_buttons()

    def resize_image(self):
        image = self.images[self.current_image_index]
        if self.selected_display_mode.get() == "Fit to Screen":
            width = self.image_frame.winfo_width() - 20  # Adjust for padding
            height = self.image_frame.winfo_height() - 20  # Adjust for padding
            resized_image = image.resize((width, height), Image.LANCZOS)  # Use LANCZOS for resizing
        else:
            resized_image = image
        self.displayed_image = ImageTk.PhotoImage(resized_image)
        self.image_label.config(image=self.displayed_image)

    def update_navigation_buttons(self):
        pass  # No need to update navigation buttons in this layout

    def change_display_mode(self, mode):
        self.show_image()

    def show_next_image(self):
        if self.current_image_index < len(self.images) - 1:
            self.current_image_index += 1
            self.show_image()

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image()

    def select_monitor(self, monitor_name):
        monitor_index = self.monitor_options.index(monitor_name)
        selected_monitor = self.screen_info[monitor_index]
        self.root.geometry(f"{selected_monitor.width}x{selected_monitor.height}+{selected_monitor.x}+{selected_monitor.y}")
        self.root.attributes('-fullscreen', False)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageGallery(root)
    root.mainloop()
