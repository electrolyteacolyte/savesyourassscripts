import tkinter as tk
import cv2
from PIL import Image, ImageTk
import os
import datetime

class WebcamApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Create a folder to save photos
        self.photo_dir = "photos"
        if not os.path.exists(self.photo_dir):
            os.makedirs(self.photo_dir)
        
        # Check existing files in the folder to avoid duplicates
        existing_files = os.listdir(self.photo_dir)
        self.used_names = set(existing_files)
        
        # Open the camera
        self.cap = cv2.VideoCapture(0)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()
        
        # Button to capture a photo
        self.btn_snapshot = tk.Button(window, text="Take a Photo", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)
        
        self.update()
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def snapshot(self):
        # Capture a frame from the camera
        ret, frame = self.cap.read()
        
        if ret:
            # Generate a unique file name based on current timestamp
            now = datetime.datetime.now()
            file_name = f"photo_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            
            # Check if the file name is already used
            while file_name in self.used_names:
                now = datetime.datetime.now()
                file_name = f"photo_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                
            # Save the frame as an image
            file_path = os.path.join(self.photo_dir, file_name)
            cv2.imwrite(file_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            print(f"Photo saved as {file_path}")
            
            # Update the set of used names
            self.used_names.add(file_name)
        else:
            print("Error: Unable to capture photo.")
        
    def update(self):
        # Get a frame from the camera
        ret, frame = self.cap.read()
        
        if ret:
            # Convert the frame to RGB format
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert the frame to ImageTk format
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(image))
            
            # Display the frame on the canvas
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        # Repeat this call every 15 milliseconds
        self.window.after(15, self.update)
        
    def on_close(self):
        # Release the camera when closing the window
        self.cap.release()
        self.window.destroy()

# Create a window and pass it to the WebcamApp class
root = tk.Tk()
app = WebcamApp(root, "Webcam App")
root.mainloop()
