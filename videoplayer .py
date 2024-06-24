import os
import tkinter as tk
from tkinter import filedialog
import av
import numpy as np
import threading

class VideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Player")

        # Initialize variables
        self.videos = []
        self.current_video_index = 0
        self.playing = False
        self.stop_requested = False
        self.volume = 50  # Initial volume level

        # Create main frame for buttons and volume control
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side="top", fill="x")

        # Create Select Folder button
        self.select_folder_button = tk.Button(self.button_frame, text="Select Folder", command=self.select_video_folder)
        self.select_folder_button.pack(side="left")

        # Create Refresh Folder button
        self.refresh_folder_button = tk.Button(self.button_frame, text="Refresh Folder", command=self.refresh_video_folder)
        self.refresh_folder_button.pack(side="left")

        # Create Previous button
        self.prev_button = tk.Button(self.button_frame, text="Previous", command=self.show_previous_video)
        self.prev_button.pack(side="left")

        # Create Play button
        self.play_button = tk.Button(self.button_frame, text="Play", command=self.play_selected_video)
        self.play_button.pack(side="left")

        # Create Stop button
        self.stop_button = tk.Button(self.button_frame, text="Stop", command=self.stop_video)
        self.stop_button.pack(side="left")

        # Create Next button
        self.next_button = tk.Button(self.button_frame, text="Next", command=self.show_next_video)
        self.next_button.pack(side="left")

        # Create Volume control
        self.volume_label = tk.Label(self.button_frame, text="Volume:")
        self.volume_label.pack(side="left")
        self.volume_slider = tk.Scale(self.button_frame, from_=0, to=100, orient="horizontal", command=self.set_volume)
        self.volume_slider.set(self.volume)
        self.volume_slider.pack(side="left")

        # Create main frame for video display
        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        # Create label for video display
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Create frame for folder navigation
        self.folder_frame = tk.Frame(self.root)
        self.folder_frame.pack(side="left", fill="y")

        # Create listbox for video files in the folder
        self.video_listbox = tk.Listbox(self.folder_frame)
        self.video_listbox.pack(side="left", fill="y")

        # Create scrollbar for the listbox
        self.scrollbar = tk.Scrollbar(self.folder_frame, orient="vertical")
        self.scrollbar.pack(side="right", fill="y")
        self.video_listbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.video_listbox.yview)
        self.video_listbox.bind("<<ListboxSelect>>", self.on_select_video)

    def select_video_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            if folder_path != getattr(self, 'selected_folder', None):
                self.selected_folder = folder_path
                self.load_videos(folder_path)
                self.update_video_listbox()

    def load_videos(self, folder_path):
        self.videos.clear()
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith((".mp4", ".avi", ".mov")):
                video_path = os.path.join(folder_path, file_name)
                self.videos.append(video_path)

    def update_video_listbox(self):
        self.video_listbox.delete(0, tk.END)
        for video in self.videos:
            self.video_listbox.insert(tk.END, os.path.basename(video))

    def refresh_video_folder(self):
        if hasattr(self, 'selected_folder'):
            self.load_videos(self.selected_folder)
            self.update_video_listbox()

    def on_select_video(self, event):
        selection = event.widget.curselection()
        if selection:
            self.current_video_index = selection[0]

    def play_selected_video(self):
        if not self.playing:
            self.playing = True
            video_path = self.videos[self.current_video_index]
            self.stop_requested = False
            threading.Thread(target=self.play_video, args=(video_path,)).start()

    def play_video(self, video_path):
        container = av.open(video_path)
        for frame in container.decode(video=0):
            if self.stop_requested:
                break
            frame_image = frame.to_image()
            frame_np = np.array(frame_image)
            frame_tk = self.rgb_to_tkimage(frame_np)
            # Schedule the update of the label widget in the main GUI thread
            self.root.after(10, self.update_video_label, frame_tk)
        container.close()
        self.playing = False

    def update_video_label(self, frame_tk):
        # Configure the label widget with the new image
        self.video_label.config(image=frame_tk)
        self.video_label.image = frame_tk

    def stop_video(self):
        self.stop_requested = True

    def show_next_video(self):
        if self.current_video_index < len(self.videos) - 1:
            self.current_video_index += 1
            self.video_listbox.select_clear(0, tk.END)
            self.video_listbox.select_set(self.current_video_index)
            self.video_listbox.see(self.current_video_index)

    def show_previous_video(self):
        if self.current_video_index > 0:
            self.current_video_index -= 1
            self.video_listbox.select_clear(0, tk.END)
            self.video_listbox.select_set(self.current_video_index)
            self.video_listbox.see(self.current_video_index)

    def set_volume(self, volume):
        self.volume = int(volume)

    @staticmethod
    def rgb_to_tkimage(rgb_image):
        import PIL.Image
        import PIL.ImageTk
        return PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(rgb_image))

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPlayer(root)
    root.mainloop()
