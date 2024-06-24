import tkinter as tk
import feedparser
import webbrowser

class RSSFeedApp:
    def __init__(self, master):
        self.master = master
        master.title("World News")

        self.label = tk.Label(master, text="Latest World News Headlines", font=("Helvetica", 18))  # Increased font size
        self.label.pack(pady=10)

        self.listbox = tk.Listbox(master, width=100, height=20, font=("Helvetica", 14))  # Increased font size
        self.listbox.pack(expand=True, fill=tk.BOTH)

        self.refresh_button = tk.Button(master, text="Refresh", command=self.refresh_feed)
        self.refresh_button.pack(pady=5)

        self.refresh_feed()

        self.listbox.bind("<Double-Button-1>", self.open_link)  # Bind double click event

    def refresh_feed(self):
        self.listbox.delete(0, tk.END)  # Clear existing items

        # Fetch the RSS feed from the provided URL
        feed_url = "https://feeds.skynews.com/feeds/rss/world.xml"
        feed = feedparser.parse(feed_url)

        if 'entries' in feed:
            for entry in feed.entries:
                title = entry.title
                link = entry.link
                self.listbox.insert(tk.END, title)  # Display only titles

        self.links = [entry.link for entry in feed.entries]  # Store links for each title

    def open_link(self, event):
        selected_index = self.listbox.curselection()  # Get the index of the selected item
        if selected_index:
            if selected_index[0] < len(self.links):
                link = self.links[selected_index[0]]  # Get the link associated with the selected item
                webbrowser.open_new(link)  # Open the link in the default web browser

if __name__ == "__main__":
    root = tk.Tk()
    app = RSSFeedApp(root)
    root.mainloop()
