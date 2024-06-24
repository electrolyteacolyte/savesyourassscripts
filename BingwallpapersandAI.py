import os
import requests
import webbrowser
from datetime import datetime

def download_bing_wallpaper():
    # Create directory if it doesn't exist
    os.makedirs("bing_wallpapers", exist_ok=True)
    
    # Get today's date in YYYY-MM-DD format
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    # URL to get Bing wallpaper of the day
    bing_url = f"https://www.bing.com/HPImageArchive.aspx?format=js&idx=0&n=1&mkt=en-US"
    
    # Send GET request to Bing API
    response = requests.get(bing_url)
    
    # Check if request was successful
    if response.status_code == 200:
        # Get image URL from response
        image_url = response.json()["images"][0]["url"]
        
        # Download image
        image_response = requests.get(f"https://www.bing.com{image_url}")
        if image_response.status_code == 200:
            with open(f"bing_wallpapers/{today_date}.jpg", "wb") as f:
                f.write(image_response.content)
            print("Bing wallpaper downloaded successfully.")
        else:
            print("Failed to download Bing wallpaper.")
    else:
        print("Failed to fetch Bing wallpaper.")

def open_bing_ai():
    # URL to Bing's AI-powered search engine
    bing_ai_url = "https://www.bing.com/chat"
    
    # Open URL in default web browser
    webbrowser.open(bing_ai_url)
    print("Opening Bing AI in browser...")

def main():
    download_bing_wallpaper()
    open_bing_ai()

if __name__ == "__main__":
    main()
