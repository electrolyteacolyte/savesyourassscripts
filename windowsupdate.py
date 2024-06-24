import win32com.client
import os
import time

def activate_windows_update_service():
    update_session = win32com.client.Dispatch("Microsoft.Update.Session")
    update_searcher = update_session.CreateUpdateSearcher()
    search_result = update_searcher.Search("IsInstalled=0 and Type='Software' and IsHidden=0")

    updates_to_download = []
    for update in search_result.Updates:
        if "Update for Windows" in update.Title:
            updates_to_download.append(update)

    if updates_to_download:
        print("Downloading updates...")
        downloader = update_session.CreateUpdateDownloader()
        downloader.Updates = updates_to_download
        downloader.Download()
        print("Updates downloaded.")

        print("Installing updates...")
        installer = update_session.CreateUpdateInstaller()
        installer.Updates = updates_to_download
        installation_result = installer.Install()
        
        if installation_result.ResultCode == 2:  # Requires restart
            print("Updates installed. Restarting the computer...")
            os.system("shutdown /r /t 0")
        else:
            print("Updates installed successfully.")
    else:
        print("No updates available.")

if __name__ == "__main__":
    activate_windows_update_service()
