from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume

def enable_sound_card():
    sessions = AudioUtilities.GetAllSessions()
    if not sessions:
        print("No active audio sessions found.")
        return
    
    for session in sessions:
        volume = session._ctl.QueryInterface(ISimpleAudioVolume)
        if volume:
            volume.SetMute(0, None)  # Unmute the session
            volume.SetMasterVolume(1.0, None)  # Set volume to maximum
        else:
            print("Failed to retrieve volume control for session.")

if __name__ == "__main__":
    enable_sound_card()
