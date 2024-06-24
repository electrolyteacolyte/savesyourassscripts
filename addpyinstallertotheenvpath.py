import os

def add_to_path(directory):
    """
    Add the specified directory to the PATH environment variable.
    """
    # Get the current value of the PATH variable
    path_variable = os.environ.get('PATH', '')

    # Split the PATH variable into individual directories
    path_dirs = path_variable.split(os.pathsep)

    # Check if the directory is already in the PATH
    if directory in path_dirs:
        print(f"The directory '{directory}' is already in the PATH.")
        return

    # Add the directory to the list of directories
    path_dirs.append(directory)

    # Join the directories back into a single string with the appropriate separator
    new_path = os.pathsep.join(path_dirs)

    # Set the modified PATH variable
    os.environ['PATH'] = new_path

    print(f"The directory '{directory}' has been added to the PATH.")

if __name__ == "__main__":
    # Directory where PyInstaller is installed
    pyinstaller_directory = r'c:\users\szaba\appdata\roaming\python\python311\site-packages\'  # Adjust this path

    # Add PyInstaller directory to the PATH
    add_to_path(pyinstaller_directory)
