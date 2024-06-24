import subprocess

def run_command(command):
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing: {command}")
        print(e)
        exit(1)

# Install PyTorch and torchvision
run_command("pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118")

# Install pyphysx from the GitHub repository
run_command("pip install --upgrade git+https://github.com/petrikvladimir/pyphysx.git@master")

print("Installation completed successfully.")