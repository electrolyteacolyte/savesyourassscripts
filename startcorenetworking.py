import subprocess

def check_service_status(service_name):
    try:
        result = subprocess.run(["sc", "query", service_name], capture_output=True, text=True)
        if "RUNNING" in result.stdout:
            print(f"Service '{service_name}' is running.")
        elif "STOPPED" in result.stdout:
            print(f"Service '{service_name}' is stopped.")
        else:
            print(f"Unable to determine the status of service '{service_name}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error while checking service '{service_name}' status: {e}")

def check_corenetworking_services_status():
    services = ["dot3svc", "dhcp", "dnscache", "nlasvc", "nlaSvc"]
    for service in services:
        check_service_status(service)

if __name__ == "__main__":
    check_corenetworking_services_status()
