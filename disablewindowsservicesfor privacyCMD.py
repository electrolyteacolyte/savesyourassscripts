import subprocess

def disable_unnecessary_services():
    try:
        # List of unnecessary Windows services to be disabled
        services_to_disable = [
            "wercplsupport",  # Problem Reports and Solutions Control Panel Support
            "DiagTrack",  # Connected User Experiences and Telemetry
            "WbioSrvc",  # Windows Biometric Service
            "WerSvc",  # Windows Error Reporting Service
            "tapisrv",  # Telephony
            "PeerDistSvc",  # BranchCache
            # Add more service names as needed
        ]
        
        for service in services_to_disable:
            # Stop and disable each service
            subprocess.run(["sc", "stop", service], check=True)
            subprocess.run(["sc", "config", service, "start=", "disabled"], check=True)
            print(f"Disabled and stopped {service} service.")
    except subprocess.CalledProcessError as e:
        print("Error disabling unnecessary services:", e)

if __name__ == "__main__":
    disable_unnecessary_services()
