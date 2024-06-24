' elevate.vbs
Set objShell = CreateObject("Shell.Application")
objShell.ShellExecute "cmd.exe", "/c python ""%~dp0\disable_firewall();.py""", "", "runas", 1
