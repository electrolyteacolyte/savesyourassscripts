' elevate.vbs
Set objShell = CreateObject("Shell.Application")
objShell.ShellExecute "cmd.exe", "/c python ""%~dp0\file_explorer.py""", "", "runas", 1
