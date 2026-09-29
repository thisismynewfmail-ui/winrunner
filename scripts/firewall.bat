@echo off
rem Allow other computers on the private (home) network to reach the WinRunner API.
rem Run as administrator. Optional argument: port (default 5070).
setlocal
set "PORT=%~1"
if "%PORT%"=="" set "PORT=5070"
net session >nul 2>&1
if errorlevel 1 (
  echo This script must be run as administrator ^(right-click ^> Run as administrator^).
  pause
  exit /b 1
)
netsh advfirewall firewall delete rule name="WinRunner API (TCP %PORT%)" >nul 2>&1
netsh advfirewall firewall add rule name="WinRunner API (TCP %PORT%)" dir=in action=allow protocol=TCP localport=%PORT% profile=private
if errorlevel 1 (
  echo Failed to add the firewall rule.
  pause
  exit /b 1
)
echo Inbound TCP port %PORT% is now allowed on private networks.
pause
