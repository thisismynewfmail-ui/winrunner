@echo off
rem Raise the Windows GPU timeout (TDR) so that long GPU jobs are not reset.
rem
rem Windows resets a GPU when one GPU job runs longer than TdrDelay (2 seconds by default).
rem llama.cpp then loses the GPU ("vk::Queue::submit: ErrorDeviceLost") and WinRunner has to
rem restart the engine. Long prompts on large models, or models partly kept in system RAM,
rem can exceed 2 seconds.
rem
rem Run as administrator, then restart Windows.
rem   gpu-timeout.bat          set the timeout to 60 seconds
rem   gpu-timeout.bat 30       set the timeout to 30 seconds (10-300)
rem   gpu-timeout.bat reset    restore the Windows defaults
setlocal
set "KEY=HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers"
set "ARG=%~1"
if "%ARG%"=="" set "ARG=60"
net session >nul 2>&1
if errorlevel 1 (
  echo This script must be run as administrator ^(right-click ^> Run as administrator^).
  pause
  exit /b 1
)
if /i "%ARG%"=="reset" goto reset
set "SECS="
for /f "delims=0123456789" %%a in ("%ARG%") do set "SECS=bad"
if defined SECS goto usage
set /a SECS=%ARG%
if %SECS% LSS 10 goto usage
if %SECS% GTR 300 goto usage
reg add "%KEY%" /v TdrDelay /t REG_DWORD /d %SECS% /f >nul
if errorlevel 1 goto failed
reg add "%KEY%" /v TdrDdiDelay /t REG_DWORD /d %SECS% /f >nul
if errorlevel 1 goto failed
echo GPU timeout set to %SECS% seconds ^(TdrDelay and TdrDdiDelay^).
echo Restart Windows for the change to take effect.
pause
exit /b 0

:reset
reg delete "%KEY%" /v TdrDelay /f >nul 2>&1
reg delete "%KEY%" /v TdrDdiDelay /f >nul 2>&1
echo GPU timeout restored to the Windows defaults ^(TdrDelay 2 s, TdrDdiDelay 5 s^).
echo Restart Windows for the change to take effect.
pause
exit /b 0

:usage
echo Usage: %~nx0 [seconds 10-300 ^| reset]    ^(default: 60 seconds^)
pause
exit /b 1

:failed
echo Failed to write the registry value.
pause
exit /b 1
