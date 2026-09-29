@echo off
rem ---------------------------------------------------------------------------
rem  WinRunner setup: creates the "winrunner" conda environment (Python 3.11),
rem  installs the Python packages and downloads the llama.cpp engine.
rem
rem    install.bat            Vulkan engine (recommended for AMD Radeon)
rem    install.bat rocm       ROCm / HIP engine instead
rem ---------------------------------------------------------------------------
setlocal EnableExtensions
cd /d "%~dp0"
title WinRunner setup
set "BACKEND=%~1"
if "%BACKEND%"=="" set "BACKEND=vulkan"

echo.
echo  WINRUNNER  -  Local Inference Server  -  setup
echo  ============================================================
echo.

call :find_conda
if not defined CONDA_BAT (
  echo [ERROR] conda was not found.
  echo         Install Miniconda ^(https://docs.conda.io/en/latest/miniconda.html^)
  echo         or open this script from an "Anaconda Prompt".
  pause
  exit /b 1
)
echo [1/4] Using conda: %CONDA_BAT%

call "%CONDA_BAT%" env list | findstr /r /c:"^winrunner " >nul
if errorlevel 1 (
  echo [2/4] Creating conda environment "winrunner" with Python 3.11 ...
  call "%CONDA_BAT%" create -y -n winrunner python=3.11 pip
  if errorlevel 1 goto :fail
) else (
  echo [2/4] Conda environment "winrunner" already exists.
)

call "%CONDA_BAT%" activate winrunner
if errorlevel 1 goto :fail

echo [3/4] Installing Python packages ...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo [4/4] Downloading the llama.cpp engine (%BACKEND%) ...
python -m winrunner --install-engine %BACKEND%
if errorlevel 1 (
  echo [WARN] Engine download failed. You can install it later from
  echo        WinRunner ^> Settings ^> Engine.
)

echo.
echo  Setup complete.
echo    Start WinRunner:        WinRunner.bat
echo    Allow LAN access:       scripts\firewall.bat  (run as administrator)
echo    API endpoint:           http://%COMPUTERNAME%:5070/v1
echo.
pause
exit /b 0

:fail
echo.
echo [ERROR] Setup failed (see the messages above).
pause
exit /b 1

:find_conda
set "CONDA_BAT="
where conda.bat >nul 2>nul && for /f "delims=" %%P in ('where conda.bat') do if not defined CONDA_BAT set "CONDA_BAT=%%P"
if defined CONDA_BAT exit /b 0
if defined CONDA_EXE (
  for %%D in ("%CONDA_EXE%\..\..") do if exist "%%~fD\condabin\conda.bat" set "CONDA_BAT=%%~fD\condabin\conda.bat"
)
if defined CONDA_BAT exit /b 0
for %%D in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniforge3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniforge3") do (
  if not defined CONDA_BAT if exist "%%~D\condabin\conda.bat" set "CONDA_BAT=%%~D\condabin\conda.bat"
)
exit /b 0
