@echo off
rem ---------------------------------------------------------------------------
rem  WinRunner setup: creates the "winrunner" conda environment (Python 3.11),
rem  installs the Python packages and downloads the llama.cpp engine.
rem
rem    install.bat            Vulkan engine (recommended for AMD Radeon)
rem    install.bat rocm       ROCm / HIP engine instead
rem
rem  Python is installed from the conda-forge channel only. Anaconda's default
rem  channels (repo.anaconda.com) require accepting Anaconda's Terms of Service,
rem  which makes non-interactive installs fail with CondaToSNonInteractiveError;
rem  WinRunner does not need them.
rem ---------------------------------------------------------------------------
setlocal EnableExtensions
cd /d "%~dp0"
title WinRunner setup
set "BACKEND=%~1"
if "%BACKEND%"=="" set "BACKEND=vulkan"
set "ENV_NAME=winrunner"
set "CHANNELS=--override-channels -c conda-forge"

echo.
echo  WINRUNNER  -  Local Inference Server  -  setup
echo  ============================================================
echo.

call :find_conda
if not defined CONDA_BAT (
  echo [ERROR] conda was not found.
  echo         Install Miniconda or Miniforge ^(https://conda-forge.org/download/^)
  echo         or run this script from an "Anaconda Prompt".
  goto :fail_nomsg
)
echo [1/4] Using conda: %CONDA_BAT%

rem ---- environment ------------------------------------------------------------
call "%CONDA_BAT%" env list | findstr /r /c:"^%ENV_NAME% " >nul
if errorlevel 1 (
  echo [2/4] Creating conda environment "%ENV_NAME%" with Python 3.11 from conda-forge ...
  call "%CONDA_BAT%" create -y -n %ENV_NAME% %CHANNELS% python=3.11 pip
  if errorlevel 1 goto :fail_create
) else (
  echo [2/4] Conda environment "%ENV_NAME%" already exists.
)

call "%CONDA_BAT%" activate %ENV_NAME%
if errorlevel 1 goto :fail

rem An earlier failed run can leave an empty environment: install Python into it.
if not exist "%CONDA_PREFIX%\python.exe" (
  echo       The environment has no Python yet - installing Python 3.11 from conda-forge ...
  call "%CONDA_BAT%" install -y -n %ENV_NAME% %CHANNELS% python=3.11 pip
  if errorlevel 1 goto :fail_create
  call "%CONDA_BAT%" activate %ENV_NAME%
)
if not exist "%CONDA_PREFIX%\python.exe" goto :fail_create

"%CONDA_PREFIX%\python.exe" -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)"
if errorlevel 1 (
  echo [ERROR] The "%ENV_NAME%" environment has an old Python version.
  echo         Remove it with:  conda env remove -n %ENV_NAME%   and run install.bat again.
  goto :fail_nomsg
)

rem ---- packages ---------------------------------------------------------------
echo [3/4] Installing Python packages ...
"%CONDA_PREFIX%\python.exe" -m pip install --upgrade pip
"%CONDA_PREFIX%\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

rem ---- engine -----------------------------------------------------------------
echo [4/4] Downloading the llama.cpp engine (%BACKEND%) ...
"%CONDA_PREFIX%\python.exe" -m winrunner --install-engine %BACKEND%
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

:fail_create
echo.
echo [ERROR] Could not create the Python environment from conda-forge.
echo         Check the internet connection and try again. If your network blocks
echo         conda-forge, accept Anaconda's channel terms instead and re-run:
echo           conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
echo           conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
echo           conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
echo         then:  conda create -y -n %ENV_NAME% python=3.11 pip
goto :fail_nomsg

:fail
echo.
echo [ERROR] Setup failed (see the messages above).

:fail_nomsg
echo.
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
