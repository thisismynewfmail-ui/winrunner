@echo off
rem Start WinRunner with a console window showing the application log.
rem   WinRunner-Console.bat              control panel in a native window
rem   WinRunner-Console.bat --headless   API server only (no control panel window)
rem   WinRunner-Console.bat --browser    control panel in the web browser
setlocal EnableExtensions
cd /d "%~dp0"
title WinRunner
call :find_conda
if not defined CONDA_BAT (
  echo conda not found - run install.bat first.
  pause
  exit /b 1
)
call "%CONDA_BAT%" activate winrunner || (echo Environment "winrunner" missing - run install.bat & pause & exit /b 1)
python -m winrunner %*
if errorlevel 1 pause
exit /b 0

:find_conda
set "CONDA_BAT="
where conda.bat >nul 2>nul && for /f "delims=" %%P in ('where conda.bat') do if not defined CONDA_BAT set "CONDA_BAT=%%P"
if defined CONDA_BAT exit /b 0
if defined CONDA_EXE for %%D in ("%CONDA_EXE%\..\..") do if exist "%%~fD\condabin\conda.bat" set "CONDA_BAT=%%~fD\condabin\conda.bat"
if defined CONDA_BAT exit /b 0
for %%D in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniforge3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniforge3") do (
  if not defined CONDA_BAT if exist "%%~D\condabin\conda.bat" set "CONDA_BAT=%%~D\condabin\conda.bat"
)
exit /b 0
