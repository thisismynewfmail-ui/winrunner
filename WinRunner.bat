@echo off
rem Start WinRunner in its own window (no console). Extra arguments are passed through,
rem e.g.  WinRunner.bat --model qwen3-32b-q4_k_m
setlocal EnableExtensions
cd /d "%~dp0"
call :find_conda
if not defined CONDA_BAT (
  echo conda not found - run install.bat first.
  pause
  exit /b 1
)
call "%CONDA_BAT%" activate winrunner || (echo Environment "winrunner" missing - run install.bat & pause & exit /b 1)
start "WinRunner" pythonw -m winrunner --window %*
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
