@echo off
REM Double-click on Windows: opens the Linux (WSL) terminal in this folder,
REM activates the FEniCSx conda environment and runs niti_wire_fem.py.
REM To use a specific env name, set FENICS_ENV below (leave empty to auto-detect).
set FENICS_ENV=

REM Pass FENICS_ENV through to WSL.
set WSLENV=FENICS_ENV/u

where wt >nul 2>nul
if %errorlevel%==0 (
    wt wsl.exe --cd "%~dp0" bash -l ./run_niti_fem.sh %*
) else (
    start "FEniCSx - niti_wire_fem" wsl.exe --cd "%~dp0" bash -l ./run_niti_fem.sh %*
)
