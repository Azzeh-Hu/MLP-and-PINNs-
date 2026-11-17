@echo off
REM Double-click on Windows: runs fem/run_all.sh inside WSL (FEniCSx conda env auto-detected).
set WSLENV=FENICS_ENV/u:ONLY/u
wsl.exe --cd "%~dp0" bash -l ./run_all.sh %*
pause
