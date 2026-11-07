@echo off
cd /d "%~dp0"
python spatial_simulation.py
if errorlevel 1 (
echo.
echo If numpy or scipy is missing, run: python -m pip install -r requirements.txt
pause
)
