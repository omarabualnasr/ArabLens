@echo off
setlocal
cd /d "%~dp0"
echo Starting the chat assistant...
echo If this is the first run, required Python packages will be installed.
py -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Could not install dependencies. Make sure Python is installed and the py command works.
  pause
  exit /b 1
)
echo.
echo Your browser will open automatically at http://127.0.0.1:8000
py app.py
pause
