@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Project virtual environment was not found:
  echo %~dp0.venv\Scripts\python.exe
  pause
  exit /b 1
)

echo Starting browser-camera pose server with local HTTPS...
start "Dzhanibekov browser camera server" cmd /k ""%~dp0.venv\Scripts\python.exe" "%~dp0realtime_server.py" --source web --port 8000 --https --yolo-imgsz 416 --yolo-every 6"
timeout /t 5 /nobreak >nul
start "Dzhanibekov dashboard" "https://127.0.0.1:8000/"
echo.
echo On the phone, open the https URL printed in the server window.
echo Accept the browser certificate warning, then click Start phone camera.
echo Keep this server window open.
pause
