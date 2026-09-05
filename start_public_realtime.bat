@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Please install Python 3.11 or newer and enable Add Python to PATH.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo First run: creating the Python environment...
  py -m venv ".venv"
  if errorlevel 1 (
    echo Failed to create the Python environment.
    pause
    exit /b 1
  )
  echo First run: installing Python dependencies. This may take several minutes...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r "requirements-online.txt"
  if errorlevel 1 (
    echo Failed to install Python dependencies.
    pause
    exit /b 1
  )
)

where cloudflared >nul 2>nul
if errorlevel 1 if not exist "%~dp0cloudflared.exe" (
  echo cloudflared is not installed.
  echo Download cloudflared-windows-amd64.exe, rename it to cloudflared.exe,
  echo and put it in this folder or add it to PATH.
  echo Download page: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
  pause
  exit /b 1
)

echo Starting local Python YOLOv8 + ArUco + PnP backend...
start "Dzhanibekov Python backend" cmd /k ""%~dp0.venv\Scripts\python.exe" "%~dp0realtime_server.py" --source web --port 8000 --https --yolo-imgsz 416 --yolo-every 6"
timeout /t 6 /nobreak >nul

echo Starting public HTTPS tunnel...
if exist "%~dp0cloudflared.exe" (
  "%~dp0cloudflared.exe" tunnel --url https://127.0.0.1:8000 --no-tls-verify
) else (
  cloudflared tunnel --url https://127.0.0.1:8000 --no-tls-verify
)

echo.
echo Keep both windows open while measuring.
pause
