@echo off
REM ============================================================================
REM  CricGiri Delivery API - start a PUBLIC shareable URL (no login required).
REM  Double-click this file. It will:
REM    1) download the cloudflared tunnel tool (first run only)
REM    2) start the API server on http://localhost:8000
REM    3) open a public https URL you can share with your team
REM
REM  WARNING: this exposes THIS PC + the model to the public internet with NO
REM  auth - anyone with the link can upload videos and use your GPU/models.
REM  The URL stays alive only while this window is open and the PC is on, and it
REM  CHANGES every time you restart (free tunnel). Close the window to stop.
REM ============================================================================
setlocal
cd /d "%~dp0\.."

set "CF=deploy\bin\cloudflared.exe"
if not exist "%CF%" (
  echo Downloading cloudflared tunnel tool ...
  if not exist "deploy\bin" mkdir "deploy\bin"
  powershell -Command "Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile '%CF%' -UseBasicParsing"
)

echo.
echo Starting the API server (loading models, ~15s) ...
set "CRICGIRI_BALL_MODELS=ball_ft_t4.pt,ball_best_leather_new.pt"
set "CRICGIRI_CONF=0.05"
set "CRICGIRI_IMGSZ=1280"
start "CricGiri Pipeline API" cmd /c "venv\Scripts\python.exe -m uvicorn api.delivery_api:app --host 0.0.0.0 --port 8000"

echo Waiting for the server to come up ...
:waitloop
timeout /t 2 >nul
powershell -Command "try { (Invoke-WebRequest http://localhost:8000/health -UseBasicParsing).StatusCode } catch { exit 1 }" >nul 2>&1
if errorlevel 1 goto waitloop

echo.
echo ============================================================================
echo  Server is UP. Opening a public URL below (look for  https://xxxxx.trycloudflare.com )
echo  Share THAT url. Your team calls:
echo     POST  https://xxxxx.trycloudflare.com/analyze   (form: video, pitch_length)
echo  Interactive test page:  https://xxxxx.trycloudflare.com/docs
echo  Keep this window OPEN to keep the link alive. Close it to stop.
echo ============================================================================
echo.
"%CF%" tunnel --url http://localhost:8000
