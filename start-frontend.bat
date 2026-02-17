@echo off
echo Starting Training Data Generator Frontend...
echo.
cd /d "%~dp0frontend"
start "" http://localhost:5173
npm run dev
pause
