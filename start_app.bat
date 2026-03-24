@echo off
cd /d "%~dp0"

echo === Stopping any existing processes on ports 8000 and 5173 ===
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5173 ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo === Starting backend (FastAPI) hidden ===
set "BACKEND_CMD=cd /d %~dp0 && uvicorn api:app --reload --host 0.0.0.0 --port 8000"
echo Set objShell = CreateObject("WScript.Shell") > "%temp%\run_backend.vbs"
echo objShell.Run "cmd /c %BACKEND_CMD%", 0, False >> "%temp%\run_backend.vbs"
cscript //nologo "%temp%\run_backend.vbs"

timeout /t 3 /nobreak >nul

echo === Starting frontend (React/Vite) hidden ===
set "FRONTEND_CMD=cd /d %~dp0frontend && npm run dev -- --host 0.0.0.0"
echo Set objShell = CreateObject("WScript.Shell") > "%temp%\run_frontend.vbs"
echo objShell.Run "cmd /c %FRONTEND_CMD%", 0, False >> "%temp%\run_frontend.vbs"
cscript //nologo "%temp%\run_frontend.vbs"

timeout /t 3 /nobreak >nul

echo.
echo === Both servers running silently in background ===
echo.
echo On this computer:
echo   App:   http://localhost:5173
echo   API:   http://localhost:8000/docs
echo.
echo On your phone (home wifi only):
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address" ^| findstr /v "127.0.0.1"') do (
    set "IP=%%a"
    setlocal enabledelayedexpansion
    set "IP=!IP: =!"
    echo   Phone: http://!IP!:5173
    endlocal
)
echo.
echo To stop the servers, run Task Manager and kill
echo uvicorn.exe and node.exe
echo.
pause