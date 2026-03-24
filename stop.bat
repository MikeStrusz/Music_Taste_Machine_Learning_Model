@echo off
echo === Stopping Album Rankings servers ===

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>&1
    echo Stopped backend (port 8000)
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5173 ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>&1
    echo Stopped frontend (port 5173)
)

echo.
echo === Both servers stopped ===
echo.
pause