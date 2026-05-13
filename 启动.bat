@echo off
chcp 65001 >nul 2>nul
cd /d "%~dp0"

echo.
echo   LocalAITools
echo   Starting...
echo   http://localhost:7860
echo.

.venv\Scripts\python app.py
pause
