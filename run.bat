@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   LocalAITools
echo   Starting...
echo ============================================
echo.
.venv\Scripts\python app.py
pause
