@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   LocalAITools - 本地 AI 工具箱
echo   正在启动...
echo ============================================
echo.
python app.py
pause
