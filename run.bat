@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo.
echo   LocalAITools - 本地AI工具箱
echo   启动中...浏览器将自动打开
echo.

if not exist ".venv\Scripts\python.exe" (
    echo   [!] 未检测到虚拟环境，请先双击运行 setup.bat 完成安装
    echo.
    pause
    exit /b 1
)

.venv\Scripts\python app.py
pause
