@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

title LocalAITools - 一键安装启动

echo.
echo   ╔══════════════════════════════════════════╗
echo   ║     LocalAITools - 本地AI工具箱          ║
echo   ║     一键安装 & 启动脚本                   ║
echo   ╚══════════════════════════════════════════╝
echo.

:: ==================== 检测 Python ====================
set PYTHON_CMD=
for %%P in (python python3 py) do (
    where %%P >nul 2>nul
    if !errorlevel!==0 (
        for /f "tokens=2 delims= " %%V in ('%%P --version 2^>^&1') do set PYTHON_CMD=%%P
        if defined PYTHON_CMD goto :python_found
    )
)

echo   [×] 未检测到 Python，请先安装 Python 3.10 或更高版本
echo.
echo   下载地址：
echo   https://www.python.org/downloads/
echo.
echo   ⚠️  安装时请务必勾选 "Add Python to PATH"（添加到系统路径）
echo.
start https://www.python.org/downloads/
pause
exit /b 1

:python_found
for /f "tokens=2 delims= " %%V in ('%PYTHON_CMD% --version 2^>^&1') do set PY_VER=%%V
echo   [√] 检测到 Python %PY_VER%
echo.

:: ==================== 检测 Python 版本 ====================
for /f "tokens=2 delims=." %%A in ("%PY_VER%") do set PY_MINOR=%%A
if %PY_MINOR% LSS 10 (
    echo   [!] Python 版本过低（需要 3.10+），请升级 Python
    echo   https://www.python.org/downloads/
    pause
    exit /b 1
)

echo   [√] Python 版本符合要求（^>= 3.10）
echo.

:: ==================== 创建虚拟环境 ====================
if not exist ".venv" (
    echo   [→] 正在创建虚拟环境...
    %PYTHON_CMD% -m venv .venv
    if !errorlevel! neq 0 (
        echo   [×] 创建虚拟环境失败，请检查 Python 安装
        pause
        exit /b 1
    )
    echo   [√] 虚拟环境创建完成
) else (
    echo   [√] 虚拟环境已存在，跳过创建
)
echo.

:: ==================== 安装依赖 ====================
echo   [→] 正在安装依赖包（首次可能需要几分钟）...
echo.
.venv\Scripts\python -m pip install --upgrade pip >nul 2>nul
.venv\Scripts\pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo.
    echo   [×] 依赖安装失败，请检查网络连接后重试
    pause
    exit /b 1
)
echo.
echo   [√] 依赖安装完成
echo.

:: ==================== 配置文件 ====================
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo   [√] 已从 .env.example 创建配置文件 .env
    )
) else (
    echo   [√] 配置文件 .env 已存在
)
echo.

:: ==================== 启动 ====================
echo   ╔══════════════════════════════════════════╗
echo   ║  环境就绪，正在启动 Web 界面...          ║
echo   ║  浏览器将自动打开 http://localhost:7860  ║
echo   ╚══════════════════════════════════════════╝
echo.
echo   提示：首次使用请先看「开始使用」页面的指引！
echo.

.venv\Scripts\python app.py
pause
