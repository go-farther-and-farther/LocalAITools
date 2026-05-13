@echo off
chcp 65001 >nul 2>nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ================================================
echo   内置本地模型安装脚本
echo ================================================
echo.

:: 检测 Python
set PYTHON_CMD=
for %%P in (python python3 py) do (
    where %%P >nul 2>nul
    if !errorlevel!==0 set PYTHON_CMD=%%P
)
if not defined PYTHON_CMD (
    echo   [错误] 未找到 Python，请先运行「安装.bat」
    pause
    exit /b 1
)

:: 优先使用 venv
if exist ".venv\Scripts\python.exe" set PYTHON_CMD=.venv\Scripts\python.exe

echo   [1/3] 安装 llama-cpp-python...
echo.
%PYTHON_CMD% -m pip install llama-cpp-python --quiet
if !errorlevel! neq 0 (
    echo   [警告] 常规安装失败，尝试预编译版本...
    %PYTHON_CMD% -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --quiet
)
echo.
echo   [完成] llama-cpp-python 安装成功
echo.

echo   [2/3] 下载模型文件...
echo.
echo   推荐模型：
echo     [1] Qwen3.5-0.6B  (~400MB，速度快，适合分类/重命名)
echo     [2] Qwen3.5-1.7B  (~1GB，更智能，适合翻译/压缩)
echo     [3] 跳过下载（稍后手动放入 models\ 目录）
echo.
set /p choice=请选择 [1/2/3]:

if "%choice%"=="1" (
    echo.
    echo   正在下载 Qwen3.5-0.6B...
    %PYTHON_CMD% -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen3.5-0.6B-GGUF', 'qwen3.5-0.6b-q4_k_m.gguf', local_dir='models')"
    echo   下载完成！
) else if "%choice%"=="2" (
    echo.
    echo   正在下载 Qwen3.5-1.7B...
    %PYTHON_CMD% -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen3.5-1.7B-GGUF', 'qwen3.5-1.7b-q4_k_m.gguf', local_dir='models')"
    echo   下载完成！
) else (
    echo   跳过下载。请手动将 .gguf 文件放入 models\ 目录。
)

echo.
echo   [3/3] 配置完成！
echo.
echo   使用方法：
echo     1. 双击「启动.bat」启动应用
echo     2. 进入「设置」-「内置本地模型」
echo     3. 点击「启动本地模型」
echo     4. 顶部供应商栏选择「本地模型」即可使用
echo.
pause
