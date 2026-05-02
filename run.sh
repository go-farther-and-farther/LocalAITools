#!/bin/bash

# 如果当前不是运行在终端中（例如从图形界面双击运行），则打开一个新终端执行自身
if [ ! -t 0 ]; then
    # 根据系统可用的终端模拟器选择合适的命令
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal -- bash "$0" "$@"
    elif command -v konsole &> /dev/null; then
        konsole -e bash "$0" "$@"
    elif command -v xterm &> /dev/null; then
        xterm -e bash "$0" "$@"
    else
        echo "错误：未找到可用的终端模拟器（gnome-terminal/konsole/xterm）。"
        echo "请手动在终端中运行此脚本。"
        exit 1
    fi
    exit 0
fi

# 切换到脚本所在的目录（即 LocalAITools-main 目录）
cd "$(dirname "$0")"

echo "🚀 启动 LocalAITools ..."

# 使用虚拟环境中的 Python 直接运行 app.py（后台运行）
./venv/bin/python app.py &
APP_PID=$!

# 等待服务启动（Gradio 通常需要几秒）
echo "⏳ 等待服务启动（约 3 秒）..."
sleep 3

# 自动打开浏览器
if command -v xdg-open &> /dev/null; then
    xdg-open http://127.0.0.1:7860
elif command -v open &> /dev/null; then
    open http://127.0.0.1:7860
else
    echo "⚠️ 无法自动打开浏览器，请手动访问：http://127.0.0.1:7860"
fi

echo "✅ 应用已运行，按 Ctrl+C 可停止服务。"
# 等待应用进程结束（前台阻塞，直到用户按 Ctrl+C）
wait $APP_PID
