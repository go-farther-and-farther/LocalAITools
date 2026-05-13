#!/usr/bin/env bash

# 如果从图形界面双击运行，则打开终端执行自身
if [ ! -t 0 ]; then
    for term in gnome-terminal konsole xterm xfce4-terminal; do
        if command -v "$term" &>/dev/null; then
            case "$term" in
                gnome-terminal) gnome-terminal -- bash "$0" "$@" ;;
                konsole)        konsole -e bash "$0" "$@" ;;
                xterm)          xterm -e bash "$0" "$@" ;;
                xfce4-terminal) xfce4-terminal -e bash "$0" "$@" ;;
            esac
            exit 0
        fi
    done
    echo "错误：未找到终端模拟器，请手动在终端中运行: bash 启动.sh"
    exit 1
fi

cd "$(dirname "$0")"

echo ""
echo "  LocalAITools - 本地AI工具箱"
echo "  启动中...浏览器将自动打开"
echo ""

if [ ! -f ".venv/bin/python" ]; then
    echo "  [!] 未检测到虚拟环境，请先运行 bash 安装.sh 完成安装"
    echo ""
    exit 1
fi

.venv/bin/python app.py
