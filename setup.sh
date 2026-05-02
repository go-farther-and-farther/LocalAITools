#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║     LocalAITools - 本地AI工具箱          ║"
echo "  ║     一键安装 & 启动脚本 (Linux/macOS)     ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""

# ==================== 检测 Python ====================
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "  [✗] 未检测到 Python，请先安装 Python 3.10 或更高版本"
    echo ""
    echo "  Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    echo "  macOS:         brew install python3"
    echo "  官网:          https://www.python.org/downloads/"
    echo ""
    exit 1
fi

PY_VER=$("$PYTHON" --version 2>&1 | awk '{print $2}')
echo "  [✓] 检测到 Python $PY_VER"

# ==================== 检测版本 ====================
PY_MINOR=$("$PYTHON" -c 'import sys; print(sys.version_info.minor)')
if [ "$PY_MINOR" -lt 10 ]; then
    echo "  [!] Python 版本过低（需要 3.10+），请升级"
    exit 1
fi
echo "  [✓] Python 版本符合要求 (>= 3.10)"
echo ""

# ==================== 创建虚拟环境 ====================
if [ ! -d ".venv" ]; then
    echo "  [→] 正在创建虚拟环境..."
    "$PYTHON" -m venv .venv
    echo "  [✓] 虚拟环境创建完成"
else
    echo "  [✓] 虚拟环境已存在，跳过创建"
fi
echo ""

# ==================== 安装依赖 ====================
echo "  [→] 正在安装依赖包（首次可能需要几分钟）..."
echo ""
.venv/bin/pip install --upgrade pip -q 2>/dev/null
.venv/bin/pip install -r requirements.txt
echo ""
echo "  [✓] 依赖安装完成"
echo ""

# ==================== 配置文件 ====================
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "  [✓] 已从 .env.example 创建配置文件 .env"
    fi
else
    echo "  [✓] 配置文件 .env 已存在"
fi
echo ""

# ==================== 启动 ====================
echo "  ╔══════════════════════════════════════════╗"
echo "  ║  环境就绪，正在启动 Web 界面...          ║"
echo "  ║  浏览器将自动打开 http://localhost:7860  ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""
echo "  提示：首次使用请先看「开始使用」页面的指引！"
echo ""

.venv/bin/python app.py
