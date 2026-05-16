"""Manage external llama-server.exe process.

Launches the llama.cpp server as a subprocess and monitors its status.
Provides an OpenAI-compatible API endpoint that the rest of the app uses.
"""
import logging
import subprocess
import socket
import time
import threading
import requests
from pathlib import Path

logger = logging.getLogger("LocalAITools")

# Module-level state (single server instance at a time)
_process: subprocess.Popen | None = None
_running = False
_port = 1234
_model_name = ""
_server_path = ""


def find_server_executable(root_dir: str) -> str | None:
    """Search for llama-server.exe in the given root directory (up to 3 levels deep)."""
    root = Path(root_dir)
    if not root.is_dir():
        return None
    # Try direct match first
    for name in ("llama-server.exe", "llama-server"):
        direct = root / name
        if direct.exists():
            return str(direct)
    # Search up to 3 levels
    for p in root.rglob("llama-server*"):
        if p.is_file() and p.suffix.lower() in ("", ".exe"):
            return str(p)
        # Limit depth
        try:
            depth = len(p.relative_to(root).parts)
            if depth > 3:
                continue
        except ValueError:
            continue
    return None


def scan_gguf_models(root_dir: str) -> list[str]:
    """Scan for .gguf model files in common locations relative to root_dir.

    Searches: root_dir, root_dir/models, and user's .lmstudio/models.
    Returns sorted list of absolute paths.
    """
    candidates = set()
    search_dirs = []

    # 1. The root dir itself
    root = Path(root_dir)
    if root.is_dir():
        search_dirs.append(root)

    # 2. root/models/
    models_sub = root / "models"
    if models_sub.is_dir():
        search_dirs.append(models_sub)

    # 3. User's .lmstudio/models/
    lmstudio = Path.home() / ".lmstudio" / "models"
    if lmstudio.is_dir():
        search_dirs.append(lmstudio)

    for d in search_dirs:
        for gguf in d.rglob("*.gguf"):
            # Skip mmproj files
            if "mmproj" in gguf.name.lower():
                continue
            candidates.add(str(gguf))

    return sorted(candidates)


def get_status() -> dict:
    """Return current llama-server status."""
    global _process, _running
    # Check if process is still alive
    if _process is not None and _process.poll() is not None:
        _running = False
        _process = None
    return {
        "running": _running,
        "pid": _process.pid if _process else None,
        "port": _port,
        "model_name": _model_name,
        "server_path": _server_path,
        "base_url": f"http://127.0.0.1:{_port}/v1" if _running else None,
    }


def is_server_ready(port: int, timeout: float = 5.0) -> bool:
    """Quick check if the server is responding on the given port."""
    try:
        resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def stop_server() -> str:
    """Stop the running llama-server process."""
    global _process, _running
    if _process is None:
        return "没有运行中的 llama-server 进程"
    try:
        _process.terminate()
        try:
            _process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _process.kill()
            _process.wait(timeout=5)
        pid = _process.pid
        _process = None
        _running = False
        logger.info(f"[llama-server] 已停止进程 PID={pid}")
        return f"llama-server 已停止 (PID={pid})"
    except Exception as e:
        _process = None
        _running = False
        return f"停止进程时出错: {e}"


def launch_server(
    server_path: str,
    model_path: str,
    mmproj_path: str = "",
    host: str = "127.0.0.1",
    port: int = 1234,
    ctx_size: int = 64000,
    expert_kv: int = 0,
    ngl: int = 999,
    parallel: int = 8,
    reasoning: bool = False,
) -> str:
    """Launch llama-server.exe with the given configuration.

    Returns a status message string.
    """
    global _process, _running, _port, _model_name, _server_path

    if _running and _process and _process.poll() is None:
        return f"llama-server 已在运行 (端口 {_port}, PID {_process.pid})"

    # Auto-find server executable if given a directory
    if Path(server_path).is_dir():
        found = find_server_executable(server_path)
        if not found:
            return f"在目录中未找到 llama-server.exe:\n{server_path}"
        server_path = found
        logger.info(f"[llama-server] 自动找到: {server_path}")

    # Validate paths
    if not Path(server_path).exists():
        return f"llama-server.exe 不存在: {server_path}"
    if not Path(model_path).exists():
        return f"模型文件不存在: {model_path}"

    # Auto-detect mmproj if not specified: look in the same directory as the model
    if not mmproj_path:
        model_dir = Path(model_path).parent
        for f in model_dir.iterdir():
            if f.is_file() and f.name.lower().startswith("mmproj") and f.suffix.lower() in (".gguf", ".bin"):
                mmproj_path = str(f)
                logger.info(f"[llama-server] 自动找到 mmproj: {mmproj_path}")
                break
    elif not Path(mmproj_path).exists():
        return f"mmproj 文件不存在: {mmproj_path}"

    # Build command
    cmd = [
        server_path,
        "-m", model_path,
        "--host", host,
        "--port", str(port),
        "-ngl", str(ngl),
        "--ctx-size", str(ctx_size),
        "--parallel", str(parallel),
        "--no-mmap",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--metrics",
    ]

    if mmproj_path:
        cmd.extend(["--mmproj", mmproj_path])

    if reasoning:
        cmd.extend(["--reasoning", "on"])
    else:
        cmd.extend(["--reasoning", "off"])

    if expert_kv and expert_kv > 0:
        cmd.extend(["--override-kv", f"qwen35moe.expert_used_count=int:{expert_kv}"])

    # Launch process
    logger.info(f"[llama-server] 启动: {' '.join(cmd)}")
    try:
        _process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
        )
    except Exception as e:
        return f"启动失败: {e}"

    _port = port
    _model_name = Path(model_path).stem
    _server_path = server_path

    # Wait for server to be ready (in a background thread to avoid blocking UI)
    model_display = _model_name

    def _wait_ready():
        global _running
        logger.info(f"[llama-server] 等待服务就绪 (端口 {port})...")
        for i in range(120):  # up to 120 seconds
            time.sleep(1)
            if _process.poll() is not None:
                stderr = ""
                try:
                    stderr = _process.stderr.read().decode(errors="replace")[-500:]
                except Exception:
                    pass
                logger.error(f"[llama-server] 进程退出 code={_process.returncode}: {stderr}")
                _running = False
                return
            if is_server_ready(port, timeout=2):
                _running = True
                logger.info(f"[llama-server] 服务就绪 → http://127.0.0.1:{port}/v1")
                return
        logger.warning("[llama-server] 等待超时，服务可能仍在加载中")
        _running = True  # Assume it's loading, will check on next status poll

    threading.Thread(target=_wait_ready, daemon=True).start()

    return (
        f"正在启动 llama-server...\n"
        f"模型: {model_display}\n"
        f"端口: {port}\n"
        f"Context: {ctx_size}\n"
        f"服务就绪后会自动可用 (通常 10-60 秒)"
    )
