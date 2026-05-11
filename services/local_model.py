"""Built-in local model server using llama-cpp-python.

Provides an OpenAI-compatible API on localhost so the rest of the app
works unchanged — just point the provider at http://localhost:{port}/v1.
"""
import logging
import socket
import threading
import time
from pathlib import Path

logger = logging.getLogger("LocalAITools")

_server_thread: threading.Thread | None = None
_server_running = False
_server_port = 8081


def _find_free_port(start: int = 8081) -> int:
    for port in range(start, start + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    return start


def is_available() -> bool:
    """Check if llama-cpp-python is installed."""
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False


def get_model_path() -> str | None:
    """Find the first .gguf file in models/ directory."""
    models_dir = Path(__file__).parent.parent / "models"
    if not models_dir.is_dir():
        return None
    gguf_files = sorted(models_dir.glob("*.gguf"))
    return str(gguf_files[0]) if gguf_files else None


def get_status() -> dict:
    """Return local model status for UI display."""
    available = is_available()
    model_path = get_model_path()
    return {
        "available": available,
        "model_path": model_path,
        "model_name": Path(model_path).stem if model_path else None,
        "running": _server_running,
        "port": _server_port,
        "base_url": f"http://127.0.0.1:{_server_port}/v1" if _server_running else None,
    }


def start_server(model_path: str = None, n_ctx: int = 4096, n_threads: int = None) -> str:
    """Start the local model server in a background thread.

    Returns a status message.
    """
    global _server_thread, _server_running, _server_port

    if _server_running:
        return f"✅ 本地模型已在运行 (端口 {_server_port})"

    if not is_available():
        return "❌ 未安装 llama-cpp-python。请运行: pip install llama-cpp-python"

    model = model_path or get_model_path()
    if not model or not Path(model).exists():
        return "❌ 未找到模型文件。请将 .gguf 文件放入 models/ 目录"

    import os
    if n_threads is None:
        n_threads = max(1, os.cpu_count() - 1)

    _server_port = _find_free_port()

    def _run():
        global _server_running
        try:
            from llama_cpp import Llama
            logger.info(f"[本地模型] 加载中: {Path(model).name} (ctx={n_ctx}, threads={n_threads})")
            llm = Llama(
                model_path=model,
                n_ctx=n_ctx,
                n_threads=n_threads,
                verbose=False,
            )
            _server_running = True
            logger.info(f"[本地模型] 启动 OpenAI 兼容服务 → http://127.0.0.1:{_server_port}/v1")

            # Run the built-in OpenAI-compatible server
            from llama_cpp.server.app import create_app
            from llama_cpp.server.settings import ServerSettings, ModelSettings

            settings = ServerSettings(host="127.0.0.1", port=_server_port)
            model_settings = ModelSettings(
                model=model,
                n_ctx=n_ctx,
                n_threads=n_threads,
            )
            app = create_app(settings=settings, model_settings=[model_settings])

            import uvicorn
            uvicorn.run(app, host="127.0.0.1", port=_server_port, log_level="warning")

        except Exception as e:
            logger.error(f"[本地模型] 启动失败: {e}")
            _server_running = False

    _server_thread = threading.Thread(target=_run, daemon=True)
    _server_thread.start()

    # Wait for server to start
    for _ in range(30):
        time.sleep(1)
        if _server_running:
            break

    if _server_running:
        return f"✅ 本地模型已启动\n模型: {Path(model).name}\n地址: http://127.0.0.1:{_server_port}/v1"
    return "⏳ 本地模型加载中... (首次加载可能需要 10-30 秒)"


def start_server_simple(model_path: str = None, n_ctx: int = 4096, n_threads: int = None) -> str:
    """Simpler approach: use llama-cpp-python's Llama directly with a minimal HTTP wrapper.

    Falls back to this if the full server dependencies (uvicorn) are missing.
    """
    global _server_thread, _server_running, _server_port

    if _server_running:
        return f"✅ 本地模型已在运行 (端口 {_server_port})"

    if not is_available():
        return "❌ 未安装 llama-cpp-python。请运行: pip install llama-cpp-python"

    model = model_path or get_model_path()
    if not model or not Path(model).exists():
        return "❌ 未找到模型文件。请将 .gguf 文件放入 models/ 目录"

    import os, json
    if n_threads is None:
        n_threads = max(1, os.cpu_count() - 1)

    _server_port = _find_free_port()

    def _run():
        global _server_running
        try:
            from llama_cpp import Llama
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import json as _json

            logger.info(f"[本地模型] 加载中: {Path(model).name} (ctx={n_ctx}, threads={n_threads})")
            llm = Llama(
                model_path=model,
                n_ctx=n_ctx,
                n_threads=n_threads,
                verbose=False,
            )
            _server_running = True
            logger.info(f"[本地模型] 已就绪 → http://127.0.0.1:{_server_port}/v1")

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def do_GET(self):
                    if self.path == "/v1/models":
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        resp = {"object": "list", "data": [{"id": "local-model", "object": "model"}]}
                        self.wfile.write(_json.dumps(resp).encode())
                    else:
                        self.send_response(404)
                        self.end_headers()

                def do_POST(self):
                    if self.path == "/v1/chat/completions":
                        length = int(self.headers.get("Content-Length", 0))
                        body = _json.loads(self.rfile.read(length))
                        messages = body.get("messages", [])
                        max_tokens = body.get("max_tokens", 512)
                        temperature = body.get("temperature", 0.7)
                        stream = body.get("stream", False)

                        if stream:
                            self.send_response(200)
                            self.send_header("Content-Type", "text/event-stream")
                            self.end_headers()
                            for chunk in llm.create_chat_completion(
                                messages=messages,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                stream=True,
                            ):
                                data = _json.dumps(chunk)
                                self.wfile.write(f"data: {data}\n\n".encode())
                                self.wfile.flush()
                            self.wfile.write(b"data: [DONE]\n\n")
                            self.wfile.flush()
                        else:
                            result = llm.create_chat_completion(
                                messages=messages,
                                max_tokens=max_tokens,
                                temperature=temperature,
                            )
                            self.send_response(200)
                            self.send_header("Content-Type", "application/json")
                            self.end_headers()
                            self.wfile.write(_json.dumps(result).encode())
                    else:
                        self.send_response(404)
                        self.end_headers()

            server = HTTPServer(("127.0.0.1", _server_port), Handler)
            server.serve_forever()

        except Exception as e:
            logger.error(f"[本地模型] 运行失败: {e}")
            _server_running = False

    _server_thread = threading.Thread(target=_run, daemon=True)
    _server_thread.start()

    # Wait for server to start
    for _ in range(60):
        time.sleep(1)
        if _server_running:
            break

    if _server_running:
        return f"✅ 本地模型已启动\n模型: {Path(model).name}\n地址: http://127.0.0.1:{_server_port}/v1"
    return "⏳ 本地模型加载中... (首次加载可能需要 10-60 秒)"


def auto_start():
    """Auto-start local model if configured. Called on app startup."""
    from config import LOCAL_MODEL_ENABLED, LOCAL_MODEL_PATH
    if not LOCAL_MODEL_ENABLED:
        return None
    if _server_running:
        return None
    result = start_server_simple(LOCAL_MODEL_PATH)
    logger.info(f"[本地模型] {result}")
    return result
