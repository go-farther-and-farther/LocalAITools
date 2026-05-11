"""
LocalAITools 统一配置文件
=======================
所有工具共享的配置项，优先从 .env 文件读取，未设置时使用默认值。
使用方式：
    from config import *
    或
    import config
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ==================== API 连接 ====================
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")

# ==================== 模型名称 ====================
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.6-27b")
TEXT_MODEL = os.getenv("TEXT_MODEL", "qwen/qwen3.5-9b")
# 个别工具默认使用不同模型，可按需覆盖
VISION_MODEL_THINKING = os.getenv("VISION_MODEL_THINKING", "qwen/qwen3.6-35b-a3b-Thinking")
RENAME_MODEL = os.getenv("RENAME_MODEL", "qwen/Qwen3.6-27b")
BENCHMARK_MODEL = os.getenv("BENCHMARK_MODEL", "qwen3.6-35b-a3b@iq2_xxs")

# ==================== Embedding 模型路径 ====================
# 留空则使用 HuggingFace 默认在线下载；填写本地绝对路径可离线使用
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
KB_CHUNK_SIZE = int(os.getenv("KB_CHUNK_SIZE", "500"))
KB_CHUNK_OVERLAP = int(os.getenv("KB_CHUNK_OVERLAP", "50"))

# ==================== 并发与重试 ====================
DEFAULT_WORKERS = int(os.getenv("DEFAULT_WORKERS", "2"))
RETRY_TIMES = int(os.getenv("RETRY_TIMES", "2"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
REQUEST_TIMEOUT_SHORT = int(os.getenv("REQUEST_TIMEOUT_SHORT", "60"))

# ==================== 图片处理参数 ====================
# 输入图片最大边长（超过则等比缩小），降低 API 传输量和显存占用
IMAGE_MAX_SIZE = int(os.getenv("IMAGE_MAX_SIZE", "2048"))
# ocr_chat_screenshots 切片参数
SLICE_HEIGHT = int(os.getenv("SLICE_HEIGHT", "2000"))
OVERLAP = int(os.getenv("OVERLAP", "400"))

# 质量检测分类目录名
HIGH_QUALITY_FOLDER = os.getenv("HIGH_QUALITY_FOLDER", "HighQuality")
LOW_QUALITY_ERRORS_FOLDER = os.getenv("LOW_QUALITY_ERRORS_FOLDER", "LowQuality_Errors")

# 支持的图片扩展名（逗号分隔字符串 → 集合）
_image_exts = os.getenv("IMAGE_EXTENSIONS", ".jpg,.jpeg,.png,.webp,.bmp,.tiff")
IMAGE_EXTENSIONS = set(ext.strip() for ext in _image_exts.split(",") if ext.strip())

# 重新分类比例
TOP_PERCENT = float(os.getenv("TOP_PERCENT", "0.05"))
BOTTOM_PERCENT = float(os.getenv("BOTTOM_PERCENT", "0.05"))

# ==================== 文本处理参数 ====================
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "20480"))
OVERLAP_MESSAGES = int(os.getenv("OVERLAP_MESSAGES", "2"))

# ==================== 翻译参数 ====================
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "30000"))
SOURCE_LANG = os.getenv("SOURCE_LANG", "Chinese")
TARGET_LANG = os.getenv("TARGET_LANG", "English")

# ==================== 更新设置 ====================
AUTO_UPDATE = os.getenv("AUTO_UPDATE", "false").lower() == "true"

# ==================== 本地模型 ====================
LOCAL_MODEL_ENABLED = os.getenv("LOCAL_MODEL_ENABLED", "false").lower() == "true"
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "")
LOCAL_MODEL_CTX = int(os.getenv("LOCAL_MODEL_CTX", "4096"))

# ==================== 思考模式 ====================
# 是否启用模型的思考/推理模式（Thinking/Reasoning）。
# 关闭后模型直接输出结果，速度更快，适合简单任务。
# 对不支持思考模式的模型，此选项无效（不会报错）。
# 注意：每次调用 get_llm_extra_body() 时动态读取环境变量，
# 确保修改 .env 后重新创建的 LLM 实例能生效。
def get_llm_extra_body(enabled: bool = None) -> dict:
    """返回 ChatOpenAI 的 extra_body 参数，用于控制思考模式。
    enabled=True 强制开启，enabled=False 强制关闭，None 读取环境变量。"""
    if enabled is None:
        import os as _os
        from dotenv import load_dotenv as _load_dotenv
        _load_dotenv(override=True)
        enabled = _os.getenv("ENABLE_THINKING", "true").lower() == "true"
    if not enabled:
        return {"enable_thinking": False}
    return {"enable_thinking": True}

# ==================== 模型分类关键词 ====================
MODEL_KEYWORDS_VLM = ['vision', 'vlm', 'visual', 'gpt-4o', 'qwen-vl', 'qwen2-vl',
                      'internvl', 'minicpm-v', 'cogvlm', 'llava', 'deepseek-vl']
MODEL_KEYWORDS_EMBED = ['embed', 'bge', 'e5-', 'gte-', 'text-embedding', 'cohere']
MODEL_KEYWORDS_CHAT = ['chat', 'gpt', 'qwen', 'deepseek', 'llama', 'mistral', 'gemma',
                       'glm', 'yi-', 'yi ', 'internlm', 'phi', 'baichuan', 'moonshot',
                       'kimi', 'doubao', 'spark', 'ernie', 'claude', 'gemini']

# ==================== 目录路径 ====================
# 项目根目录
ROOT_DIR = Path(__file__).parent.absolute()
# 默认输入文件存放目录
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT_DIR / "data")))
# 默认输出文件存放目录
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(ROOT_DIR / "outputs")))

# 自动创建必要目录
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 状态持久化（记住上次输入的所有参数） ====================
import json as _json

_STATE_FILE = ROOT_DIR / "data" / "state.json"

def load_state(tool_key: str = None) -> dict:
    """加载持久化状态。指定 tool_key 返回该工具参数字典，不指定返回全部。"""
    if _STATE_FILE.exists():
        try:
            data = _json.loads(_STATE_FILE.read_text(encoding="utf-8"))
            if tool_key:
                return data.get(tool_key, {})
            return data
        except Exception:
            pass
    return {} if tool_key else {}

def save_state(tool_key: str, params: dict = None, **kwargs):
    """保存某个工具的所有参数。支持 save_state('rename', input_dir='...', model='...', workers=4)"""
    state = {}
    if _STATE_FILE.exists():
        try:
            state = _json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    if params:
        state[tool_key] = params
    else:
        state[tool_key] = {k: v for k, v in kwargs.items() if v is not None and v != ""}
    try:
        _STATE_FILE.write_text(_json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def clear_state():
    """清除所有保存的状态，恢复默认设置"""
    try:
        if _STATE_FILE.exists():
            _STATE_FILE.unlink()
    except Exception:
        pass


# ==================== 供应商管理 ====================
def load_providers():
    """加载供应商列表和当前活动供应商名。首次使用时从 .env 创建默认供应商。"""
    state = {}
    if _STATE_FILE.exists():
        try:
            state = _json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    providers = state.get("providers")
    if not providers or not providers.get("list"):
        # 首次：从当前 .env 配置创建默认供应商
        default = {
            "list": [
                {"name": "默认", "base_url": OPENAI_BASE_URL, "api_key": OPENAI_API_KEY}
            ],
            "active": "默认"
        }
        state["providers"] = default
        try:
            _STATE_FILE.write_text(_json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return default["list"], default["active"]

    return providers["list"], providers.get("active", providers["list"][0]["name"])


def save_providers(provider_list, active_name):
    """保存供应商列表和当前活动供应商名"""
    state = {}
    if _STATE_FILE.exists():
        try:
            state = _json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    state["providers"] = {"list": provider_list, "active": active_name}
    try:
        _STATE_FILE.write_text(_json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def get_active_provider():
    """获取当前活动供应商的 {name, base_url, api_key}"""
    provider_list, active_name = load_providers()
    for p in provider_list:
        if p["name"] == active_name:
            return p
    # fallback: 返回第一个
    if provider_list:
        return provider_list[0]
    return {"name": "默认", "base_url": OPENAI_BASE_URL, "api_key": OPENAI_API_KEY}
