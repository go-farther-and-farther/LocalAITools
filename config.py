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
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")

# ==================== 并发与重试 ====================
DEFAULT_WORKERS = int(os.getenv("DEFAULT_WORKERS", "2"))
RETRY_TIMES = int(os.getenv("RETRY_TIMES", "2"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
REQUEST_TIMEOUT_SHORT = int(os.getenv("REQUEST_TIMEOUT_SHORT", "60"))

# ==================== 图片处理参数 ====================
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
