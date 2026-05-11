"""OCR service functions -- pure Python, no Gradio dependency."""
import io
import logging
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger("LocalAITools")


def extract_chat_text(
    input_dir: str,
    vision_model: str = "",
    temperature: float = 0.3,
    workers: int = 4,
    internal_workers: int = 2,
    max_tokens: int = 5000,
    max_size: int = None,
    thinking: bool = True,
    progress_callback: Callable[[int, int], None] = None,
) -> str:
    """Extract text from chat screenshots. Returns log text.

    The caller (UI layer) is responsible for calling ``_apply_provider()``
    and setting ``os.environ["ENABLE_THINKING"]`` before invoking this.
    """
    os.environ["ENABLE_THINKING"] = "true" if thinking else "false"
    logger.info(f"[图片解释] 目录={input_dir} 模型={vision_model} 线程={workers} max_size={max_size}")
    from image_tools.ocr_chat_screenshots import process_folder

    input_path = Path(input_dir)
    if not input_path.is_dir():
        return "❌ 请输入有效的文件夹路径"

    output_dir = input_path / "chat_text_output"
    output_dir.mkdir(exist_ok=True)

    def on_progress(completed, total):
        if progress_callback:
            progress_callback(completed, total)

    effective_max_size = max_size if max_size and max_size > 0 else None
    buf = io.StringIO()
    with redirect_stdout(buf):
        process_folder(
            input_path,
            output_dir,
            vision_model or config.VISION_MODEL_THINKING,
            temperature,
            workers,
            max_tokens,
            internal_workers,
            progress_callback=on_progress,
            max_size=effective_max_size,
        )

    return f"✅ 输出目录: {output_dir}\n\n{buf.getvalue()}"
