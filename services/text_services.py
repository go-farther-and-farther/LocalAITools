"""Text tool service functions -- pure Python, no Gradio dependency."""
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


def compress_text(
    input_path: str,
    model: str = "",
    temperature: float = 0.3,
    chunk_size: int = 20480,
    internal_workers: int = 2,
    max_tokens: int = 500,
    thinking: bool = True,
    progress_callback: Callable[[int, int], None] = None,
) -> str:
    """Compress chat text files. Returns log text.

    The caller (UI layer) is responsible for calling ``_apply_provider()``
    and setting ``os.environ["ENABLE_THINKING"]`` before invoking this.
    """
    os.environ["ENABLE_THINKING"] = "true" if thinking else "false"
    logger.info(f"[聊天压缩] 路径={input_path} 模型={model} 块大小={chunk_size}")
    from text_tools.compress_chat import process_single_text_file, process_folder

    def on_progress(completed, total):
        if progress_callback:
            progress_callback(completed, total)

    p = Path(input_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        if p.is_file():
            process_single_text_file(
                p, None,
                model or config.VISION_MODEL_THINKING,
                temperature, max_tokens, chunk_size, internal_workers,
                progress_callback=on_progress,
            )
        elif p.is_dir():
            process_folder(
                p, None,
                model or config.VISION_MODEL_THINKING,
                temperature, max_tokens, chunk_size, internal_workers,
                progress_callback=on_progress,
            )
        else:
            return "❌ 路径无效"

    return f"✅ 处理完成\n\n{buf.getvalue()}"


def get_chapter_progress(input_file: str) -> list[list]:
    """Read input file, split into chapters, and return progress table rows.

    Each row is ``[chapter_index, title_snippet, status, char_count]``.
    Returns an empty list if the file is invalid.
    """
    if not input_file or not Path(input_file).is_file():
        return []
    from text_tools.translate import split_into_chapters_fast, load_progress, PROGRESS_FILE

    text = Path(input_file).read_text(encoding="utf-8")
    chapters = split_into_chapters_fast(text)
    progress = load_progress(PROGRESS_FILE)
    done_idx = progress.get("last_chapter_index", 0)
    rows = []
    for i, (title, content) in enumerate(chapters):
        status = "已完成" if i < done_idx else ("翻译中" if i == done_idx else "待翻译")
        rows.append([i + 1, title[:20], status, len(content)])
    return rows


def translate_text(
    input_file: str,
    output_file: str = "",
    model: str = "",
    batch_size: int = 10,
    workers: int = 4,
    thinking: bool = True,
    glossary_text: str = "",
    progress_callback: Callable[[int, int], None] = None,
) -> str:
    """Translate long text file. Returns log text.

    The caller (UI layer) is responsible for calling ``_apply_provider()``
    and setting ``os.environ["ENABLE_THINKING"]`` before invoking this.
    """
    os.environ["ENABLE_THINKING"] = "true" if thinking else "false"
    logger.info(f"[文本翻译] 文件={input_file} 模型={model} 批大小={batch_size} 线程={workers}")
    from text_tools.translate import translate_book_parallel

    if not Path(input_file).is_file():
        return "❌ 输入文件不存在"

    output_file = output_file or str(config.OUTPUT_DIR / "translation" / "translation.txt")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    def on_progress(completed, total):
        if progress_callback:
            progress_callback(completed, total)

    buf = io.StringIO()
    with redirect_stdout(buf):
        translate_book_parallel(
            input_file, output_file,
            model or config.TEXT_MODEL,
            batch_size, workers,
            resume=True,
            progress_callback=on_progress,
            glossary_text=glossary_text,
        )

    return f"✅ 输出: {output_file}\n\n{buf.getvalue()}"
