#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片一键生成中文短句并重命名（整合优化版）
- 合并作品名与通用描述为统一提示词
- 按时间顺序（文件名时间戳 > 文件修改时间）处理图片
- 将最近3次生成结果作为上下文输入，保持命名一致性
- 支持并行处理、试运行、仅处理截图
"""

import io as io_module
import re
import sys
import base64
import argparse
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from tqdm import tqdm

DEFAULT_MODEL = config.RENAME_MODEL

_stop_flag = threading.Event()

def request_stop():
    """请求停止当前正在执行的重命名任务"""
    _stop_flag.set()

# ========== 统一的提示词模板 ==========
PROMPT = """请用3-25个汉字简洁描述这张图片的内容。

要求：
1. **仅当你非常确定**确实是一个广为人知的作品名（游戏/动漫/电影/小说）、角色名或知名IP名称，并且该名称能与图像内容合理对应时，才可以在短句中**开头**使用它，格式为：《作品名》画面描述。
2. **绝对不要编造或猜测作品名**。宁可不用，也不要用错。

以下信息可能对你有帮助：
- 最近处理的图片的描述（可作风格参考，不必强制保持一致）:
{recent_descriptions}

- 原始文件名（不含扩展名）: {original_stem}
  注意：原始文件名可能不准确，仅作为参考。如果它与图像内容明显不符，请忽略。

仅输出短句本身，不要加引号、换行或任何解释："""

# ========== 全局 LLM 实例 ==========
_llm_instance: Optional[ChatOpenAI] = None
_llm_lock = threading.Lock()
_rename_lock = threading.Lock()
_history_lock = threading.Lock()


def get_shared_llm(model: str, temperature: float = 0.5) -> ChatOpenAI:
    global _llm_instance
    if _llm_instance is None or getattr(_llm_instance, 'model', None) != model:
        with _llm_lock:
            if _llm_instance is None or getattr(_llm_instance, 'model', None) != model:
                _llm_instance = ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    max_tokens=2048,
                    base_url=config.OPENAI_BASE_URL,
                    api_key=config.OPENAI_API_KEY,
                    extra_body=config.get_llm_extra_body()
                )
    return _llm_instance


def get_recent_descriptions(recent_queue: deque) -> List[str]:
    """线程安全地获取最近描述列表的副本"""
    with _history_lock:
        return list(recent_queue)


def add_recent_description(recent_queue: deque, description: str):
    """线程安全地添加新描述"""
    with _history_lock:
        recent_queue.append(description)


def encode_image(image_path: str, max_size: int = None) -> Optional[str]:
    if max_size is None:
        max_size = config.IMAGE_MAX_SIZE
    from PIL import Image, ImageOps
    try:
        img = Image.open(image_path)
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io_module.BytesIO()
        fmt = img.format or 'JPEG'
        # JPEG/WebP 不支持透明通道，需转为 RGB
        if img.mode in ('RGBA', 'P', 'LA') and fmt.upper() in ('JPEG', 'JPG', 'WEBP'):
            img = img.convert('RGB')
        img.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"   ⚠️ 图片读取失败: {Path(image_path).name} - {e}")
        return None


def extract_timestamp(filename: str) -> Optional[str]:
    """
    从文件名中提取日期时间，支持两种常见格式：
    1. "屏幕截图 2026-04-11 005848"  （空格分隔，时间6位数字）
    2. "Client-Win64-Shipping-2025-11-12-19-37-57-600.jpg" （短横线分隔，可选毫秒）
    返回统一格式: "YYYY-MM-DD HHMMSS"
    """
    m1 = re.search(r'(\d{4}-\d{2}-\d{2})\s+(\d{6})', filename)
    if m1:
        return f"{m1.group(1)} {m1.group(2)}"

    m2 = re.search(r'(\d{4}-\d{2}-\d{2})-(\d{2})-(\d{2})-(\d{2})(?:-\d+)?', filename)
    if m2:
        date_part = m2.group(1)
        hour, minute, second = m2.group(2), m2.group(3), m2.group(4)
        time_part = f"{hour}{minute}{second}"
        return f"{date_part} {time_part}"

    return None


def get_timestamp_key(filepath: Path) -> str:
    """返回用于排序的时间字符串（优先文件名时间戳，其次文件修改时间）"""
    name = filepath.stem
    ts = extract_timestamp(name)
    if ts:
        return ts
    # 回退到文件修改时间
    mtime = filepath.stat().st_mtime
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H%M%S")


def generate_short_name(image_path: Path, model: str, original_stem: str,
                        recent_descriptions: List[str]) -> Optional[str]:
    """
    调用多模态模型生成中文短句。
    recent_descriptions: 最近生成的短句列表（用于上下文参考）
    """
    llm = get_shared_llm(model, temperature=0.5)
    img_base64 = encode_image(str(image_path))
    if img_base64 is None:
        return None

    # 准备历史描述文本
    if recent_descriptions:
        desc_str = "\n".join(f"- {d}" for d in recent_descriptions)
    else:
        desc_str = "（无）"

    prompt = PROMPT.format(original_stem=original_stem, recent_descriptions=desc_str)

    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
    ])

    try:
        resp = llm.invoke([msg])
        phrase = resp.content.strip()
        # 保留中文、字母、数字、空格、《》、常见的分隔符
        phrase = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9《》 _-]', '', phrase).strip()

        # 如果没有中文，带上下文重试一次
        if not re.search(r'[\u4e00-\u9fa5]', phrase):
            fallback_prompt = prompt + "\n你之前返回的内容缺少中文，请确保输出中文短句。仅输出短句："
            fallback_msg = HumanMessage(content=[
                {"type": "text", "text": fallback_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ])
            resp2 = llm.invoke([fallback_msg])
            phrase = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9《》 _-]', '', resp2.content.strip()).strip()

        # 长度验证（中文汉字数量）
        chinese_len = len(re.findall(r'[\u4e00-\u9fa5]', phrase))
        min_len, max_len = 3, 25
        if min_len <= chinese_len <= max_len:
            return phrase
        else:
            retry_prompt = prompt + f"\n你之前输出的短句长度不符合要求（需{min_len}-{max_len}个汉字），请重新输出一句："
            retry_msg = HumanMessage(content=[
                {"type": "text", "text": retry_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ])
            resp2 = llm.invoke([retry_msg])
            phrase2 = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9《》 _-]', '', resp2.content.strip()).strip()
            chinese_len2 = len(re.findall(r'[\u4e00-\u9fa5]', phrase2))
            if min_len <= chinese_len2 <= max_len:
                return phrase2
            else:
                print(f"   ⚠️ 生成的短句长度不符（{chinese_len2}字），已忽略")
                return None
    except Exception as e:
        print(f"   ❌ API 调用失败: {e}")
        return None
    finally:
        del img_base64


def safe_rename(old_path: Path, new_stem: str) -> bool:
    with _rename_lock:
        ext = old_path.suffix.lower()
        new_name = f"{new_stem}{ext}"
        new_path = old_path.parent / new_name

        if not new_path.exists():
            old_path.rename(new_path)
            return True

        counter = 1
        while True:
            candidate = old_path.parent / f"{new_stem}_{counter}{ext}"
            if not candidate.exists():
                old_path.rename(candidate)
                return True
            counter += 1


def process_one_image(img_path: Path, model: str, dry_run: bool,
                      recent_history: deque) -> tuple:
    print(f"📷 处理: {img_path.name}")
    try:
        original_stem = img_path.stem
        # 获取最近的历史描述（线程安全）
        recent_list = get_recent_descriptions(recent_history)

        short_name = generate_short_name(img_path, model, original_stem, recent_list)
        if short_name is None:
            print(f"   ⚠️ 短句生成失败，保留原文件名")
            return img_path.name, "生成失败(保留原名)"

        print(f"   ✂️  短句: {short_name}")

        # 记录到历史（即使重命名失败也记录，用于后续上下文）
        add_recent_description(recent_history, short_name)

        # 提取并保留时间戳
        timestamp = extract_timestamp(original_stem)
        if timestamp:
            new_stem = f"{short_name} {timestamp}"
        else:
            new_stem = short_name

        # 清理文件名中不允许的字符
        safe_stem = re.sub(r'[\\/*?:"<>|]', '', new_stem).strip()
        if not safe_stem:
            safe_stem = "未命名图片"

        if dry_run:
            print(f"   🧪 [试运行] 将重命名为: {safe_stem}{img_path.suffix}")
        else:
            if safe_rename(img_path, safe_stem):
                print(f"   ✅ 重命名成功: {safe_stem}{img_path.suffix}")
            else:
                print(f"   ⚠️ 重命名失败")
        return img_path.name, short_name
    except Exception as e:
        print(f"   ❌ 处理失败: {e}")
        return img_path.name, f"错误: {e}"


def main():
    parser = argparse.ArgumentParser(description="图片一键生成中文短句并重命名（整合优化版）")
    parser.add_argument("--input_dir", "-i", default=str(config.DATA_DIR / "images"), help="图片文件夹路径")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="多模态模型名称")
    parser.add_argument("--workers", "-w", type=int, default=4, help="并行线程数")
    parser.add_argument("--dry_run", action="store_true", help="仅模拟，不实际重命名")
    parser.add_argument("--screenshots_only", action="store_true",
                        help="仅处理文件名包含“屏幕截图”的图片（例如微信/系统截图）")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"❌ 错误：{input_dir} 不是有效目录")
        return

    # 支持的图片格式
    exts = [".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif"]
    images = []
    for ext in exts:
        images.extend(input_dir.glob(f"*{ext}"))
        images.extend(input_dir.glob(f"*{ext.upper()}"))

    if args.screenshots_only:
        images = [img for img in images if "屏幕截图" in img.stem]

    # 去重并按照时间顺序排序
    images = sorted(set(images), key=get_timestamp_key)

    if not images:
        print("📂 未找到符合条件的图片文件。")
        return

    print(f"📂 找到 {len(images)} 张图片，使用 {args.workers} 个线程并行处理\n")

    # 初始化 LLM（确保模型可用）
    get_shared_llm(args.model)

    # 最近3次生成历史（线程安全队列）
    recent_history = deque(maxlen=5)

    _stop_flag.clear()
    with tqdm(total=len(images), desc="处理进度", unit="张") as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one_image, img, args.model, args.dry_run, recent_history): img
                       for img in images}
            for future in as_completed(futures):
                if _stop_flag.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    print("\n⏹️ 已请求停止重命名")
                    break
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ 线程内部异常: {e}")
                pbar.update(1)

    if not _stop_flag.is_set():
        print("\n🎉 全部处理完成！")
        if args.dry_run:
            print("（试运行模式）")


if __name__ == "__main__":
    main()