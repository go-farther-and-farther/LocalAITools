#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聊天记录长截图 → 切片 → 智能并发识别 → 完整文字输出
支持单图内部并发、多图并发，内置重试与图片模式转换。
"""

import sys
import base64
import re
import time
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

import threading

from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from tqdm import tqdm

_stop_flag = threading.Event()

def request_stop():
    """请求停止当前正在执行的 OCR 任务"""
    _stop_flag.set()

EXTRACT_PROMPT = """你是一个聊天记录整理助手。我将按从上到下的顺序给你多张聊天记录长截图切片，它们是同一段对话的连续画面。

请仔细阅读所有图片中的内容，并按以下格式逐条输出完整的聊天消息和图片描述：

【发言归属规则】
- 每条消息必须明确标注说话人。
- 判断方法：观察消息气泡在屏幕中的左右位置。
  - 位于屏幕左侧的气泡（通常绿色或白色）是“对方”发送的。
  - 位于屏幕右侧的气泡（通常绿色或带发送状态）是“我”发送的。
  - 如果无法从位置判断，请根据上下文或头像推测，并在输出时标记为“（可能对方）”或“（可能我）”。

【时间戳规则】
- **仅当截图中明确显示该条消息的时间戳，才在输出中包含时间戳。** 若无显示则不要添加任何时间信息。

【图片描述规则】
- 对话中出现的图片、表情包、贴纸、语音消息、文件等非文字内容，需用文字描述其内容，放在对应发言人的消息中。
- 描述格式示例：
  - [图片描述：一张小猫卖萌的表情包]
  - [语音消息：时长 15 秒，未播放]

【输出格式】
严格逐行输出，每条消息占一行：
[时间戳] 对方：消息内容
或
[时间戳] 我：消息内容
若无时间戳，则只输出说话人和内容：
对方：消息内容
我：消息内容

【注意事项】
- 忽略顶部和底部被切断的不完整消息气泡。
- 不要添加任何总结、说明或额外解释，只输出识别结果。

请输出识别结果："""


def encode_image_to_base64(pil_image: Image.Image, max_size: int = None) -> str:
    if max_size is None:
        max_size = config.IMAGE_MAX_SIZE
    """将 PIL 图片转为 base64 JPEG，处理 RGBA 模式，自动压缩大图"""
    import io
    w, h = pil_image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        pil_image = pil_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buffered = io.BytesIO()
    if pil_image.mode in ("RGBA", "P"):
        pil_image = pil_image.convert("RGB")
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def slice_image_in_memory(image_path: Path) -> List[Image.Image]:
    """将长截图按高度切片，返回 PIL Image 列表"""
    img = Image.open(image_path)
    width, height = img.size
    step = config.SLICE_HEIGHT - config.OVERLAP

    slices = []
    y = 0
    while y + config.SLICE_HEIGHT < height:
        box = (0, y, width, y + config.SLICE_HEIGHT)
        slices.append(img.crop(box))
        y += step
    last_start = max(0, height - config.SLICE_HEIGHT)
    box = (0, last_start, width, height)
    slices.append(img.crop(box))
    return slices


def extract_text_from_slices(slices: List[Image.Image], model: str,
                             temperature: float = 0.3,
                             max_tokens: int = 5000,
                             retries: int = 1) -> str:
    """识别一组切片，带重试机制"""
    for attempt in range(retries + 1):
        try:
            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=config.OPENAI_BASE_URL,
                api_key=config.OPENAI_API_KEY,
                request_timeout=config.REQUEST_TIMEOUT,
                extra_body=config.get_llm_extra_body()
            )

            content = [{"type": "text", "text": EXTRACT_PROMPT}]
            for sl in slices:
                img_b64 = encode_image_to_base64(sl)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })

            resp = llm.invoke([HumanMessage(content=content)])
            return resp.content.strip()

        except Exception as e:
            print(f"   ⚠️ 模型调用失败 (尝试 {attempt+1}/{retries+1}): {e}")
            if attempt < retries:
                time.sleep(5)
    return ""


def process_single_image_parallel(image_path: Path,
                                  output_txt_path: Optional[Path] = None,
                                  vision_model: str = config.VISION_MODEL_THINKING,
                                  temperature: float = 0.3,
                                  max_tokens: int = 5000,
                                  internal_workers: int = 2) -> str:
    """单张图片处理：切片→并发识别→合并保存"""
    print(f"\n📷 处理: {image_path.name}")
    slices = slice_image_in_memory(image_path)
    total = len(slices)
    if total == 0:
        print("   ❌ 无切片内容")
        return ""

    print(f"   ✂️  图片已切为 {total} 片 (高度 {config.SLICE_HEIGHT}px, 重叠 {config.OVERLAP}px)")

    # 如果切片很少或用户要求单线程，直接一次识别
    if total <= 2 or internal_workers <= 1:
        print(f"   📤 一次性发送全部 {total} 张切片...")
        full_text = extract_text_from_slices(slices, vision_model, temperature, max_tokens)
    else:
        mid = total // 2
        part1 = slices[:mid]
        part2 = slices[mid:]
        print(f"   📤 分为两部分并发: Part1: {len(part1)} 片, Part2: {len(part2)} 片")

        def recognize_part(part, name):
            print(f"      🚀 开始识别 {name}...")
            text = extract_text_from_slices(part, vision_model, temperature, max_tokens)
            print(f"      ✅ {name} 识别完成，字符数: {len(text)}")
            return text

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(recognize_part, part1, "Part1")
            f2 = executor.submit(recognize_part, part2, "Part2")
            txt1 = f1.result()
            txt2 = f2.result()

        full_text = f"{txt1}\n\n[=== 以下为后半部分切片 ===]\n\n{txt2}" if txt1 and txt2 else (txt1 or txt2)

    # 清理与保存
    if not full_text or full_text.strip() == "":
        print("   ❌ 识别结果为空，跳过保存")
        return ""

    full_text = re.sub(r'\n\s*\n', r'\n', full_text)
    print(f"   ✅ 识别完成，总字符数: {len(full_text)}")

    if output_txt_path is None:
        output_txt_path = image_path.with_suffix(".txt")
    else:
        output_txt_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"图片: {image_path.name}\n")
        f.write("=" * 50 + "\n")
        f.write(full_text)
    print(f"   💾 已保存: {output_txt_path}")
    return full_text


def process_folder(input_dir: Path,
                   output_dir: Optional[Path] = None,
                   vision_model: str = config.VISION_MODEL_THINKING,
                   temperature: float = 0.3,
                   max_workers: int = 1,        # 建议初期设为1
                   max_tokens: int = 5000,
                   internal_workers: int = 2,   # 建议初期设为1，稳定后提至2
                   progress_callback=None):
    """批量处理文件夹内所有图片"""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    images = [f for f in input_dir.iterdir() if f.suffix.lower() in exts]
    if not images:
        print(f"❌ 文件夹 {input_dir} 中没有找到图片。")
        return

    total = len(images)
    print(f"📂 共发现 {total} 张图片。")
    if output_dir is None:
        output_dir = input_dir / "chat_text_output"
    output_dir.mkdir(exist_ok=True)

    tasks = [(img, output_dir / f"{img.stem}.txt") for img in images]
    completed = 0

    _stop_flag.clear()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_img = {
            executor.submit(
                process_single_image_parallel,
                img, out, vision_model, temperature, max_tokens, internal_workers
            ): img
            for img, out in tasks
        }
        for future in tqdm(as_completed(future_to_img), total=total, desc="总体进度"):
            if _stop_flag.is_set():
                executor.shutdown(wait=False, cancel_futures=True)
                print("\n⏹️ 已请求停止 OCR")
                break
            img = future_to_img[future]
            try:
                future.result()
            except Exception as e:
                print(f"❌ 处理 {img.name} 时发生异常: {e}")
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    if not _stop_flag.is_set():
        print("\n🎉 所有图片处理完成！")


# ================= 主程序 =================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="聊天记录长截图 → 切片识别 → 文本输出")
    parser.add_argument("--input", "-i", default=str(config.DATA_DIR / "screenshots"), help="输入图片路径或文件夹")
    parser.add_argument("--output", "-o", default=None, help="文本输出目录（默认在输入目录下创建 chat_text_output）")
    parser.add_argument("--vision-model", default=config.VISION_MODEL_THINKING, help="多模态模型名称")
    parser.add_argument("--temperature", "-t", type=float, default=0.3, help="识别温度（默认0.3）")
    parser.add_argument("--workers", "-w", type=int, default=1, help="同时处理图片数（默认1）")
    parser.add_argument("--internal-workers", type=int, default=2, help="单张图片内部切片并发数（默认1）")
    parser.add_argument("--max-tokens", type=int, default=5000, help="单次请求最大token数")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_file():
        process_single_image_parallel(
            input_path,
            args.output,
            args.vision_model,
            args.temperature,
            args.max_tokens,
            args.internal_workers
        )
    elif input_path.is_dir():
        process_folder(
            input_path,
            args.output,
            args.vision_model,
            args.temperature,
            args.workers,
            args.max_tokens,
            args.internal_workers
        )
    else:
        print("❌ 输入路径无效")