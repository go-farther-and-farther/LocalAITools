#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import sys
import time
import threading
import argparse
from datetime import datetime
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

from langchain_openai import ChatOpenAI

_stop_flag = threading.Event()

def request_stop():
    """请求停止当前正在执行的翻译任务"""
    _stop_flag.set()

# -------------------- 环境配置 --------------------
DEFAULT_MODEL = config.TEXT_MODEL
MAX_CONTEXT_TOKENS = config.MAX_CONTEXT_TOKENS
SOURCE_LANG = config.SOURCE_LANG
TARGET_LANG = config.TARGET_LANG
PROGRESS_FILE = str(config.OUTPUT_DIR / "translation" / "progress.json")
OUTPUT_DIR = str(config.OUTPUT_DIR / "translation")

# -------------------- 术语表 --------------------
def parse_glossary(glossary_text: str) -> Dict[str, str]:
    """解析术语表文本为字典。
    每行格式: 原文=译文
    跳过空行和不含 = 的行，去首尾空格，重复键取最后一个。
    """
    result = {}
    if not glossary_text or not glossary_text.strip():
        return result
    for line in glossary_text.strip().splitlines():
        line = line.strip()
        if not line or '=' not in line:
            continue
        key, _, value = line.partition('=')
        key, value = key.strip(), value.strip()
        if key and value:
            result[key] = value
    return result

def build_glossary_prompt(glossary: Dict[str, str]) -> str:
    """将术语表字典构建为 prompt 前缀指令。"""
    if not glossary:
        return ""
    lines = ["术语表（必须遵守）："]
    for src, tgt in glossary.items():
        lines.append(f"- {src} → {tgt}")
    lines.append("")  # 空行分隔
    return "\n".join(lines)

# -------------------- Prompt 模板 --------------------
PROMPT_FIRST = f"""{{glossary}}You are a professional literary translator. Translate the following {SOURCE_LANG} text into {TARGET_LANG}.
Requirements:
- Accurate and fluent literary translation.
- Keep paragraph breaks exactly as in the original.
- Output ONLY the {TARGET_LANG} translation. No extra notes, no explanations, no preamble.

{SOURCE_LANG} text:
{{text}}

{TARGET_LANG} translation:"""

PROMPT_CONTINUE = f"""{{glossary}}Continue translating the following {SOURCE_LANG} text into {TARGET_LANG}.
The end of the preceding translation (last ~1000 characters only):
---
{{prev_translation}}
---
Now translate the next segment. Do NOT repeat the preceding translation. Do NOT add summaries or commentary.
Maintain consistent terminology and style with the preceding translation. Output ONLY the new translation part.

{SOURCE_LANG} text to translate now:
{{text}}

{TARGET_LANG} translation (new part only):"""

# -------------------- 工具函数 --------------------
def count_tokens(text: str) -> int:
    """粗略估算 token 数（用于展示，非精确限制）"""
    chinese = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    other = len(text) - chinese - sum(len(w) for w in re.findall(r'[a-zA-Z]+', text))
    return chinese + int(english_words * 1.3) + other

def load_full_text(file_path: str) -> str:
    """读取整个文件为字符串，并显示大小信息"""
    print(f"📂 正在读取文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    size_mb = len(text.encode('utf-8')) / (1024 * 1024)
    print(f"✅ 读取完成，文件大小: {size_mb:.2f} MB，字符数: {len(text)}，预估 Token: {count_tokens(text)}")
    return text

def split_into_chapters_fast(text: str) -> List[Tuple[str, str]]:
    """
    高效章节切分：用正则一次性找到所有章节/卷名位置，按位置切片。
    匹配模式包括：
    - 第X章 / 第X节 / 第X卷
    - 上卷 / 中卷 / 下卷 / 正文卷 / 番外卷
    """
    print("🔪 正在切分章节（正则匹配中）...")
    start_time = time.time()

    # 组合正则：章节号支持中文数字/阿拉伯数字，卷名支持常见卷标
    pattern = re.compile(
        r'^(?:正文卷|上卷|中卷|下卷|第[一二三四五六七八九十百千万\d]+[卷章节].*)$',
        re.MULTILINE
    )

    # 找出所有标题的起始位置和内容
    matches = list(pattern.finditer(text))
    if not matches:
        print("⚠️ 未找到任何章节标题，将整本书作为单章处理")
        return [("全文", text.strip())]

    chapters = []
    # 处理第一个标题之前的内容（前言/简介）
    first_match = matches[0]
    if first_match.start() > 0:
        preface = text[:first_match.start()].strip()
        if preface:
            chapters.append(("前言", preface))

    # 依次处理每个章节
    for i, match in enumerate(matches):
        title = match.group().strip()
        start = match.end()  # 内容开始位置（标题之后）
        # 内容结束位置：下一个标题的开始，或文本末尾
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        content = text[start:end].strip()
        chapters.append((title, content))

    elapsed = time.time() - start_time
    print(f"✅ 切分完成，共 {len(chapters)} 个章节，耗时 {elapsed:.2f} 秒")
    return chapters

def split_long_chapter(content: str, max_chars: int = 4000) -> List[str]:
    """将过长的章节内容按段落切分为多个子块"""
    if len(content) <= max_chars:
        return [content]

    paragraphs = content.split('\n')
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 > max_chars and current:
            chunks.append(current)
            current = para
        else:
            current = current + '\n' + para if current else para
    if current:
        chunks.append(current)
    return chunks

def load_progress(progress_file: str) -> Dict:
    """加载翻译进度"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"last_chapter_index": 0, "translations": []}

def save_progress(progress_file: str, last_idx: int, translations: List[str], lock: threading.Lock = None):
    """保存翻译进度（线程安全）"""
    data = {"last_chapter_index": last_idx, "translations": translations}
    if lock:
        with lock:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def append_to_output(output_file: str, new_translations: List[str]):
    """追加译文到输出文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a', encoding='utf-8') as f:
        for trans in new_translations:
            f.write(trans + '\n\n')

def get_llm(model_name: str) -> ChatOpenAI:
    """初始化 LLM 实例"""
    return ChatOpenAI(
        model=model_name,
        streaming=False,
        temperature=0.5,
        max_tokens=MAX_CONTEXT_TOKENS,
        base_url=config.OPENAI_BASE_URL,
        api_key=config.OPENAI_API_KEY,
        extra_body=config.get_llm_extra_body()
    )

# -------------------- 单章翻译（供线程调用）--------------------
def translate_single_chapter(
    chapter_idx: int,                # 全局章节索引（0-based）
    chapter_title: str,
    chapter_content: str,
    prev_translation: str,           # 上一章完整译文（用于本章第一个子块的上下文）
    model_name: str,
    llm_instance: ChatOpenAI = None,
    glossary_prompt: str = ""        # 术语表 prompt 前缀（可选）
) -> str:
    """
    翻译一个完整章节，内部自动切分子块，返回带英文标题的译文。
    """
    if llm_instance is None:
        llm_instance = get_llm(model_name)

    # 切分子块
    sub_chunks = split_long_chapter(chapter_content, max_chars=4000)
    translated_parts = []

    for sub_idx, chunk in enumerate(sub_chunks):
        # 确定上下文
        if chapter_idx == 0 and sub_idx == 0:
            prompt = PROMPT_FIRST.format(glossary=glossary_prompt, text=chunk)
        else:
            if sub_idx == 0:
                # 本章第一块，使用上一章完整译文（可能为空）
                ctx = prev_translation if prev_translation else ""
            else:
                # 同章内后续块，使用前一块的译文
                ctx = translated_parts[-1]
            # 截断上下文防止过长（取末尾1000字符）
            ctx_short = ctx[-1000:] if len(ctx) > 1000 else ctx
            prompt = PROMPT_CONTINUE.format(glossary=glossary_prompt, prev_translation=ctx_short, text=chunk)

        # 调用 LLM（带重试机制）
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = llm_instance.invoke(prompt)
                translation = response.content.strip()
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"   ⚠️ 章节 {chapter_idx+1} 子块 {sub_idx+1} 翻译失败，重试 {attempt+1}/{max_retries}: {e}")
                time.sleep(2)

        translated_parts.append(translation)

    full_translation = '\n\n'.join(translated_parts)

    # 生成英文标题（简单映射）
    ch_match = re.search(r'第([零一二三四五六七八九十百千万\d]+)[章卷]', chapter_title)
    if ch_match:
        num_cn = ch_match.group(1)
        # 中文数字转阿拉伯数字（简化版，仅支持常见形式）
        cn_num_map = {'零':'0','一':'1','二':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9','十':'10','百':'100'}
        num_ar = cn_num_map.get(num_cn, num_cn)
        eng_title = f"### Chapter {num_ar}"
    else:
        eng_title = f"### {chapter_title}"

    return f"{eng_title}\n\n{full_translation}"

# -------------------- 主翻译流程（并行批次）--------------------
def translate_book_parallel(
    input_file: str,
    output_file: str,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 10,
    workers: int = 2,
    resume: bool = True,
    max_chapters: Optional[int] = None,
    progress_callback=None,
    glossary_text: str = ""
):
    print(f"\n{'='*60}")
    print(f"🚀 开始翻译任务")
    print(f"📖 输入: {input_file}")
    print(f"📝 输出: {output_file}")
    print(f"🤖 模型: {model_name}")
    print(f"🧵 每批: {batch_size} 章 | 并行数: {workers}")
    print(f"{'='*60}\n")

    # 0. 解析术语表
    glossary = parse_glossary(glossary_text)
    glossary_prompt = build_glossary_prompt(glossary)
    if glossary:
        print(f"📋 术语表已加载，共 {len(glossary)} 条：")
        for k, v in glossary.items():
            print(f"   {k} → {v}")
        print()

    # 1. 读取原文
    full_text = load_full_text(input_file)

    # 2. 切分章节
    chapters = split_into_chapters_fast(full_text)
    total_chapters = len(chapters)
    if max_chapters:
        chapters = chapters[:max_chapters]
        total_chapters = len(chapters)
        print(f"🔒 限制翻译前 {max_chapters} 章")

    # 3. 加载进度
    progress = load_progress(PROGRESS_FILE) if resume else {"last_chapter_index": 0, "translations": []}
    start_idx = progress["last_chapter_index"]
    existing_translations = progress["translations"]

    if start_idx >= total_chapters:
        print("✅ 所有章节已翻译完成，无需继续。")
        return

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 如果是从头开始且输出文件存在，清空旧内容
    if start_idx == 0 and os.path.exists(output_file):
        open(output_file, 'w', encoding='utf-8').close()
        print("🧹 已清空旧输出文件")

    # 4. 初始化 LLM（所有线程共享）
    llm = get_llm(model_name)
    progress_lock = threading.Lock()

    # 5. 分批处理章节
    _stop_flag.clear()
    for batch_start in range(start_idx, total_chapters, batch_size):
        if _stop_flag.is_set():
            print("\n⏹️ 已请求停止翻译")
            break
        batch_end = min(batch_start + batch_size, total_chapters)
        print(f"\n{'─'*50}")
        print(f"📦 正在处理批次: 第 {batch_start+1} ~ {batch_end} 章 (总进度 {batch_start}/{total_chapters})")
        batch_start_time = time.time()

        # 准备本批次任务
        tasks = []
        # 上一章的译文（用于本批次第一个章节的上下文）
        prev_translation = existing_translations[-1] if existing_translations else ""

        for idx in range(batch_start, batch_end):
            title, content = chapters[idx]
            # 该章节的前一章译文（如果已存在）
            if idx == 0:
                prev_for_this = ""
            elif idx <= len(existing_translations):
                prev_for_this = existing_translations[idx - 1]
            else:
                # 理论上不应出现，因为本批次之前的章节都应已翻译并存入 existing_translations
                prev_for_this = prev_translation  # fallback
            tasks.append((idx, title, content, prev_for_this))

        # 并发执行本批次
        batch_results = [None] * (batch_end - batch_start)
        batch_done = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {}
            for idx, title, content, prev_trans in tasks:
                future = executor.submit(
                    translate_single_chapter,
                    idx, title, content, prev_trans, model_name, llm, glossary_prompt
                )
                future_to_idx[future] = idx

            # 收集结果（保持顺序）
            for future in as_completed(future_to_idx):
                if _stop_flag.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    print("   ⏹️ 已请求停止翻译")
                    break
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    batch_results[idx - batch_start] = result
                    print(f"   ✅ 第 {idx+1} 章完成: {chapters[idx][0]}")
                except Exception as e:
                    print(f"   ❌ 第 {idx+1} 章失败: {e}")
                    batch_results[idx - batch_start] = f"### Chapter {idx+1} (Failed)\n\n[Translation Error]"
                batch_done += 1
                if progress_callback:
                    progress_callback(batch_start + batch_done, total_chapters)

        # 过滤掉 None（理论上不应有）
        valid_results = [r for r in batch_results if r is not None]

        # 更新总译文列表
        all_translations = existing_translations + valid_results
        # 写入文件并保存进度
        append_to_output(output_file, valid_results)
        save_progress(PROGRESS_FILE, batch_end, all_translations, progress_lock)

        batch_elapsed = time.time() - batch_start_time
        print(f"💾 进度已保存: {batch_end}/{total_chapters} 章，本批次耗时 {batch_elapsed:.1f} 秒")
        existing_translations = all_translations

    if _stop_flag.is_set():
        print(f"\n⏹️ 翻译已停止。已完成的译文已保存至: {os.path.abspath(output_file)}")
    else:
        print(f"\n🎉 全部翻译完成！输出文件: {os.path.abspath(output_file)}")

# -------------------- 命令行入口 --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="高效并行翻译长篇小说（按章节）")
    parser.add_argument("--input", "-i", type=str, default=str(config.DATA_DIR / "texts" / "新白蛇问仙.txt"), help="输入文件路径")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出文件路径（默认自动生成）")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, help="模型名称")
    parser.add_argument("--batch", "-b", type=int, default=10, help="每批处理的章节数")
    parser.add_argument("--workers", "-w", type=int, default=2, help="并行线程数（建议1~4）")
    parser.add_argument("--resume", action="store_true", default=True, help="从上次中断处继续（默认）")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="忽略进度，从头开始")
    parser.add_argument("--max-chapters", type=int, default=None, help="最多翻译章数（测试用）")
    parser.add_argument("--glossary", "-g", type=str, default="", help="术语表文本（每行: 原文=译文）或术语表文件路径")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 错误：找不到输入文件 {args.input}")
        sys.exit(1)

    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # output_file = os.path.join(OUTPUT_DIR, f"translation_{timestamp}.txt")
        output_file = os.path.join(OUTPUT_DIR, f"translation.txt")
    else:
        output_file = args.output

    # 解析术语表：支持文件路径或内联文本
    glossary_text = ""
    if args.glossary:
        if os.path.isfile(args.glossary):
            with open(args.glossary, 'r', encoding='utf-8') as gf:
                glossary_text = gf.read()
        else:
            glossary_text = args.glossary

    translate_book_parallel(
        input_file=args.input,
        output_file=output_file,
        model_name=args.model,
        batch_size=args.batch,
        workers=args.workers,
        resume=args.resume,
        max_chapters=args.max_chapters,
        glossary_text=glossary_text
    )