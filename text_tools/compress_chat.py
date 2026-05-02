#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超长聊天记录 .txt → 消息级分块 → 智能压缩 → 轻量文本输出
修复版：压缩失败时保留原文，确保信息零丢失
"""

import re
import sys
import config
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from tqdm import tqdm

# ================= 配置 =================
config.OPENAI_BASE_URL = "http://localhost:1234/v1"
config.OPENAI_API_KEY = "lm-studio"
config.VISION_MODEL_THINKING = "qwen/qwen3.6-35b-a3b-Thinking"

config.DEFAULT_CHUNK_SIZE = 20480      # 每块最大字符数
config.OVERLAP_MESSAGES = 2            # 块间重叠消息数
config.RETRY_TIMES = 2                  # 压缩失败重试次数

COMPRESS_PROMPT = """你是一个聊天记录压缩助手，请将以下原始聊天记录压缩成简洁版本。

**压缩要求：**
1. **绝不丢失任何聊天内容**：所有消息必须完整保留，包括表情包、图片占位符、文件提示等。
2. **合并冗余时间**：
   - 如果连续多条消息发生在很短时间内（比如 10 分钟内），只需在第一行标注时间范围，后续消息省略时间。
   - 当对话出现明显停顿（超过 30 分钟空档）或日期变更时，重新标注新的时间点。
   - 标注格式示例：`【01-09 22:13-22:24】` 或 `【01-09 22:17】`（单条无连续可只标一点）。
3. **保留发言者标识**：保持原有的昵称或标识不变。
4. **不添加任何总结、解释或额外空行**：输出纯压缩后的对话文本。
5. **忽略系统消息（如“以上是打招呼的消息”）** 如果它不影响对话理解，可以删除；如果含有实质内容则保留。

**输出格式（直接输出压缩文本）：**
【01-09 22:13】🐱：你已添加了🐱...打招呼的消息
【01-09 22:17－22:24】
我：[表情包]
🐱：[表情包]
🐱：听说你小汁要找对象
...
"""


def parse_messages(text: str) -> List[Tuple[str, str, str]]:
    """增强的消息解析，支持多种时间戳和昵称格式"""
    text = text.lstrip('\ufeff').strip()
    lines = text.split('\n')
    messages = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # 匹配时间戳行，兼容有/无引号、用 T 分隔等
        m = re.match(
            r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}:\d{2})\s+"
            r"(?:'([^']*)'|【.*?】|\[.*?\]|(\S+))\s*$",
            line
        )
        if not m:
            m2 = re.match(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}:\d{2})\s+(.+)", line)
            if m2:
                raw_speaker = m2.group(2).strip().strip("'\"“”")
                timestamp = m2.group(1) + " '" + raw_speaker + "'"
                speaker = raw_speaker
                i += 1
            else:
                i += 1
                continue
        else:
            timestamp = m.group(0).strip()
            if m.group(2) is not None:
                speaker = m.group(2)
            elif m.group(3) is not None:
                speaker = m.group(3)
            else:
                i += 1
                continue
            i += 1

        # 收集消息体
        body_lines = []
        while i < n:
            next_line = lines[i]
            if re.match(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}:\d{2}\s+", next_line):
                break
            if next_line.strip() == "" and i+1 < n and re.match(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}:\d{2}\s+", lines[i+1]):
                break
            body_lines.append(next_line.rstrip())
            i += 1
        body = '\n'.join(body_lines).strip()
        messages.append((timestamp, speaker, body))
    return messages


def split_messages_into_chunks(messages, max_chunk_size, overlap=config.OVERLAP_MESSAGES):
    """按消息完整分块，带重叠，保证不丢消息"""
    chunks = []
    current_texts = []
    current_len = 0
    for msg in messages:
        ts, sp, body = msg
        msg_text = f"{ts}\n{body}"
        length = len(msg_text) + 1
        if current_texts and current_len + length > max_chunk_size:
            chunks.append('\n'.join(current_texts))
            # 重叠选用最后 overlap 条消息
            current_texts = current_texts[-overlap:] if len(current_texts) >= overlap else current_texts[:]
            current_len = sum(len(m)+1 for m in current_texts)
        current_texts.append(msg_text)
        current_len += length
    if current_texts:
        chunks.append('\n'.join(current_texts))
    return chunks


def compress_chunk(chunk_text, chunk_idx, model, temperature, max_tokens):
    """
    压缩单个块，带重试和降级：失败后保留原始文本。
    """
    for attempt in range(config.RETRY_TIMES + 1):
        try:
            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=config.OPENAI_BASE_URL,
                api_key=config.OPENAI_API_KEY,
                request_timeout=300,
                extra_body=config.get_llm_extra_body()
            )
            prompt = COMPRESS_PROMPT + f"\n[这是第{chunk_idx+1}块，请压缩并确保与前后连贯]\n\n"
            resp = llm.invoke([HumanMessage(content=prompt + chunk_text)])
            return resp.content.strip()
        except Exception as e:
            if attempt < config.RETRY_TIMES:
                print(f"   ⚠️ 第{chunk_idx+1}块压缩失败 (尝试 {attempt+1}/{config.RETRY_TIMES+1}): {e}，等待重试...")
                import time
                time.sleep(2)
            else:
                print(f"   ❌ 第{chunk_idx+1}块压缩彻底失败，将保留原始文本以免丢失信息。错误: {e}")
                # 降级：返回原始块文本（加上标记）
                return f"[压缩失败，保留原文]\n{chunk_text}"
    return ""


def remove_duplicate_time_header(text, prev_tail):
    """去除相邻块开头重复的时间段"""
    m = re.match(r'(【[^】]+】)', text)
    if m and prev_tail.endswith(m.group(1)):
        return text[len(m.group(1)):].strip()
    return text


def process_single_text_file(txt_path, output_txt_path=None, model=config.VISION_MODEL_THINKING,
                             temperature=0.2, max_tokens=4000, chunk_size=config.DEFAULT_CHUNK_SIZE,
                             internal_workers=2):
    print(f"\n📄 处理: {txt_path.name}")
    with open(txt_path, 'r', encoding='utf-8-sig') as f:
        raw_text = f.read()

    messages = parse_messages(raw_text)
    if not messages:
        print("   ❌ 未识别到任何消息。以下是文件前5行供检查：")
        preview = '\n'.join(raw_text.split('\n')[:5])
        print(repr(preview))
        return ""
    print(f"   📊 共解析 {len(messages)} 条消息")  # 这里是原始消息总数，检查是否就少了

    chunks = split_messages_into_chunks(messages, chunk_size, config.OVERLAP_MESSAGES)
    total = len(chunks)
    print(f"   ✂️  分为 {total} 块 (≤{chunk_size}字符/块, 重叠{config.OVERLAP_MESSAGES}条)")

    results = [None] * total

    # 根据块数量决定是否并发
    if total == 1 or internal_workers <= 1:
        for i, chunk in enumerate(chunks):
            print(f"      🚀 压缩第{i+1}/{total}块...")
            results[i] = compress_chunk(chunk, i, model, temperature, max_tokens)
    else:
        print(f"   📤 启动 {internal_workers} 路并发压缩...")
        with ThreadPoolExecutor(max_workers=internal_workers) as executor:
            fut_map = {executor.submit(compress_chunk, chunks[i], i, model, temperature, max_tokens): i
                       for i in range(total)}
            for fut in as_completed(fut_map):
                idx = fut_map[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    print(f"      ❌ 第{idx+1}块执行异常: {e}，保留原文")
                    results[idx] = f"[执行异常，保留原文]\n{chunks[idx]}"
                if results[idx]:
                    print(f"      ✅ 第{idx+1}块完成，输出 {len(results[idx])} 字符")
                else:
                    # 极端情况，仍然用原文兜底
                    print(f"      ❌ 第{idx+1}块结果为空，使用原文")
                    results[idx] = f"[结果为空，保留原文]\n{chunks[idx]}"

    # 合并所有块（绝不跳过任何一块）
    merged = ""
    for i, res in enumerate(results):
        if not res:  # 理论上不会出现了，但以防万一
            print(f"   ⚠️ 第{i+1}块结果意外为空，使用原文填充")
            res = f"[意外丢失，保留原文]\n{chunks[i]}"
        if i == 0:
            merged = res
        else:
            prev_tail = merged.strip().split('\n')[-1] if merged else ""
            cleaned = remove_duplicate_time_header(res.strip(), prev_tail)
            merged = merged.strip() + "\n" + cleaned

    if not merged.strip():
        print("   ❌ 合并结果为空")
        return ""

    # 简单整理
    merged = re.sub(r'\n{3,}', '\n\n', merged)
    print(f"   ✅ 压缩完成，原始 {len(raw_text)} 字符 → 压缩后 {len(merged)} 字符")

    if output_txt_path is None:
        output_txt_path = txt_path.with_suffix(".compressed.txt")
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(merged)
    print(f"   💾 已保存: {output_txt_path}")
    return merged


def process_folder(input_dir, output_dir=None, model=config.VISION_MODEL_THINKING, temperature=0.2,
                   max_tokens=4000, chunk_size=config.DEFAULT_CHUNK_SIZE, internal_workers=2):
    txt_files = list(Path(input_dir).glob("*.txt"))
    if not txt_files:
        print("❌ 无 .txt 文件")
        return
    print(f"📂 找到 {len(txt_files)} 个文件")
    if output_dir is None:
        output_dir = Path(input_dir) / "compressed_output"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for f in txt_files:
        out = output_dir / f"{f.stem}.compressed.txt"
        try:
            process_single_text_file(f, out, model, temperature, max_tokens, chunk_size, internal_workers)
        except Exception as e:
            print(f"❌ 处理 {f.name} 失败: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default=str(config.DATA_DIR / "screenshots" / "texts"))
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--model", default=config.VISION_MODEL_THINKING)
    parser.add_argument("--temperature", "-t", type=float, default=0.2)
    parser.add_argument("--chunk-size", type=int, default=config.DEFAULT_CHUNK_SIZE)
    parser.add_argument("--internal-workers", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4000)
    args = parser.parse_args()

    inp = Path(args.input)
    if inp.is_file():
        process_single_text_file(inp, args.output, args.model, args.temperature,
                                 args.max_tokens, args.chunk_size, args.internal_workers)
    elif inp.is_dir():
        process_folder(inp, args.output, args.model, args.temperature,
                       args.max_tokens, args.chunk_size, args.internal_workers)
    else:
        print("❌ 输入路径无效")