#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LocalAITools - Gradio Web 界面
启动方式：python app.py  或  双击 run.bat
然后浏览器打开 http://localhost:7860
"""

import os
import sys
import io
import logging
import subprocess
from pathlib import Path
from contextlib import redirect_stdout

# 控制台日志：记录用户操作，方便排查问题
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("LocalAITools")

sys.path.insert(0, str(Path(__file__).parent))
import config
import history

import gradio as gr

ROOT_DIR = Path(__file__).parent


def _make_title(text):
    return f"## {text}"


def _capture_log(fn, *args, **kwargs):
    """捕获 logging 输出为字符串，异常时返回错误信息"""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        fn(*args, **kwargs)
    except Exception as e:
        import traceback
        stream.write(f"❌ 处理出错: {e}\n")
        stream.write(traceback.format_exc())
    finally:
        root.removeHandler(handler)
    return stream.getvalue() or "✅ 处理完成"


# ============================================================
# 工具函数通用：应用供应商配置
# ============================================================
def _apply_provider(prov):
    """将供应商信息写入环境变量，使 config 模块读取到正确的 API 地址和密钥"""
    if prov and isinstance(prov, dict):
        if prov.get("base_url"):
            os.environ["OPENAI_BASE_URL"] = prov["base_url"]
            config.OPENAI_BASE_URL = prov["base_url"]
        if prov.get("api_key") is not None:
            os.environ["OPENAI_API_KEY"] = prov["api_key"]
            config.OPENAI_API_KEY = prov["api_key"]


# ============================================================
# Tab 1: 图片重命名
# ============================================================
def _rename_images(input_dir, model, workers, dry_run, keep_original, mode, context_count, max_size, provider, thinking=True, progress=gr.Progress()):
    _apply_provider(provider)
    os.environ["ENABLE_THINKING"] = "true" if thinking else "false"
    logger.info(f"[图片重命名] 目录={input_dir} 模型={model} 模式={mode} 线程={workers} 保留原名={keep_original} max_size={max_size}")
    from image_tools.rename_images import process_one_image, get_shared_llm
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor, as_completed

    input_path = Path(input_dir)
    if not input_path.is_dir():
        return "❌ 请输入有效的文件夹路径"

    exts = [".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif"]
    images = []
    for ext in exts:
        images.extend(input_path.glob(f"*{ext}"))
        images.extend(input_path.glob(f"*{ext.upper()}"))
    images = sorted(set(images))

    if not images:
        return "📂 未找到图片文件"

    get_shared_llm(model or config.RENAME_MODEL)
    recent_history = deque(maxlen=max(1, context_count))
    effective_max_size = max_size if max_size and max_size > 0 else None
    results = []
    total = len(images)
    completed = 0

    from image_tools.rename_images import _stop_flag
    _stop_flag.clear()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one_image, img, model or config.RENAME_MODEL, dry_run, recent_history, keep_original, mode, effective_max_size): img
                   for img in images}
        for i, future in enumerate(as_completed(futures)):
            if _stop_flag.is_set():
                executor.shutdown(wait=False, cancel_futures=True)
                results.append("⏹️ 已请求停止")
                break
            try:
                old_name, new_phrase = future.result()
                results.append(f"{old_name} → {new_phrase}")
            except Exception as e:
                results.append(f"❌ 错误: {e}")
            completed += 1
            progress(completed / total, desc=f"重命名中 {completed}/{total}")

    if _stop_flag.is_set():
        result = f"已停止：已完成 {completed}/{total} 张图片\n\n" + "\n".join(results)
    else:
        result = f"处理完成：{total} 张图片\n\n" + "\n".join(results)
    config.save_state("rename", input_dir=input_dir, model=model or config.RENAME_MODEL, workers=workers, dry_run=dry_run)
    history.add_entry("图片重命名", input_dir, f"处理 {total} 张图片")
    return result


def _classify_by_work(input_dir, dry_run, min_count):
    logger.info(f"[作品分类] 目录={input_dir} 试运行={dry_run} 最少={min_count}")
    from image_tools.rename_images import classify_by_work
    results = classify_by_work(input_dir, dry_run, min_count=int(min_count))
    if not results:
        return "📂 未找到可分类的图片"
    prefix = "🧪 [试运行]\n" if dry_run else ""
    return prefix + "\n".join(results)


# ============================================================
# Tab 2: 图片质量评分
# ============================================================
def _score_images(input_dir, mode, custom_prompt, model, max_size, provider, thinking=True, progress=gr.Progress()):
    _apply_provider(provider)
    os.environ["ENABLE_THINKING"] = "true" if thinking else "false"
    logger.info(f"[质量评分] 目录={input_dir} 模式={mode} 模型={model} max_size={max_size}")
    from image_tools.detect_ai_errors import score_images

    effective_max_size = max_size if max_size and max_size > 0 else None

    def on_progress(completed, total):
        progress(completed / total, desc=f"评分中 {completed}/{total}")

    result = _capture_log(score_images, input_dir, mode, on_progress, custom_prompt, model or None, effective_max_size)
    config.save_state("score", input_dir=input_dir, mode=mode, model=model, custom_prompt=custom_prompt)
    mode_labels = {"ai":"AI检测","photo":"漫展摄影","general":"通用照片","portrait":"人像","landscape":"风景","document":"文档扫描","art":"绘画插图"}
    mode_label = mode_labels.get(mode, mode)
    history.add_entry(f"质量评分({mode_label})", input_dir, "评分完成")
    return result


def _classify_images(input_dir, classify_method, top_percent, bottom_percent,
                     min_score, max_score, progress=gr.Progress()):
    logger.info(f"[质量分类] 目录={input_dir} 方法={classify_method} top={top_percent} bottom={bottom_percent}")
    from image_tools.detect_ai_errors import classify_images

    use_threshold = (classify_method == "threshold")
    result = _capture_log(classify_images, input_dir, None,
                           top_percent / 100 if top_percent else None,
                           bottom_percent / 100 if bottom_percent else None,
                           min_score, max_score, use_threshold)
    config.save_state("score", input_dir=input_dir, classify_method=classify_method,
                       top_percent=top_percent, bottom_percent=bottom_percent,
                       min_score=min_score, max_score=max_score)
    history.add_entry("质量分类", input_dir, "分类完成")
    return result


# ============================================================
# Tab 3: 聊天截图识别
# ============================================================
def _explain_images(input_dir, vision_model, temperature, workers, internal_workers, max_tokens,
                    max_size, provider, thinking=True, progress=gr.Progress()):
    logger.info(f"[图片解释] 目录={input_dir} 模型={vision_model} 线程={workers} max_size={max_size}")
    _apply_provider(provider)
    os.environ["ENABLE_THINKING"] = "true" if thinking else "false"
    from image_tools.ocr_chat_screenshots import process_folder

    input_path = Path(input_dir)
    if not input_path.is_dir():
        return "❌ 请输入有效的文件夹路径"

    output_dir = input_path / "chat_text_output"
    output_dir.mkdir(exist_ok=True)

    def on_progress(completed, total):
        progress(completed / total, desc=f"识别中 {completed}/{total}")

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
            max_size=effective_max_size
        )

    result = f"✅ 输出目录: {output_dir}\n\n{buf.getvalue()}"
    config.save_state("ocr", input_dir=input_dir, model=vision_model, temperature=temperature,
                       workers=workers, internal_workers=internal_workers, max_tokens=max_tokens)
    history.add_entry("截图识别", input_dir, "文字提取完成")
    return result


# ============================================================
# Tab 4: 聊天记录压缩
# ============================================================
def _compress_text(input_path, model, temperature, chunk_size, internal_workers, max_tokens,
                   provider, thinking=True, progress=gr.Progress()):
    _apply_provider(provider)
    os.environ["ENABLE_THINKING"] = "true" if thinking else "false"
    logger.info(f"[聊天压缩] 路径={input_path} 模型={model} 块大小={chunk_size}")
    from text_tools.compress_chat import process_single_text_file, process_folder

    def on_progress(completed, total):
        progress(completed / total, desc=f"压缩中 {completed}/{total}")

    p = Path(input_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        if p.is_file():
            process_single_text_file(
                p, None,
                model or config.VISION_MODEL_THINKING,
                temperature, max_tokens, chunk_size, internal_workers,
                progress_callback=on_progress
            )
        elif p.is_dir():
            process_folder(
                p, None,
                model or config.VISION_MODEL_THINKING,
                temperature, max_tokens, chunk_size, internal_workers,
                progress_callback=on_progress
            )
        else:
            return "❌ 路径无效"

    result = f"✅ 处理完成\n\n{buf.getvalue()}"
    config.save_state("compress", input_path=input_path, model=model, temperature=temperature,
                       chunk_size=chunk_size, internal_workers=internal_workers, max_tokens=max_tokens)
    history.add_entry("聊天压缩", input_path, "压缩完成")
    return result


# ============================================================
# Tab 5: 文本翻译
# ============================================================
def _translate(input_file, output_file, model, batch_size, workers, provider, thinking=True, progress=gr.Progress()):
    _apply_provider(provider)
    os.environ["ENABLE_THINKING"] = "true" if thinking else "false"
    logger.info(f"[文本翻译] 文件={input_file} 模型={model} 批大小={batch_size} 线程={workers}")
    from text_tools.translate import translate_book_parallel

    if not Path(input_file).is_file():
        return "❌ 输入文件不存在"

    output_file = output_file or str(config.OUTPUT_DIR / "translation" / "translation.txt")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    def on_progress(completed, total):
        progress(completed / total, desc=f"翻译中 {completed}/{total}")

    buf = io.StringIO()
    with redirect_stdout(buf):
        translate_book_parallel(
            input_file, output_file,
            model or config.TEXT_MODEL,
            batch_size, workers,
            resume=True,
            progress_callback=on_progress
        )

    result = f"✅ 输出: {output_file}\n\n{buf.getvalue()}"
    config.save_state("translate", input_file=input_file, output_file=output_file,
                       model=model, batch_size=batch_size, workers=workers)
    history.add_entry("文本翻译", input_file, "翻译完成")
    return result


# ============================================================
# Tab 6: 知识库问答（对话式）
# ============================================================
_KB_HISTORY_DIR = config.OUTPUT_DIR / "kb_chats"
_KB_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _kb_save_chat(chat_messages, chat_name):
    """保存聊天记录到文件"""
    if not chat_messages:
        return "❌ 没有聊天记录可保存"
    logger.info(f"[保存聊天] 名称={chat_name} 消息数={len(chat_messages)}")
    if not chat_name or not chat_name.strip():
        chat_name = f"chat_{__import__('time').strftime('%Y%m%d_%H%M%S')}"
    chat_name = chat_name.strip()
    safe_name = __import__('re').sub(r'[\\/*?:"<>|]', '', chat_name)
    path = _KB_HISTORY_DIR / f"{safe_name}.json"
    data = {
        "name": chat_name,
        "time": __import__('time').strftime("%Y-%m-%d %H:%M:%S"),
        "messages": chat_messages,
    }
    path.write_text(__import__('json').dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"✅ 已保存: {path.name}"


def _kb_get_saved_chat(selection):
    """根据名称查找已保存的聊天文件，返回 (path, data) 或 (None, None)"""
    for f in sorted(_KB_HISTORY_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = __import__('json').loads(f.read_text(encoding="utf-8"))
            name = _kb_clean_name(data.get("name", f.stem))
            if name == selection or f.stem == selection:
                return f, data
        except Exception:
            continue
    return None, None


def _kb_clean_name(raw: str) -> str:
    """去掉旧版本可能附带的 ' (YYYY-MM-DD HH:MM:SS, N条)' 后缀"""
    import re as _re
    return _re.sub(r'\s*\(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\s*\d+条\)\s*$', '', raw)


def _kb_list_chats():
    """列出所有保存的聊天记录"""
    files = sorted(_KB_HISTORY_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return [], "暂无保存的聊天记录"
    choices = []
    for f in files[:50]:
        try:
            data = __import__('json').loads(f.read_text(encoding="utf-8"))
            choices.append(_kb_clean_name(data.get("name", f.stem)))
        except Exception:
            choices.append(f.stem)
    return choices, f"共 {len(files)} 条记录"


def _kb_load_chat(selection):
    """加载选中的聊天记录"""
    if not selection:
        return [], ""
    _, data = _kb_get_saved_chat(selection)
    if data:
        return data.get("messages", []), f"✅ 已加载: {data.get('name', selection)}"
    return [], "❌ 未找到匹配的记录"


def _kb_delete_chat(selection):
    """删除选中的聊天记录"""
    if not selection:
        return "❌ 请选择要删除的记录"
    f, data = _kb_get_saved_chat(selection)
    if f:
        name = data.get("name", selection)
        f.unlink()
        return f"✅ 已删除: {name}"
    return "❌ 未找到匹配的记录"


def _do_web_search(query, max_results=5):
    """联网搜索，优先用 Bing 中国版，失败回退到 DuckDuckGo"""
    results = _search_bing_cn(query, max_results)
    if results is not None:
        return results
    results = _search_ddg(query, max_results)
    if results is not None:
        return results
    return None


def _search_bing_cn(query, max_results=5):
    """Bing 中国版搜索 (cn.bing.com)"""
    try:
        import requests
        from lxml import html
    except ImportError:
        return None

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        resp = requests.get("https://cn.bing.com/search",
                            params={"q": query}, headers=headers, timeout=10)
        resp.encoding = "utf-8"
        tree = html.fromstring(resp.text)
        items = tree.xpath('//li[contains(@class, "b_algo")]')
        results = []
        for r in items[:max_results]:
            title_els = r.xpath(".//h2//text()") or r.xpath(".//a//text()")
            title = " ".join(title_els).strip() if title_els else ""
            snippet_els = r.xpath(
                './/p[contains(@class, "b_lineclamp") or contains(@class, "b_snippet")]//text()'
            )
            snippet = " ".join(snippet_els).strip() if snippet_els else ""
            if not snippet:
                snippet_els = r.xpath('.//div[contains(@class, "b_caption")]//p//text()')
                snippet = " ".join(snippet_els).strip() if snippet_els else ""
            link_els = r.xpath(".//h2//a/@href")
            link = link_els[0] if link_els else ""
            if title:
                results.append(f"【{title}】\n{snippet}\n来源: {link}")
        return "\n\n".join(results) if results else None
    except Exception as e:
        logger.warning(f"Bing 搜索失败: {e}")
        return None


def _search_ddg(query, max_results=5):
    """DuckDuckGo 搜索（通过 ddgs 库）"""
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"【{r.get('title', '')}】\n{r.get('body', '')}\n来源: {r.get('href', '')}")
        return "\n\n".join(results) if results else None
    except ImportError:
        logger.warning("ddgs 未安装，DuckDuckGo 搜索不可用")
        return None
    except Exception as e:
        logger.warning(f"DuckDuckGo 搜索失败: {e}")
        return None


def _query_kb_chat(query, keyword, model, k, batch_size, chat_messages, provider, thinking=True, use_kb=True, use_web=False, progress=gr.Progress()):
    """对话式知识库问答 handler（阻塞版，旧接口保留兼容）"""
    _apply_provider(provider)
    os.environ["ENABLE_THINKING"] = "true" if thinking else "false"

    if not query or not query.strip():
        return chat_messages, "❌ 请输入问题"

    progress_lines = []
    def on_progress(msg: str):
        progress_lines.append(msg)

    web_results = _do_web_search(query.strip()) if use_web else None

    from text_tools.kb_chat import prepare_kb_context
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

    llm = ChatOpenAI(
        base_url=config.OPENAI_BASE_URL,
        api_key=config.OPENAI_API_KEY,
        model=model or config.TEXT_MODEL,
        temperature=0.7,
        extra_body=config.get_llm_extra_body(thinking),
    )

    if use_kb:
        prompt, msgs = prepare_kb_context(
            query=query.strip(), keyword=keyword or "", k=int(k),
            batch_size=int(batch_size), chat_history=chat_messages,
            web_context=web_results, progress_callback=on_progress,
        )
        if prompt is not None:
            answer = llm.invoke(prompt).content
        else:
            answer = llm.invoke(msgs).content
    elif web_results:
        lc_msgs = [SystemMessage(content=f"以下是联网搜索的最新结果，请基于这些信息回答用户问题：\n\n{web_results}")]
        for m in (chat_messages or []):
            if m["role"] == "user": lc_msgs.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant": lc_msgs.append(AIMessage(content=m["content"]))
        lc_msgs.append(HumanMessage(content=query.strip()))
        answer = llm.invoke(lc_msgs).content
    else:
        lc_msgs = []
        for m in (chat_messages or []):
            if m["role"] == "user": lc_msgs.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant": lc_msgs.append(AIMessage(content=m["content"]))
        lc_msgs.append(HumanMessage(content=query.strip()))
        answer = llm.invoke(lc_msgs).content

    progress(1.0, desc="完成")
    new_messages = list(chat_messages) if chat_messages else []
    new_messages.append({"role": "user", "content": query.strip()})
    new_messages.append({"role": "assistant", "content": answer})
    history.add_entry("知识库问答", query[:50], "查询完成")
    return new_messages, ""


def _kb_create_kb(name):
    """创建新知识库目录"""
    if not name or not name.strip():
        return "❌ 请输入知识库名称", gr.update()
    import re
    safe = re.sub(r'[\\/*?:"<>|\s]+', '_', name.strip())
    if not safe:
        return "❌ 名称无效", gr.update()
    docs_dir = config.DATA_DIR / safe
    docs_dir.mkdir(parents=True, exist_ok=True)
    # 重新获取列表
    from text_tools.kb_manager import list_knowledge_bases
    bases = list_knowledge_bases()
    choices = [Path(b["docs_dir"]).as_posix() for b in bases]
    labels = {Path(b["docs_dir"]).as_posix(): b["name"] for b in bases}
    new_path = Path(docs_dir).as_posix()
    return f"✅ 已创建: {docs_dir}", gr.update(choices=choices, value=new_path)


def _kb_delete_msg(messages, msg_index):
    """删除指定索引的消息"""
    if not messages:
        return messages, messages, "❌ 没有消息"
    idx = int(msg_index)
    if idx < 0 or idx >= len(messages):
        return messages, messages, "❌ 索引无效"
    new_msgs = list(messages)
    new_msgs.pop(idx)
    return new_msgs, new_msgs, f"✅ 已删除第 {idx+1} 条消息"


def _kb_auto_title(provider, msgs):
    """用 AI 根据对话内容生成简短标题（限30字）"""
    if not msgs:
        return ""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    user_qs = [m["content"][:200] for m in msgs if m.get("role") == "user"][:3]
    text = " | ".join(user_qs)
    _apply_provider(provider)
    llm = ChatOpenAI(
        base_url=config.OPENAI_BASE_URL,
        api_key=config.OPENAI_API_KEY,
        model=config.TEXT_MODEL,
        temperature=0.3,
        max_tokens=30,
        extra_body=config.get_llm_extra_body(False),
        request_timeout=30,
    )
    sys = SystemMessage(content="你是一个标题生成助手。根据用户的问题，用5-15个字生成一个简短的对话标题。只回复标题，不要加引号或解释。")
    user = HumanMessage(content=f"生成标题：{text}")
    try:
        resp = llm.invoke([sys, user])
        title = resp.content.strip()[:30]
        import re as _re
        title = _re.sub(r'[\\/*?:"<>|]', '', title)
        return title or ""
    except Exception:
        return ""
# ============================================================
def _kb_get_choices():
    """获取知识库选择列表"""
    from text_tools.kb_manager import list_knowledge_bases
    bases = list_knowledge_bases()
    choices = [Path(b["docs_dir"]).as_posix() for b in bases]
    labels = {Path(b["docs_dir"]).as_posix(): b["name"] for b in bases}
    return bases, choices, labels


def _kb_list_docs(selected_kb=None):
    from text_tools.kb_manager import list_documents, get_docs_dir
    from datetime import datetime
    docs_dir = Path(selected_kb) if selected_kb else None
    docs = list_documents(docs_dir)
    if not docs:
        dir_path = docs_dir or get_docs_dir()
        return f"📂 知识库目录为空，请先上传文档\n\n目录: {dir_path}", ""
    lines = []
    for d in docs:
        size = d['size']
        size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/1024/1024:.1f}MB"
        dt = datetime.fromtimestamp(d['modified']).strftime("%m-%d %H:%M")
        lines.append(f"| {d['name']} | {size_str} | {dt} |")
    table = "| 文件名 | 大小 | 修改时间 |\n|---|---|---|\n" + "\n".join(lines)
    dir_path = docs_dir or get_docs_dir()
    info = f"共 {len(docs)} 个文档，目录: {dir_path}"
    return table, info


def _kb_upload(files, selected_kb=None):
    logger.info(f"[上传文档] 文件数={len(files) if files else 0}")
    if not files:
        return "❌ 请选择要上传的文件"
    from text_tools.kb_manager import upload_documents
    docs_dir = Path(selected_kb) if selected_kb else None
    paths = [f.name if hasattr(f, 'name') else str(f) for f in files]
    results = upload_documents(paths, docs_dir)
    return "\n".join(results)


def _kb_delete(filename, selected_kb=None):
    if not filename or not filename.strip():
        return "❌ 请输入要删除的文件名"
    from text_tools.kb_manager import delete_document
    docs_dir = Path(selected_kb) if selected_kb else None
    return delete_document(filename.strip(), docs_dir)


def _kb_delete_all(selected_kb=None):
    from text_tools.kb_manager import delete_all_documents
    docs_dir = Path(selected_kb) if selected_kb else None
    return delete_all_documents(docs_dir)


def _kb_build_index(chunk_size, chunk_overlap, embedding_model, selected_kb=None, progress=gr.Progress()):
    logger.info(f"[构建索引] 块大小={chunk_size} 重叠={chunk_overlap} 模型={embedding_model}")
    from text_tools.kb_manager import build_index
    from pathlib import Path as _Path
    progress(0.1, desc="准备构建索引...")

    docs_dir = _Path(selected_kb) if selected_kb else None
    lines = []
    def on_progress(msg):
        lines.append(msg)
        if "向量化" in msg:
            progress(0.5, desc="向量化中...")
        elif "保存" in msg:
            progress(0.9, desc="保存索引...")

    result = build_index(
        docs_dir=docs_dir,
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        embedding_model=embedding_model.strip() or None,
        progress_callback=on_progress,
    )
    progress(1.0, desc="完成")
    log_text = "\n".join(f"⏳ {l}" for l in lines)
    return f"{log_text}\n\n{result}"


def _kb_stop_build():
    from text_tools.kb_manager import request_stop
    request_stop()
    return "⏹️ 已请求停止索引构建..."


def _kb_get_stats(selected_kb=None):
    from text_tools.kb_manager import get_index_stats_quick
    from pathlib import Path as _Path
    docs_dir = _Path(selected_kb) if selected_kb else None
    stats = get_index_stats_quick(docs_dir=docs_dir)
    index_status = "✅ 已存在" if stats["exists"] else "❌ 未构建"
    size_mb = stats["index_size"] / (1024 * 1024) if stats["index_size"] else 0
    src_size_kb = stats["source_total_size"] / 1024 if stats["source_total_size"] else 0
    return (
        f"**索引状态:** {index_status}\n"
        f"**索引路径:** {stats['path']}\n"
        f"**索引大小:** {size_mb:.2f} MB\n\n"
        f"**源文档数:** {stats['source_file_count']} 个\n"
        f"**源文档总大小:** {src_size_kb:.1f} KB\n"
        f"**源文档目录:** {stats['docs_dir']}"
    )


# ============================================================
# Tab 7: LLM 压测
# ============================================================
def _benchmark(url, model, api_key, concurrency, timeout, output_tokens, lengths_str, progress=gr.Progress()):
    logger.info(f"[LLM压测] URL={url} 模型={model} 并发={concurrency} 长度={lengths_str}")
    from benchmarks.speedtest import run_benchmark

    if lengths_str.strip():
        lengths = [int(x.strip()) for x in lengths_str.split(",")]
    else:
        lengths = [512, 1024, 2048, 4096]

    save_json = str(config.OUTPUT_DIR / "benchmarks" / "benchmark_results.json")
    save_plot = str(config.OUTPUT_DIR / "benchmarks" / "throughput_chart.png")

    total_steps = len(lengths)

    def _progress_cb(done, total):
        progress(done / total_steps, desc=f"测试中 {done}/{total} 并发")

    buf = io.StringIO()
    with redirect_stdout(buf):
        run_benchmark(
            token_lengths=lengths,
            concurrency=concurrency,
            base_url=url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            output_tokens=output_tokens,
            save_json=save_json,
            save_plot=save_plot,
            progress_callback=_progress_cb,
        )

    text_output = buf.getvalue()
    config.save_state("benchmark", url=url, model=model, api_key=api_key,
                       concurrency=concurrency, timeout=timeout,
                       output_tokens=output_tokens, lengths_str=lengths_str)
    history.add_entry("LLM压测", model or url, "压测完成")
    plot_path = Path(save_plot)
    if plot_path.exists():
        return text_output, str(plot_path)
    return text_output, None


# ============================================================
# 设置页：读写 .env
# ============================================================
ENV_PATH = Path(__file__).parent / ".env"
EXAMPLE_PATH = Path(__file__).parent / ".env.example"

# 需要显示在设置页的配置项及分组
SETTINGS_SCHEMA = [
    ("🔗 API 连接", [
        ("OPENAI_BASE_URL", "API 地址", "http://localhost:1234/v1", "text"),
        ("OPENAI_API_KEY", "API 密钥", "lm-studio", "password"),
    ]),
    ("🤖 模型名称", [
        ("VISION_MODEL", "视觉模型（图片识别、质量检测）", "qwen/qwen3.6-27b", "text"),
        ("VISION_MODEL_THINKING", "视觉模型-思考版（截图 OCR 推荐）", "qwen/qwen3.6-35b-a3b-Thinking", "text"),
        ("RENAME_MODEL", "图片重命名模型", "qwen/Qwen3.6-27b", "text"),
        ("TEXT_MODEL", "文本模型（翻译、压缩、知识库）", "qwen/qwen3.5-9b", "text"),
        ("BENCHMARK_MODEL", "压测模型", "qwen3.6-35b-a3b@iq2_xxs", "text"),
    ]),
    ("⚡ 并发与超时", [
        ("DEFAULT_WORKERS", "默认并行线程数", "2", "int"),
        ("RETRY_TIMES", "失败重试次数", "2", "int"),
        ("REQUEST_TIMEOUT", "请求超时-长任务（秒）", "300", "int"),
        ("REQUEST_TIMEOUT_SHORT", "请求超时-短任务（秒）", "60", "int"),
    ]),
    ("🖼️ 图片处理", [
        ("IMAGE_MAX_SIZE", "输入图片最大边长（像素）", "2048", "int"),
        ("SLICE_HEIGHT", "长截图切片高度（像素）", "2000", "int"),
        ("OVERLAP", "切片重叠高度（像素）", "400", "int"),
        ("TOP_PERCENT", "高质量比例（小数）", "0.05", "float"),
        ("BOTTOM_PERCENT", "低质量比例（小数）", "0.05", "float"),
        ("HIGH_QUALITY_FOLDER", "高质量目录名", "HighQuality", "text"),
        ("LOW_QUALITY_ERRORS_FOLDER", "低质量目录名", "LowQuality_Errors", "text"),
    ]),
    ("📝 文本 / 翻译", [
        ("DEFAULT_CHUNK_SIZE", "聊天分块大小（字符）", "20480", "int"),
        ("OVERLAP_MESSAGES", "块间重叠消息数", "2", "int"),
        ("SOURCE_LANG", "翻译源语言", "Chinese", "text"),
        ("TARGET_LANG", "翻译目标语言", "English", "text"),
    ]),
    ("🔄 更新", [
        ("AUTO_UPDATE", "启动时自动检查更新", "true", "bool"),
    ]),
    ("📂 目录与索引", [
        ("DATA_DIR", "数据输入目录", "data", "text"),
        ("OUTPUT_DIR", "输出目录", "outputs", "text"),
        ("FAISS_INDEX_PATH", "FAISS 索引路径", "faiss_index", "text"),
        ("EMBEDDING_MODEL_PATH", "Embedding 模型路径（留空=在线下载）", "", "text"),
        ("KB_CHUNK_SIZE", "知识库文本块大小（字符）", "500", "int"),
        ("KB_CHUNK_OVERLAP", "知识库块间重叠（字符）", "50", "int"),
    ]),
]


def _read_env() -> dict:
    """读取 .env 文件返回 {KEY: value}"""
    if not ENV_PATH.exists():
        return {}
    import re
    result = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(\w+)\s*=\s*(.*)$", line)
        if m:
            result[m.group(1)] = m.group(2).strip()
    return result


def _save_env(updates: dict) -> str:
    """将 updates 中的键值写入 .env 文件，保留原有注释和其他行"""
    logger.info(f"[保存设置] {list(updates.keys())}")
    try:
        import re
        if ENV_PATH.exists():
            lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
        elif EXAMPLE_PATH.exists():
            lines = EXAMPLE_PATH.read_text(encoding="utf-8").splitlines()
        else:
            lines = []

        updated_keys = set()
        new_lines = []
        for line in lines:
            m = re.match(r"^(\w+)\s*=\s*(.*)$", line.strip())
            if m:
                key = m.group(1)
                if key in updates:
                    new_lines.append(f"{key}={updates[key]}")
                    updated_keys.add(key)
                    continue
            new_lines.append(line)

        # 追加新增的 key
        for key, val in updates.items():
            if key not in updated_keys:
                new_lines.append(f"{key}={val}")

        ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return "✅ 设置已保存。部分配置需要重启应用才能生效。"
    except Exception as e:
        return f"❌ 保存失败: {e}"


def _get_current_version():
    """获取当前版本号"""
    try:
        r = subprocess.run(
            ["git", "describe", "--tags", "--always"],
            cwd=ROOT_DIR, capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"

def _check_update():
    """检查远程是否有更新，返回 (has_update, local, remote)"""
    try:
        subprocess.run(
            ["git", "fetch", "origin"],
            cwd=ROOT_DIR, capture_output=True, text=True, timeout=30
        )
        local = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        remote = subprocess.run(
            ["git", "rev-parse", "origin/main"],
            cwd=ROOT_DIR, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        return local != remote, local[:8], remote[:8]
    except Exception as e:
        return None, None, None

def _do_update():
    """执行 git pull 更新，返回结果文本"""
    try:
        ver_before = _get_current_version()
        r = subprocess.run(
            ["git", "pull", "origin", "main"],
            cwd=ROOT_DIR, capture_output=True, text=True, timeout=60
        )
        if r.returncode != 0:
            return f"❌ 更新失败:\n{r.stderr}"
        ver_after = _get_current_version()
        if r.stdout.strip() == "Already up to date.":
            return f"✅ 已是最新版本 ({ver_after})"
        return f"✅ 更新成功!\n\n{ver_before} → {ver_after}\n\n```\n{r.stdout}\n```\n\n⚠️ 部分功能需重启应用才能生效。"
    except Exception as e:
        return f"❌ 更新出错: {e}"

def _check_and_update_on_startup():
    """启动时自动检查更新（如果开启了 AUTO_UPDATE）"""
    if not config.AUTO_UPDATE:
        return None
    has_update, local, remote = _check_update()
    if has_update:
        return f"📢 发现新版本! 本地: {local} → 远程: {remote}，正在自动更新...\n{_do_update()}"
    elif has_update is False:
        return f"✅ 当前已是最新版本 ({local})"
    return "⚠️ 无法检查更新（网络异常或非 git 环境）"

def _test_api_connection(base_url, api_key, model):
    """测试 API 连接是否正常"""
    logger.info(f"[测试连接] URL={base_url} 模型={model}")
    if not base_url.strip():
        return "❌ 请填写 API 地址"
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url.strip(), api_key=api_key.strip() or "no-key")
        models = client.models.list()
        model_names = [m.id for m in models]
        snippet = ", ".join(model_names[:8])
        lines = [f"✅ 连接成功！找到 {len(model_names)} 个模型"]
        if model_names:
            lines.append(f"可用模型示例: {snippet}{'...' if len(model_names) > 8 else ''}")
        if model.strip() and model.strip() not in model_names:
            lines.append(f"⚠️ 注意: 当前配置的模型「{model.strip()}」不在可用列表中")
        return "\n".join(lines)
    except Exception as e:
        msg = str(e)
        if "Connection" in msg or "refused" in msg.lower():
            return f"❌ 无法连接到 {base_url}\n请确认：\n1. LM Studio 或其他 API 服务是否已启动\n2. 地址和端口是否正确"
        return f"❌ 连接失败: {msg}"


def _fetch_models(base_url, api_key):
    """从 API 获取可用模型列表，按类型分类"""
    if not base_url or not base_url.strip():
        return {}, "❌ 请填写 API 地址"
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url.strip(), api_key=(api_key or "").strip() or "no-key")
        models = client.models.list()
        model_ids = sorted([m.id for m in models])
    except Exception as e:
        return {}, f"❌ 获取失败: {e}"

    if not model_ids:
        return {}, "⚠️ 未找到任何模型"

    # 分类
    vlm_kw = ['vision', 'vlm', 'visual', 'gpt-4o', 'qwen-vl', 'qwen2-vl', 'internvl', 'minicpm-v', 'cogvlm', 'llava', 'deepseek-vl']
    embed_kw = ['embed', 'bge', 'e5-', 'gte-', 'text-embedding', 'cohere']
    chat_kw = ['chat', 'gpt', 'qwen', 'deepseek', 'llama', 'mistral', 'gemma', 'glm', 'yi-', 'yi ', 'internlm', 'phi', 'baichuan', 'moonshot', 'kimi', 'doubao', 'spark', 'ernie', 'claude', 'gemini']

    def _classify(mid):
        low = mid.lower()
        if any(k in low for k in embed_kw):
            return 'embed'
        if any(k in low for k in vlm_kw):
            return 'vlm'
        if any(k in low for k in chat_kw):
            return 'chat'
        return 'other'

    categorized = {'全部模型': [], 'chat 模型': [], 'vlm 视觉模型': [], 'embed 模型': [], '其他': []}
    for mid in model_ids:
        categorized['全部模型'].append(mid)
        cat = _classify(mid)
        if cat == 'embed':
            categorized['embed 模型'].append(mid)
        elif cat == 'vlm':
            categorized['vlm 视觉模型'].append(mid)
        elif cat == 'chat':
            categorized['chat 模型'].append(mid)
        else:
            categorized['其他'].append(mid)

    # 移除空分类
    categorized = {k: v for k, v in categorized.items() if v}
    msg = f"✅ 获取到 {len(model_ids)} 个模型"
    return categorized, msg


def _make_model_selector(label, default_value, info="", show_thinking=True):
    """创建模型选择器组件（类型筛选 + 下拉选择 + 重新获取按钮 + 思考模式开关）"""
    with gr.Row(equal_height=True):
        model_type = gr.Dropdown(
            choices=["全部模型", "chat 模型", "vlm 视觉模型", "embed 模型", "其他"],
            value="全部模型",
            label="分类",
            scale=1,
            container=False,
        )
        model_select = gr.Dropdown(
            choices=[default_value] if default_value else [],
            value=default_value,
            label=label,
            info=info,
            filterable=True,
            allow_custom_value=True,
            scale=3,
            container=False,
        )
        fetch_btn = gr.Button("🔄", scale=0, variant="secondary", min_width=40)
    thinking_toggle = gr.Checkbox(
        label="思考模式（Thinking）",
        value=True,
        visible=show_thinking,
        info="开启后模型会先推理再回答，关闭可加快速度",
    )
    fetch_status = gr.Textbox(visible=False, max_lines=1)
    return model_type, model_select, fetch_btn, fetch_status, thinking_toggle


def _bind_model_fetch(fetch_btn, model_type, model_select, fetch_status, provider_info, default_value):
    """绑定模型获取和筛选事件。provider_info 是 gr.State，存储当前供应商的 {base_url, api_key}。"""
    _models_cache = gr.State({})

    def _do_fetch(prov):
        url = prov.get("base_url", "") if isinstance(prov, dict) else ""
        key = prov.get("api_key", "") if isinstance(prov, dict) else ""
        categorized, msg = _fetch_models(url, key)
        all_models = categorized.get("全部模型", [])
        return (
            gr.update(choices=list(categorized.keys()), value="全部模型"),
            gr.update(choices=all_models, value=default_value if default_value in all_models else (all_models[0] if all_models else "")),
            msg,
            categorized,
        )

    fetch_btn.click(
        _do_fetch,
        inputs=[provider_info],
        outputs=[model_type, model_select, fetch_status, _models_cache],
    )

    def _filter_models(type_val, cache):
        if not cache:
            return gr.update()
        if type_val in cache:
            return gr.update(choices=cache[type_val])
        return gr.update(choices=cache.get("全部模型", []))

    model_type.change(
        _filter_models,
        inputs=[model_type, _models_cache],
        outputs=[model_select],
    )

    return _do_fetch  # 返回函数供供应商切换时调用


def _bind_model_fetch_local(fetch_btn, model_type, model_select, fetch_status, url_component, key_component, default_value):
    """绑定模型获取事件（使用 Tab 自有的 URL/Key 输入组件）。"""
    _models_cache = gr.State({})

    def _do_fetch(url, key):
        categorized, msg = _fetch_models(url, key)
        all_models = categorized.get("全部模型", [])
        return (
            gr.update(choices=list(categorized.keys()), value="全部模型"),
            gr.update(choices=all_models, value=default_value if default_value in all_models else (all_models[0] if all_models else "")),
            msg,
            categorized,
        )

    fetch_btn.click(
        _do_fetch,
        inputs=[url_component, key_component],
        outputs=[model_type, model_select, fetch_status, _models_cache],
    )

    def _filter_models(type_val, cache):
        if not cache:
            return gr.update()
        if type_val in cache:
            return gr.update(choices=cache[type_val])
        return gr.update(choices=cache.get("全部模型", []))

    model_type.change(
        _filter_models,
        inputs=[model_type, _models_cache],
        outputs=[model_select],
    )


# ============================================================
# 构建 Gradio 界面
# ============================================================
CSS = """
.output-text { font-size: 0.9em; max-height: 500px; overflow-y: auto; }
.hint { font-size: 0.85em; color: #888; margin-top: -8px; margin-bottom: 12px; }
footer { visibility: hidden; }
.top-bar { align-items: center !important; gap: 8px; background: #fff; padding: 8px 12px; border-bottom: 1px solid #e5e5e5; margin-bottom: 8px; }
.top-bar h1 { margin: 0 !important; font-size: clamp(0.9em, 1.5vw, 1.2em) !important; white-space: nowrap; }
.top-bar .gr-dropdown { min-width: 100px; }
/* 聊天页面 - 侧边栏 */
.kb-sidebar { background: #f8f9fa; border-right: 1px solid #e0e0e0; padding: 12px 10px !important; min-width: 200px; max-width: 280px; height: 100%; }
.kb-sidebar > .wrap { display: flex !important; flex-direction: column !important; height: 100% !important; }
.kb-sidebar .gr-button { width: 100%; margin-bottom: 4px; flex-shrink: 0; }
#kb_chat_list { flex: 1 1 auto; overflow-y: auto; font-size: 0.85em; min-height: 400px; margin: 8px 0; }
#kb_chat_list .wrap { gap: 2px; flex-direction: column !important; }
#kb_chat_list label { padding: 8px 10px; border-radius: 6px; cursor: pointer; display: block !important; width: 100% !important; margin-bottom: 2px; }
#kb_chat_list label:hover { background: #e8e8e8; }
#kb_chat_list [type="radio"]:checked + label { background: #d0e4ff; font-weight: 600; }
/* 聊天页面 - 主区域 */
.kb-chat-fill { min-height: 500px; }
.kb-chat-fill > .wrap { display: flex !important; flex-direction: column !important; }
#kb_chatbot { flex: 1 1 auto; min-height: 400px; }
#kb_chatbot > .wrap { height: 100% !important; }
/* 聊天页面 - 设置工具栏 */
.kb-toolbar { gap: 8px !important; padding: 4px 0 8px 0 !important; flex-wrap: wrap !important; }
.kb-toolbar .gr-checkbox { margin-right: 4px; }
/* 聊天页面 - 输入区 */
.kb-input-row { border: 1px solid #d0d0d0; border-radius: 10px; padding: 8px 12px 6px 12px !important; background: #fff; margin-top: 8px; }
.kb-input-row:focus-within { border-color: #1976d2; box-shadow: 0 0 0 1px rgba(25,118,210,0.2); }
.kb-disclaimer { text-align: center; margin-top: 4px; flex-shrink: 0 !important; }
.kb-disclaimer p { font-size: 0.75em !important; color: #999 !important; }
"""

JS_ONLOAD = """
function resizeKbChat() {
    var el = document.getElementById('kb_chatbot');
    if (!el) return;
    var fill = el.closest('.kb-chat-fill');
    if (!fill) return;
    var toolbar = fill.querySelector('.kb-toolbar');
    var inputRow = fill.querySelector('.kb-input-row');
    var disc = fill.querySelector('.kb-disclaimer');
    var usedH = (toolbar ? toolbar.offsetHeight : 40) + (inputRow ? inputRow.offsetHeight : 80) + (disc ? disc.offsetHeight : 20);
    var fillH = fill.getBoundingClientRect().height;
    el.style.height = Math.max(300, fillH - usedH - 16) + 'px';
}
setTimeout(resizeKbChat, 400);
window.addEventListener('resize', resizeKbChat);
"""

def build_ui():
    s = {k: config.load_state(k) for k in ["rename","score","ocr","compress","translate","benchmark"]}

    # 加载供应商列表
    _prov_list, _prov_active = config.load_providers()
    _active_prov = config.get_active_provider()

    with gr.Blocks(title="LocalAITools") as app:
        # ---- 顶部栏：标题 + 供应商 + 操作按钮 ----
        with gr.Row(equal_height=True, elem_classes="top-bar"):
            gr.Markdown("# LocalAITools - 本地 AI 工具箱")
            provider_select = gr.Dropdown(
                choices=[p["name"] for p in _prov_list],
                value=_prov_active,
                label="供应商",
                container=False,
                scale=2,
            )
            prov_add_btn = gr.Button("添加", size="sm", scale=0, min_width=50)
            prov_edit_btn = gr.Button("编辑", size="sm", scale=0, min_width=50)
            prov_del_btn = gr.Button("删除", size="sm", scale=0, min_width=50)
            restart_btn_top = gr.Button("🔄", size="sm", scale=0, min_width=40)
        prov_info_text = gr.Markdown(
            f"📡 `{_active_prov['base_url']}`",
            elem_classes="hint",
        )
        restart_msg = gr.Textbox("", interactive=False, container=False, show_label=False, visible=False)

        def _on_restart():
            print("\n🔄 用户请求重启...")
            import threading, subprocess
            def _do_restart():
                import time
                time.sleep(1.5)
                script = str(Path(__file__).resolve())
                subprocess.Popen([sys.executable, script] + sys.argv[1:])
                os._exit(0)
            threading.Thread(target=_do_restart, daemon=True).start()
            return gr.update(value="🔄 正在重启...", visible=True)

        restart_btn_top.click(_on_restart, outputs=[restart_msg])

        # 供应商编辑区域（默认隐藏）
        with gr.Accordion("添加/编辑供应商", open=False, visible=True) as prov_editor:
            with gr.Row():
                prov_name = gr.Textbox(label="名称", placeholder="如：硅基流动", scale=1)
                prov_url = gr.Textbox(label="API Base URL", placeholder="https://api.siliconflow.cn/v1", scale=2)
                prov_key = gr.Textbox(label="API Key", type="password", scale=2)
            with gr.Row():
                prov_save_btn = gr.Button("💾 保存", variant="primary", scale=1)
                prov_cancel_btn = gr.Button("取消", variant="secondary", scale=1)
            prov_msg = gr.Markdown("")

        # 全局状态：当前供应商信息
        provider_info = gr.State({"base_url": _active_prov["base_url"], "api_key": _active_prov["api_key"]})
        # 供应商列表状态
        providers_state = gr.State({"list": _prov_list, "active": _prov_active})
        # 编辑模式状态：None 或 "add" 或 "edit"
        prov_edit_mode = gr.State(None)

        # ---- 供应商操作函数 ----
        def _refresh_provider_select(providers):
            """刷新供应商下拉框选项"""
            names = [p["name"] for p in providers["list"]]
            return gr.update(choices=names, value=providers["active"])

        def _on_provider_change(name, providers):
            """切换供应商"""
            if not name:
                return gr.update(), {}, providers
            for p in providers["list"]:
                if p["name"] == name:
                    providers["active"] = name
                    config.save_providers(providers["list"], name)
                    info_md = f"📡 `{p['base_url']}`"
                    return info_md, {"base_url": p["base_url"], "api_key": p["api_key"]}, providers
            return gr.update(), {}, providers

        def _on_prov_add(providers):
            """打开添加面板"""
            return gr.update(open=True), "", "", "", "add", ""

        def _on_prov_edit(providers):
            """打开编辑面板"""
            name = providers.get("active", "")
            for p in providers["list"]:
                if p["name"] == name:
                    return gr.update(open=True), p["name"], p["base_url"], p["api_key"], "edit", ""
            return gr.update(open=True), "", "", "", "edit", ""

        def _on_prov_save(mode, old_name, name, url, key, providers):
            """保存供应商"""
            if not name.strip():
                return "❌ 名称不能为空", providers
            if not url.strip():
                return "❌ URL 不能为空", providers

            if mode == "add":
                if any(p["name"] == name.strip() for p in providers["list"]):
                    return f"❌ 已存在同名供应商「{name.strip()}」", providers
                providers["list"].append({"name": name.strip(), "base_url": url.strip(), "api_key": key.strip()})
                providers["active"] = name.strip()
            elif mode == "edit":
                for p in providers["list"]:
                    if p["name"] == old_name:
                        p["name"] = name.strip()
                        p["base_url"] = url.strip()
                        p["api_key"] = key.strip()
                        break
                if providers["active"] == old_name:
                    providers["active"] = name.strip()

            config.save_providers(providers["list"], providers["active"])
            return f"✅ 已保存供应商「{name.strip()}」", providers

        def _on_prov_delete(providers):
            """删除当前供应商"""
            name = providers.get("active", "")
            if len(providers["list"]) <= 1:
                return "❌ 至少保留一个供应商", providers, gr.update(), gr.update()
            providers["list"] = [p for p in providers["list"] if p["name"] != name]
            providers["active"] = providers["list"][0]["name"]
            config.save_providers(providers["list"], providers["active"])
            new_active = providers["list"][0]
            names = [p["name"] for p in providers["list"]]
            info_md = f"📡 `{new_active['base_url']}`"
            return f"✅ 已删除供应商「{name}」", providers, gr.update(choices=names, value=providers["active"]), info_md

        # ---- 绑定供应商事件 ----
        prov_add_btn.click(
            _on_prov_add,
            inputs=[providers_state],
            outputs=[prov_editor, prov_name, prov_url, prov_key, prov_edit_mode, prov_msg],
        )
        prov_edit_btn.click(
            _on_prov_edit,
            inputs=[providers_state],
            outputs=[prov_editor, prov_name, prov_url, prov_key, prov_edit_mode, prov_msg],
        )
        def _on_prov_save_and_refresh(mode, old_name, name, url, key, providers):
            """保存供应商并返回所有需要更新的状态"""
            msg, providers = _on_prov_save(mode, old_name, name, url, key, providers)
            prov_names = [p["name"] for p in providers["list"]]
            active_prov = None
            for p in providers["list"]:
                if p["name"] == providers["active"]:
                    active_prov = p
                    break
            info_md = f"📡 `{active_prov['base_url']}`" if active_prov else ""
            prov_info = {"base_url": active_prov["base_url"], "api_key": active_prov["api_key"]} if active_prov else {}
            return msg, providers, gr.update(choices=prov_names, value=providers["active"]), info_md, prov_info

        def _on_prov_delete_and_refresh(providers):
            """删除供应商并返回所有需要更新的状态"""
            msg, providers, new_select, info_md = _on_prov_delete(providers)
            active_prov = None
            for p in providers["list"]:
                if p["name"] == providers["active"]:
                    active_prov = p
                    break
            prov_info = {"base_url": active_prov["base_url"], "api_key": active_prov["api_key"]} if active_prov else {}
            return msg, providers, new_select, info_md, prov_info

        # ==================== Tab 0: 开始使用（新手引导） ====================
        with gr.Tab("🏠 开始使用"):
            gr.Markdown(_make_title("欢迎使用 LocalAITools！"))
            gr.Markdown("一套调用本地大模型的 AI 工具集，**所有功能免费、数据不上传云端**。")

            gr.Markdown("---")

            gr.Markdown("### 🔌 第一步：获取 AI 模型服务")

            gr.Markdown("""本工具需要连接一个 AI 模型服务才能工作。推荐以下方式（任选一种）：

**⭐ 方式一：云端 API（最简单，免下载模型，按量付费）**
1. [硅基流动 SiliconFlow](https://cloud.siliconflow.cn/) — 注册送额度，支持 Qwen 系列
2. [DeepSeek 开放平台](https://platform.deepseek.com/) — 便宜好用
3. [阿里云百炼](https://bailian.console.aliyun.com/) — Qwen 官方 API
4. 注册后在后台创建 API Key，填到下方设置页的 API 地址和密钥即可

**⭐⭐ 方式二：LM Studio（推荐本地运行，免费）**
1. 下载安装 [LM Studio](https://lmstudio.ai/)（支持 Windows / Mac）
2. 打开 LM Studio，在搜索框搜 `qwen3` 或 `qwen3.6`
3. 下载一个视觉模型（如 `qwen3.6-27b`，约 16 GB）
4. 切换到 **Local Server** 标签页，点击 **Start Server**
5. 默认地址就是 `http://localhost:1234/v1`，无需修改

**⭐⭐⭐ 方式三：Ollama（免费，需命令行基础）**
```bash
ollama serve          # 启动服务
ollama pull qwen3     # 下载模型
```
默认地址 `http://localhost:11434/v1`

**⭐⭐⭐ 方式四：vLLM / SGLang / Xinference（自建推理服务，适合多人共享）**
- [vLLM](https://github.com/vllm-project/vllm) — 生产级推理引擎，支持 PagedAttention，吞吐量高
- [SGLang](https://github.com/sgl-project/sglang) — 高效推理框架，结构化生成能力强
- [Xinference](https://github.com/xorbitsai/inference) — 一键部署，支持 Web UI 管理模型
- 部署后 OpenAI 兼容端点填到设置页即可使用

> 💡 **简单总结：** 不想折腾 → 云端 API；有显卡想省钱 → LM Studio；技术党想折腾 → Ollama/vLLM
""")

            gr.Markdown("---")

            gr.Markdown("### ⚙️ 第二步：填写配置")
            gr.Markdown("""切换到 **⚙️ 设置** 标签页，填写你的 API 信息：

- **API 地址**：本地服务默认 `http://localhost:1234/v1`（LM Studio），云端 API 填对应地址
- **API 密钥**：本地服务填任意值（如 `lm-studio`），云端 API 填真实的 Key
- 填好后点 **💾 保存设置**，然后点 **🔗 测试连接** 确认能连上

> 💡 云端 API 需填写平台提供的地址和 Key；本地服务一般无需修改地址。""")

            gr.Markdown("---")

            gr.Markdown("### 🎯 第三步：开始使用")
            gr.Markdown("""配置完成后，切换到对应的功能标签页即可使用：

| Tab | 功能 | 需要什么模型 | 输入 → 输出 |
|-----|------|-------------|-------------|
| 🖼️ 图片重命名 | AI 看图起中文名 | 视觉模型 | 图片文件夹 → 文件重命名 |
| 🔍 质量评分分类 | 评分 + 自动分拣 | 视觉模型 | 图片文件夹 → HighQuality / LowQuality_Errors |
| 💬 截图识别 | 聊天截图 → 文字 | 视觉模型（Thinking 最佳） | 截图 → TXT 文件 |
| 📝 聊天压缩 | 合并冗余时间戳 | 文本模型 | TXT → 精简 TXT |
| 🌐 文本翻译 | 长篇章节翻译 | 文本模型 | TXT → 翻译 TXT |
| 📚 知识库问答 | RAG 混合检索 | 文本 + Embedding | 问题 → 答案 |
| ⚡ LLM 压测 | API 吞吐量测试 | 被测模型 | 参数 → 图表 |

> 💡 每个标签页顶部都有「📖 使用方法」折叠面板，点开即可查看详细步骤。
""")

            gr.Markdown("---")
            gr.Markdown('<div style="text-align:center;color:#888;font-size:0.85em">'
                        '遇到问题？<a href="https://github.com/go-farther-and-farther/LocalAITools/issues" target="_blank">GitHub Issues</a>'
                        '</div>')

            # ---- 历史记录 ----
            gr.Markdown("---")
            with gr.Accordion("📋 处理历史", open=False):
                history_md = gr.Markdown("")

            def _load_history():
                entries = history.get_recent(20)
                if not entries:
                    return "暂无处理记录。\n\n每次使用工具处理文件后，这里会自动记录。"
                lines = ["| 时间 | 工具 | 输入 | 摘要 |",
                         "|------|------|------|------|"]
                for e in entries:
                    inp = str(e["input"])[:40]
                    lines.append(f"| {e['time']} | {e['tool']} | {inp} | {e['summary']} |")
                return "\n".join(lines)

            # 每次切换到该 Tab 时刷新历史
            app.load(_load_history, outputs=[history_md])

        # ==================== Tab 1: 图片工具（重命名 + 评分分类 + 手动审核）====================
        with gr.Tab("🖼️ 图片工具"):
            with gr.Tabs():
                # ---- 子 Tab: 图片重命名 ----
                with gr.Tab("🏷️ 图片重命名"):
                    gr.Markdown(_make_title("图片 AI 重命名 — 用中文短句替代杂乱的文件名"))
                    with gr.Accordion("📖 使用方法", open=False):
                        gr.Markdown("**步骤：** ① 把图片放进文件夹 → ② 点「开始重命名」→ ③ 文件名变成中文描述短语\n\n"
                                    "> 💡 建议先勾选「试运行」预览效果，满意后再取消勾选正式改名。")
                    with gr.Row():
                        with gr.Column(scale=2):
                            rn_input = gr.Textbox(label="图片文件夹",
                                                  value=s["rename"].get("input_dir", str(config.DATA_DIR / "images")),
                                                  placeholder="粘贴图片所在文件夹的完整路径",
                                                  info="支持 .jpg / .jpeg / .png / .webp / .avif / .gif")
                            with gr.Accordion("⚙️ 高级设置", open=False):
                                rn_model_type, rn_model, rn_fetch_btn, rn_fetch_st, rn_thinking = _make_model_selector(
                                    "视觉模型", s["rename"].get("model", config.RENAME_MODEL),
                                    "需要视觉模型，能理解图片内容")
                                rn_workers = gr.Slider(1, 8, value=s["rename"].get("workers", config.DEFAULT_WORKERS), step=1,
                                                       label="并行线程数", info="越大越快，但可能触发 API 限流")
                            rn_mode = gr.Dropdown(
                                label="命名模式",
                                choices=[
                                    ("通用描述", "general"),
                                    ("人像聚焦", "portrait"),
                                    ("风景聚焦", "landscape"),
                                    ("截图识别", "screenshot"),
                                    ("美食聚焦", "food"),
                                    ("动漫二次元", "anime"),
                                ],
                                value=s["rename"].get("mode", "general"),
                                info="不同模式侧重不同描述风格"
                            )
                            rn_ctx = gr.Slider(0, 10, value=s["rename"].get("context_count", 3), step=1,
                                               label="上下文数量",
                                               info="参考前几张图的描述作为上下文。0=不参考，适合高级模型独立判断")
                            rn_maxsz = gr.Slider(512, 4096, value=config.IMAGE_MAX_SIZE, step=256,
                                                 label="图片最大边长（px）",
                                                 info="超过此值的图片会等比缩小，越小速度越快但精度可能下降")
                            rn_dry = gr.Checkbox(label="试运行（只预览不实际改名）", value=s["rename"].get("dry_run", False),
                                                 info="强烈建议第一次使用时先试运行，看看效果")
                            rn_keep = gr.Checkbox(label="保留原文件名", value=s["rename"].get("keep_original", False),
                                                  info="在描述后面附加原始文件名，适合文件名含时间戳等有用信息的情况")
                            with gr.Row():
                                rn_btn = gr.Button("开始重命名", variant="primary")
                                rn_stop = gr.Button("停止", variant="stop")
                        with gr.Column(scale=3):
                            rn_output = gr.Textbox(label="处理结果", lines=15, elem_classes="output-text",
                                                   placeholder="处理完成后这里会显示每张图片的新名字...")
                    rn_btn.click(_rename_images, [rn_input, rn_model, rn_workers, rn_dry, rn_keep, rn_mode, rn_ctx, rn_maxsz, provider_info, rn_thinking], [rn_output])
                    def _stop_rename():
                        from image_tools.rename_images import request_stop
                        request_stop()
                        return "⏹️ 已请求停止..."
                    rn_stop.click(_stop_rename, outputs=[rn_output])
                    _bind_model_fetch(rn_fetch_btn, rn_model_type, rn_model, rn_fetch_st,
                                     provider_info, config.RENAME_MODEL)

                    # ---- 按作品分类 ----
                    gr.Markdown("---")
                    gr.Markdown("### 📁 按《》作品名自动分类")
                    gr.Markdown("将已重命名的图片按文件名中的《作品名》自动归入子文件夹。")
                    with gr.Row():
                        rn_cls_dry = gr.Checkbox(label="试运行", value=False,
                                                 info="先预览分类结果，不实际移动")
                        rn_cls_min = gr.Slider(1, 20, value=3, step=1,
                                               label="最少图片数",
                                               info="该作品图片少于此数量则不移动")
                        rn_cls_btn = gr.Button("开始分类", variant="secondary")
                    rn_cls_output = gr.Textbox(label="分类结果", lines=8, elem_classes="output-text",
                                               placeholder="点击后显示分类结果...")

                    rn_cls_btn.click(_classify_by_work, [rn_input, rn_cls_dry, rn_cls_min], [rn_cls_output])

                # ---- 子 Tab: 图片质量评分与分类 ----
                with gr.Tab("🔍 评分与分类"):
                    gr.Markdown(_make_title("AI 评分 + 自动分拣 — 多维度评估，高分/低分图片自动归类"))
                    with gr.Accordion("📖 使用方法", open=False):
                        gr.Markdown("**评分：** ① 选择检测模式 → ② 填文件夹路径 → ③ 点「开始评分」→ 每张图生成 `.txt` 评分文件\n\n"
                                    "**分类：** ④ 评分完成后点「开始分类」→ 优质图片移至 `HighQuality/`，劣质/错误图片移至 `LowQuality_Errors/`\n\n"
                                    "**七种检测模式：**\n"
                                    "- 🎨 **AI 图片错误检测** — 检测 AI 生成图的肢体错乱、面部畸形、结构崩坏等问题\n"
                                    "- 📸 **漫展摄影筛选** — 检测跑焦模糊、过曝欠曝等拍摄问题，筛选可出片的 Cosplay 照片\n"
                                    "- 🖼️ **通用照片质量** — 综合评估清晰度、曝光、构图、内容趣味性\n"
                                    "- 👤 **人像摄影评估** — 侧重面部清晰度、肤色、表情、虚化氛围\n"
                                    "- 🌄 **风景摄影评估** — 侧重光影层次、构图法则、色彩氛围\n"
                                    "- 📄 **文档扫描清晰度** — 评估文字可读性、光照均匀度、畸变、完整度\n"
                                    "- 🖌️ **绘画插图质量** — 评估造型比例、线条笔触、色彩光影、完成度\n\n"
                                    "> 💡 评分和分类是独立的两步，评完分可以先看分数再决定如何分类")

                    # ---- 评分区域 ----
                    gr.Markdown("### ① 评分")
                    with gr.Row():
                        with gr.Column(scale=2):
                            de_mode = gr.Dropdown(
                                label="检测模式",
                                choices=[
                                    ("🎨 AI 图片错误检测", "ai"),
                                    ("📸 漫展摄影筛选", "photo"),
                                    ("🖼️ 通用照片质量", "general"),
                                    ("👤 人像摄影评估", "portrait"),
                                    ("🌄 风景摄影评估", "landscape"),
                                    ("📄 文档扫描清晰度", "document"),
                                    ("🖌️ 绘画插图质量", "art"),
                                ],
                                value=s["score"].get("mode", "ai"),
                                info="AI错误：找肢体畸形/崩坏 | 摄影：找跑焦/过曝 | 通用：综合评估 | 人像/风景/文档/绘画各有侧重"
                            )
                            de_input = gr.Textbox(label="图片文件夹",
                                                  value=s["score"].get("input_dir", str(config.DATA_DIR / "images")),
                                                  placeholder="粘贴图片所在文件夹的完整路径",
                                                  info="评分结果保存为同名 .txt 文件")
                            de_model_type, de_model, de_fetch_btn, de_fetch_st, de_thinking = _make_model_selector(
                                "视觉模型", s["score"].get("model", config.VISION_MODEL),
                                "需要视觉模型。留空使用设置页默认值")
                            de_maxsz = gr.Slider(512, 4096, value=config.IMAGE_MAX_SIZE, step=256,
                                                 label="图片最大边长（px）",
                                                 info="超过此值的图片会等比缩小，越小速度越快但精度可能下降")
                            de_prompt = gr.Textbox(
                                label="自定义评分提示词（留空使用默认）",
                                value=s["score"].get("custom_prompt", ""),
                                lines=6,
                                placeholder="留空则使用当前模式的默认提示词。\n切换检测模式后请清空此框或重新填写对应模式的提示词。",
                                info="留空 = 使用默认提示词"
                            )
                            with gr.Row():
                                de_score_btn = gr.Button("开始评分", variant="primary")
                                de_score_stop = gr.Button("停止", variant="stop")
                        with gr.Column(scale=3):
                            de_score_output = gr.Textbox(label="评分日志", lines=15, elem_classes="output-text",
                                                         placeholder="评分完成后这里会显示每张图的分数和理由...")

                    # ---- 分类区域 ----
                    gr.Markdown("### ② 分类（根据已有评分结果）")
                    with gr.Row():
                        with gr.Column(scale=2):
                            de_cls_method = gr.Radio(
                                label="分类方式",
                                choices=[("按比例（前/后 N%）", "percent"), ("按分值（≥N 分 或 <M 分）", "threshold")],
                                value=s["score"].get("classify_method", "percent"),
                                info="按比例：分数前N%入高分、后N%入低分 | 按分值：设定分数线"
                            )
                            with gr.Column():
                                with gr.Row(visible=True) as de_percent_row:
                                    de_top = gr.Slider(1, 50, value=s["score"].get("top_percent", int(config.TOP_PERCENT * 100)), step=1,
                                                       label="高分比例（%）",
                                                       info="评分前 N% 的图片移入 HighQuality")
                                    de_bottom = gr.Slider(1, 50, value=s["score"].get("bottom_percent", int(config.BOTTOM_PERCENT * 100)), step=1,
                                                          label="低分比例（%）",
                                                          info="评分后 N% + 所有 ERR 图片移入 LowQuality_Errors")
                                with gr.Row(visible=False) as de_threshold_row:
                                    de_min = gr.Slider(0.0, 10.0, value=s["score"].get("min_score", 7.0), step=0.1,
                                                       label="高分线（≥）",
                                                       info="分数 ≥ 此值的图片移入 HighQuality")
                                    de_max = gr.Slider(0.0, 10.0, value=s["score"].get("max_score", 4.0), step=0.1,
                                                       label="低分线（<）",
                                                       info="分数 < 此值的图片 + ERR 移入 LowQuality_Errors")
                            with gr.Row():
                                de_cls_btn = gr.Button("开始分类", variant="primary")
                        with gr.Column(scale=3):
                            de_cls_output = gr.Textbox(label="分类日志", lines=12, elem_classes="output-text",
                                                       placeholder="分类完成后这里会显示移动结果...")

                    # 显示/隐藏分类方式
                    def _toggle_cls_method(method):
                        if method == "percent":
                            return gr.update(visible=True), gr.update(visible=False)
                        else:
                            return gr.update(visible=False), gr.update(visible=True)

                    de_cls_method.change(_toggle_cls_method, [de_cls_method], [de_percent_row, de_threshold_row])

                    def _stop_scoring():
                        from image_tools.detect_ai_errors import request_stop
                        request_stop()
                        return "⏹️ 已请求停止..."

                    de_score_event = de_score_btn.click(_score_images, [de_input, de_mode, de_prompt, de_model, de_maxsz, provider_info, de_thinking], [de_score_output])
                    de_score_stop.click(_stop_scoring, outputs=[de_score_output])
                    de_cls_btn.click(_classify_images, [de_input, de_cls_method, de_top, de_bottom, de_min, de_max], [de_cls_output])
                    _bind_model_fetch(de_fetch_btn, de_model_type, de_model, de_fetch_st,
                                     provider_info, config.VISION_MODEL)

                    # ---- 手动审核区域 ----
                    gr.Markdown("### ③ 手动审核（逐张查看评分并手动分类）")
                    gr.Markdown("加载评分结果后，逐张查看缩略图和点评，手动决定每张图片的分类。")

                    review_state = gr.State({"queue": [], "index": 0, "last_moved": None})

                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Row():
                                review_input = gr.Textbox(
                                    label="图片文件夹（已完成评分的）",
                                    value=s["score"].get("input_dir", str(config.DATA_DIR / "images")),
                                    placeholder="与上方评分使用同一目录",
                                    scale=3
                                )
                                review_load_btn = gr.Button("📂 加载评分结果", variant="secondary", scale=1)

                            review_image = gr.Image(label="当前图片", height=420, show_label=True,
                                                    elem_classes="output-text")
                            with gr.Row():
                                review_progress = gr.Textbox(label="进度", value="未加载", interactive=False, scale=1)

                        with gr.Column(scale=1):
                            review_info = gr.Markdown("等待加载评分结果...", elem_classes="output-text")
                            with gr.Row():
                                review_high_btn = gr.Button("✅ 高质量", variant="primary", size="lg")
                                review_low_btn = gr.Button("❌ 低质量", variant="stop", size="lg")
                            with gr.Row():
                                review_skip_btn = gr.Button("⏭️ 跳过", variant="secondary")
                                review_undo_btn = gr.Button("↩️ 撤销上一步", variant="secondary")
                            review_log = gr.Textbox(label="操作日志", lines=4, interactive=False, elem_classes="output-text")

                    def _load_image_safe(path):
                        """安全加载图片为 numpy 数组，避免 Gradio 路径权限问题"""
                        from PIL import Image, ImageOps
                        img = Image.open(path)
                        try:
                            img = ImageOps.exif_transpose(img)
                        except Exception:
                            pass
                        import numpy as np
                        return np.array(img)

                    def _on_review_load(input_dir):
                        from image_tools.detect_ai_errors import load_review_queue
                        queue = load_review_queue(input_dir)
                        if not queue:
                            return {}, None, "❌ 未找到评分文件（.txt），请先执行评分。", "0/0", "未找到评分文件"
                        item = queue[0]
                        info = _format_review_item(item)
                        progress = f"第 1/{len(queue)} 张"
                        log = f"✅ 已加载 {len(queue)} 张已评分图片"
                        return {"queue": queue, "index": 0, "last_moved": None}, _load_image_safe(item["path"]), info, progress, log

                    def _format_review_item(item):
                        if item["error"]:
                            return f"### ❌ 错误图片\n\n**错误**: {item['error']}\n\n---"
                        score = item["score"]
                        emoji = "🟢" if score >= 7 else ("🟡" if score >= 5 else ("🟠" if score >= 3 else "🔴"))
                        return f"### {emoji} 评分: {score} 分\n\n**点评**: {item['reason']}\n\n---"

                    def _review_action(action, state, input_dir):
                        if not state or not state["queue"]:
                            return state, None, "无图片可处理", "0/0", "请先加载评分结果"
                        queue = state["queue"]
                        idx = state["index"]
                        if idx >= len(queue):
                            return state, None, "✅ 审核完成！所有图片已处理。", f"{len(queue)}/{len(queue)}", "审核完成"
                        item = queue[idx]
                        img_path = str(item["path"])
                        log = ""

                        if action == "undo":
                            last = state.get("last_moved")
                            if last:
                                from image_tools.detect_ai_errors import move_single_to_category
                                result = move_single_to_category(last["path"], "undo", input_dir)
                                log = result["msg"]
                                from image_tools.detect_ai_errors import load_review_queue
                                queue = load_review_queue(input_dir)
                                idx = max(0, idx - 1)
                            else:
                                log = "没有可撤销的操作"
                        elif action == "skip":
                            log = f"跳过: {Path(img_path).name}"
                            idx += 1
                        else:
                            from image_tools.detect_ai_errors import move_single_to_category
                            result = move_single_to_category(img_path, action, input_dir)
                            log = result["msg"]
                            if result["ok"]:
                                state["last_moved"] = {"path": img_path, "category": action}
                                idx += 1

                        if idx >= len(queue):
                            return state, None, "✅ 审核完成！", f"{len(queue)}/{len(queue)}", log

                        next_item = queue[idx]
                        info = _format_review_item(next_item)
                        progress = f"第 {idx+1}/{len(queue)} 张"
                        return {"queue": queue, "index": idx, "last_moved": state.get("last_moved")}, _load_image_safe(next_item["path"]), info, progress, log

                    review_load_btn.click(_on_review_load, [review_input],
                                          [review_state, review_image, review_info, review_progress, review_log])

                    review_high_btn.click(lambda s, d: _review_action("high", s, d),
                                          [review_state, review_input],
                                          [review_state, review_image, review_info, review_progress, review_log])
                    review_low_btn.click(lambda s, d: _review_action("low", s, d),
                                         [review_state, review_input],
                                         [review_state, review_image, review_info, review_progress, review_log])
                    review_skip_btn.click(lambda s, d: _review_action("skip", s, d),
                                          [review_state, review_input],
                                          [review_state, review_image, review_info, review_progress, review_log])
                    review_undo_btn.click(lambda s, d: _review_action("undo", s, d),
                                          [review_state, review_input],
                                          [review_state, review_image, review_info, review_progress, review_log])

        # ==================== Tab 3: 聊天截图识别 ====================
        with gr.Tab("💬 聊天截图识别"):
            gr.Markdown(_make_title("聊天记录长截图 → 文字提取 — 切片 + VLM 识别"))
            with gr.Accordion("📖 使用方法", open=False):
                gr.Markdown("**步骤：** ① 把微信/QQ 聊天长截图放进文件夹 → ② 点「开始识别」→ ③ 去 `chat_text_output/` 找生成的 TXT 文件\n\n"
                            "> 💡 长图会被自动切成 2000px 高的薄片，每片之间有 400px 重叠防止漏字\n"
                            "> 💡 建议使用 **Thinking 模型**（如 qwen3.6-35b-a3b-Thinking），识别更准确")
            with gr.Row():
                with gr.Column(scale=2):
                    ei_input = gr.Textbox(label="截图文件夹",
                                          value=s["ocr"].get("input_dir", str(config.DATA_DIR / "screenshots")),
                                          placeholder="粘贴聊天截图所在文件夹的完整路径",
                                          info="支持 .jpg / .jpeg / .png / .bmp / .webp")
                    with gr.Accordion("⚙️ 高级设置", open=False):
                        ei_model_type, ei_model, ei_fetch_btn, ei_fetch_st, ei_thinking = _make_model_selector(
                            "视觉模型", s["ocr"].get("model", config.VISION_MODEL_THINKING),
                            "需要视觉模型，推荐带 Thinking 能力的模型")
                        ei_temp = gr.Slider(0.0, 1.0, value=s["ocr"].get("temperature", 0.3), step=0.1, label="温度",
                                            info="越低越稳定，越高越有创意。OCR 任务建议 0.2-0.4")
                        ei_workers = gr.Slider(1, 4, value=s["ocr"].get("workers", 1), step=1, label="并行图片数",
                                               info="同时处理几张图。注意：过大可能导致显存不足")
                        ei_iworkers = gr.Slider(1, 4, value=s["ocr"].get("internal_workers", 2), step=1, label="单图内部并发",
                                                info="每张图内部分两半并行处理")
                        ei_maxtok = gr.Slider(1000, 8000, value=s["ocr"].get("max_tokens", 5000), step=500, label="最大 Token 数",
                                              info="输出上限，长截图可适当调大")
                        ei_maxsz = gr.Slider(512, 4096, value=config.IMAGE_MAX_SIZE, step=256,
                                             label="图片最大边长（px）",
                                             info="超过此值的图片会等比缩小，越小速度越快但精度可能下降")
                    with gr.Row():
                        ei_btn = gr.Button("开始识别", variant="primary")
                        ei_stop = gr.Button("停止", variant="stop")
                with gr.Column(scale=3):
                    ei_output = gr.Textbox(label="处理日志", lines=15, elem_classes="output-text",
                                           placeholder="处理完成后这里会显示识别进度和结果...")
            ei_btn.click(_explain_images, [ei_input, ei_model, ei_temp, ei_workers, ei_iworkers, ei_maxtok, ei_maxsz, provider_info, ei_thinking], [ei_output])
            def _stop_ocr():
                from image_tools.ocr_chat_screenshots import request_stop
                request_stop()
                return "⏹️ 已请求停止..."
            ei_stop.click(_stop_ocr, outputs=[ei_output])
            _bind_model_fetch(ei_fetch_btn, ei_model_type, ei_model, ei_fetch_st,
                             provider_info, config.VISION_MODEL_THINKING)

        # ==================== Tab 6: 聊天 ====================
        with gr.Tab("💬 聊天") as kb_chat_tab:
            with gr.Row():
                # ---- 左侧边栏：对话列表 ----
                with gr.Column(scale=1, min_width=200, elem_classes="kb-sidebar"):
                    gr.Markdown("### 💬 对话")
                    kb_new_chat_btn = gr.Button("＋ 新建对话", size="sm")
                    kb_chat_list = gr.Radio(label="", choices=[], interactive=True,
                                             container=False, elem_id="kb_chat_list")
                    with gr.Row():
                        kb_del_btn = gr.Button("🗑 删除", size="sm")
                        kb_refresh_btn = gr.Button("🔄 刷新", size="sm")
                    kb_chat_status = gr.Markdown("")

                # ---- 右侧：聊天主区域 ----
                with gr.Column(scale=4, elem_classes="kb-chat-fill"):
                    # 模型选择
                    kb_model_type, kb_model, kb_fetch_btn, kb_fetch_st, _ = _make_model_selector(
                        "模型", config.TEXT_MODEL, "", show_thinking=False)

                    # 功能开关
                    with gr.Row(elem_classes="kb-toolbar"):
                        kb_use_rag = gr.Checkbox(label="📚 知识库检索", value=True,
                                                  container=False, scale=0, min_width=80)
                        kb_use_web = gr.Checkbox(label="🌐 联网搜索", value=False,
                                                  container=False, scale=0, min_width=80)
                        kb_thinking = gr.Checkbox(label="🧠 深度思考", value=True,
                                                   container=False, scale=0, min_width=80)

                    # 聊天区域
                    kb_chatbot = gr.Chatbot(label="", placeholder="输入问题开始对话...",
                                             elem_id="kb_chatbot", show_label=False, height=500)

                    # 输入区域
                    with gr.Column(elem_classes="kb-input-row"):
                        kb_query = gr.Textbox(label="", placeholder="输入你的问题...",
                                              lines=2, show_label=False, container=False,
                                              elem_id="kb_query_box")
                        with gr.Row():
                            kb_btn = gr.Button("发送", variant="primary", scale=0, min_width=60)
                            kb_clear_btn = gr.Button("清空", size="sm", scale=0, min_width=50)

                    gr.Markdown("*AI 生成的内容可能不准确，请核实重要信息。*", elem_classes="kb-disclaimer")

            # ---- 状态 ----
            kb_all_chats = gr.State({})
            kb_active_name = gr.State("")

            # ---- 切换对话 ----
            def _switch_chat(selection, all_chats):
                if not selection:
                    return all_chats, "", [], ""
                all_chats = dict(all_chats) if all_chats else {}
                msgs = all_chats.get(selection, [])
                msgs, _ = _kb_load_chat(selection) if not msgs else (msgs, "")
                all_chats[selection] = msgs
                return all_chats, selection, msgs, f"📌 {selection}"

            # ---- 发送查询（流式） ----
            def _send_query_stream(query, model, all_chats, active_name, provider, thinking, use_rag, use_web):
                if not query or not query.strip():
                    yield all_chats, active_name, all_chats.get(active_name, []), gr.update(), gr.update(), "❌ 请输入问题"
                    return
                _apply_provider(provider)
                os.environ["ENABLE_THINKING"] = "true" if thinking else "false"
                all_chats = dict(all_chats) if all_chats else {}
                msgs = list(all_chats.get(active_name, []))

                model_name = model or config.TEXT_MODEL
                from langchain_openai import ChatOpenAI
                from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

                # 联网搜索
                web_results = _do_web_search(query.strip()) if use_web else None

                # 构建 LLM 输入
                if use_rag:
                    from text_tools.kb_chat import prepare_kb_context
                    prompt, lc_msgs_fb = prepare_kb_context(
                        query=query.strip(), keyword="", k=50,
                        batch_size=20, chat_history=msgs, web_context=web_results,
                    )
                    if prompt is not None:
                        llm_input = prompt
                    else:
                        llm_input = lc_msgs_fb
                elif web_results:
                    lc_msgs = [SystemMessage(content=f"以下是联网搜索的最新结果，请基于这些信息回答用户问题：\n\n{web_results}")]
                    for m in msgs:
                        if m["role"] == "user": lc_msgs.append(HumanMessage(content=m["content"]))
                        elif m["role"] == "assistant": lc_msgs.append(AIMessage(content=m["content"]))
                    lc_msgs.append(HumanMessage(content=query.strip()))
                    llm_input = lc_msgs
                else:
                    lc_msgs = []
                    for m in msgs:
                        if m["role"] == "user": lc_msgs.append(HumanMessage(content=m["content"]))
                        elif m["role"] == "assistant": lc_msgs.append(AIMessage(content=m["content"]))
                    lc_msgs.append(HumanMessage(content=query.strip()))
                    llm_input = lc_msgs

                llm = ChatOpenAI(
                    base_url=config.OPENAI_BASE_URL,
                    api_key=config.OPENAI_API_KEY,
                    model=model_name,
                    temperature=0.7,
                    streaming=True,
                    extra_body=config.get_llm_extra_body(thinking),
                )

                def _get_reasoning(chunk) -> str:
                    ak = getattr(chunk, 'additional_kwargs', {}) or {}
                    for key in ('reasoning_content', 'thinking', 'reasoning'):
                        if ak.get(key):
                            return ak[key]
                    rm = getattr(chunk, 'response_metadata', {}) or {}
                    for key in ('reasoning_content', 'thinking', 'reasoning'):
                        if rm.get(key):
                            return rm[key]
                    return ""

                # 流式生成
                new_msgs = list(msgs)
                new_msgs.append({"role": "user", "content": query.strip()})
                new_msgs.append({"role": "assistant", "content": ""})
                all_chats[active_name] = new_msgs
                radio_update = gr.update()
                new_active = active_name

                reasoning = ""
                answer = ""

                def _build_msg(reasoning, answer):
                    if reasoning:
                        return f"<details>\n<summary>🤔 思考过程</summary>\n\n{reasoning}\n\n</details>\n\n{answer}"
                    return answer

                try:
                    for chunk in llm.stream(llm_input):
                        r = _get_reasoning(chunk)
                        c = chunk.content if hasattr(chunk, 'content') and chunk.content else ""
                        if r:
                            reasoning += r
                        if c:
                            answer += c
                        new_msgs[-1]["content"] = _build_msg(reasoning, answer)
                        yield all_chats, new_active, new_msgs, gr.update(), radio_update, ""
                except Exception as e:
                    new_msgs[-1]["content"] = f"❌ 调用失败: {e}"
                    yield all_chats, new_active, new_msgs, gr.update(), radio_update, ""

                # 自动命名
                if active_name.startswith("新对话_") and len(new_msgs) >= 2:
                    try:
                        title = _kb_auto_title(provider, new_msgs)
                        if title and title != active_name:
                            del all_chats[active_name]
                            all_chats[title] = new_msgs
                            choices = sorted(all_chats.keys())
                            radio_update = gr.update(choices=choices, value=title)
                            new_active = title
                    except Exception:
                        pass

                # 自动保存
                _kb_save_chat(new_msgs, new_active)

                history.add_entry("知识库问答", query[:50], "查询完成")
                yield all_chats, new_active, new_msgs, gr.update(), radio_update, ""

            # ---- 清空当前对话 ----
            def _clear_chat_multi(all_chats, active_name):
                if not active_name:
                    return all_chats, active_name, [], "", "❌ 没有激活的对话"
                all_chats = dict(all_chats) if all_chats else {}
                all_chats[active_name] = []
                return all_chats, active_name, [], "", "✅ 已清空"

            # ---- 新建对话 ----
            def _new_chat_multi(all_chats, active_name):
                all_chats = dict(all_chats) if all_chats else {}
                import time as _time
                new_name = f"新对话_{_time.strftime('%H%M%S')}"
                all_chats[new_name] = []
                choices = sorted(all_chats.keys())
                return all_chats, new_name, [], gr.update(choices=choices, value=new_name), f"✅ {new_name}"

            # ---- 刷新对话列表 ----
            def _refresh_chat_list_multi(all_chats, active_name):
                all_chats = dict(all_chats) if all_chats else {}
                saved_choices, status = _kb_list_chats()
                merged = set(saved_choices) | set(all_chats.keys())
                choices = sorted(merged)
                return gr.update(choices=choices), status

            # ---- 删除对话 ----
            def _del_chat_multi(selection, all_chats, active_name):
                if not selection:
                    return all_chats, active_name, [], gr.update(), "❌ 请选择要删除的对话"
                all_chats = dict(all_chats) if all_chats else {}
                all_chats.pop(selection, None)
                _kb_delete_chat(selection)
                choices = sorted(all_chats.keys())
                new_active = active_name if active_name != selection else ""
                new_msgs = all_chats.get(new_active, []) if new_active else []
                status = f"✅ 已删除 {selection}"
                return (all_chats, new_active, new_msgs,
                        gr.update(choices=choices, value=new_active if new_active else None), status)

            # ---- 事件绑定 ----
            kb_chat_list.change(_switch_chat, [kb_chat_list, kb_all_chats],
                               [kb_all_chats, kb_active_name, kb_chatbot, kb_chat_status])

            kb_btn.click(_send_query_stream,
                         [kb_query, kb_model, kb_all_chats, kb_active_name, provider_info, kb_thinking, kb_use_rag, kb_use_web],
                         [kb_all_chats, kb_active_name, kb_chatbot, kb_query, kb_chat_list, kb_chat_status])
            kb_query.submit(_send_query_stream,
                            [kb_query, kb_model, kb_all_chats, kb_active_name, provider_info, kb_thinking, kb_use_rag, kb_use_web],
                            [kb_all_chats, kb_active_name, kb_chatbot, kb_query, kb_chat_list, kb_chat_status])

            kb_clear_btn.click(_clear_chat_multi, [kb_all_chats, kb_active_name],
                              [kb_all_chats, kb_active_name, kb_chatbot, kb_query, kb_chat_status])
            kb_new_chat_btn.click(_new_chat_multi, [kb_all_chats, kb_active_name],
                                 [kb_all_chats, kb_active_name, kb_chatbot, kb_chat_list, kb_chat_status])
            kb_refresh_btn.click(_refresh_chat_list_multi, [kb_all_chats, kb_active_name],
                                [kb_chat_list, kb_chat_status])
            kb_del_btn.click(_del_chat_multi, [kb_chat_list, kb_all_chats, kb_active_name],
                            [kb_all_chats, kb_active_name, kb_chatbot, kb_chat_list, kb_chat_status])

            _kb_do_fetch = _bind_model_fetch(kb_fetch_btn, kb_model_type, kb_model, kb_fetch_st,
                             provider_info, config.TEXT_MODEL)

            def _auto_refresh_models(prov):
                result = _kb_do_fetch(prov)
                return result[0], result[1], result[2]

            kb_chat_tab.select(_auto_refresh_models, [provider_info],
                              [kb_model_type, kb_model, kb_fetch_st])
            kb_chat_tab.select(_refresh_chat_list_multi, [kb_all_chats, kb_active_name],
                              [kb_chat_list, kb_chat_status])

        # ==================== Tab 7: 知识库设置 ====================
        with gr.Tab("📚 知识库"):
            gr.Markdown(_make_title("知识库管理 — 创建知识库、上传文档、构建索引"))
            # ---- 顶部：知识库选择 + 创建 ----
            _kb_bases, _kb_choices, _kb_labels = _kb_get_choices()
            _kb_default = _kb_choices[0] if _kb_choices else ""
            with gr.Row():
                kb_selector = gr.Dropdown(
                    label="知识库名称",
                    choices=_kb_choices,
                    value=_kb_default,
                    interactive=True,
                    allow_custom_value=True,
                    scale=4,
                )
                kb_selector_refresh = gr.Button("🔄", scale=0, min_width=50)
                kb_create_name = gr.Textbox(label="", placeholder="输入新知识库名称", scale=2,
                                            show_label=False, container=False)
                kb_create_btn = gr.Button("➕ 创建", variant="secondary", scale=0, min_width=70)
            kb_selector_info = gr.Markdown("")
            kb_selected_state = gr.State(_kb_default)

            def _on_kb_select(selected):
                if not selected:
                    return "", selected
                from text_tools.kb_manager import list_documents
                docs = list_documents(Path(selected))
                label = _kb_labels.get(selected, Path(selected).name)
                info = f"**当前知识库:** {label}  |  路径: `{selected}`  |  文档数: {len(docs)}"
                return info, selected

            def _on_kb_refresh():
                bases, choices, labels = _kb_get_choices()
                default = choices[0] if choices else ""
                info, selected = _on_kb_select(default)
                return gr.update(choices=choices, value=default), info, selected

            kb_selector.change(_on_kb_select, [kb_selector], [kb_selector_info, kb_selected_state])
            kb_selector_refresh.click(_on_kb_refresh, outputs=[kb_selector, kb_selector_info, kb_selected_state])
            kb_create_btn.click(_kb_create_kb, [kb_create_name], [kb_selector_info, kb_selector])

            # 初始加载
            _init_info = ""
            if _kb_default:
                from text_tools.kb_manager import list_documents as _list_docs_init
                _init_docs = _list_docs_init(Path(_kb_default))
                _init_label = _kb_labels.get(_kb_default, Path(_kb_default).name)
                _init_info = f"**当前知识库:** {_init_label}  |  路径: `{_kb_default}`  |  文档数: {len(_init_docs)}"
            kb_selector_info.value = _init_info

            with gr.Tabs():
                # ---- 子 Tab: 文档管理 ----
                with gr.Tab("📄 文档管理"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("### 上传文档")
                            kb_upload_files = gr.File(label="选择文件（支持 .txt .md .csv .json .log 等）",
                                                      file_count="multiple",
                                                      file_types=[".txt", ".md", ".csv", ".json", ".jsonl", ".log", ".py", ".rst"])
                            kb_upload_btn = gr.Button("📤 上传到知识库", variant="primary")
                            kb_upload_result = gr.Textbox(label="上传结果", lines=4, interactive=False)

                            gr.Markdown("### 删除文档")
                            kb_del_name = gr.Textbox(label="文件名", placeholder="输入要删除的文件名")
                            with gr.Row():
                                kb_del_btn = gr.Button("🗑️ 删除指定文件")
                                kb_del_all_btn = gr.Button("🗑️ 清空全部", variant="stop")
                            kb_del_result = gr.Textbox(label="删除结果", lines=2, interactive=False)

                        with gr.Column(scale=3):
                            gr.Markdown("### 文档列表")
                            kb_doc_list = gr.Markdown("*点击「刷新」加载文档列表*")
                            kb_doc_info = gr.Textbox(label="统计", lines=1, interactive=False)
                            kb_doc_refresh_btn = gr.Button("🔄 刷新文档列表")

                    kb_upload_btn.click(lambda f, kb: _kb_upload(f, kb), [kb_upload_files, kb_selected_state], [kb_upload_result])
                    kb_del_btn.click(lambda n, kb: _kb_delete(n, kb), [kb_del_name, kb_selected_state], [kb_del_result])
                    kb_del_all_btn.click(lambda kb: _kb_delete_all(kb), [kb_selected_state], [kb_del_result])
                    kb_doc_refresh_btn.click(lambda kb: _kb_list_docs(kb), [kb_selected_state], [kb_doc_list, kb_doc_info])

                # ---- 子 Tab: 索引构建 ----
                with gr.Tab("🔧 索引构建"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("### 构建参数")
                            kb_chunk_size = gr.Slider(100, 2000, value=config.KB_CHUNK_SIZE, step=50, label="文本块大小（字符）",
                                                      info="每个文本块的最大字符数")
                            kb_chunk_overlap = gr.Slider(0, 200, value=config.KB_CHUNK_OVERLAP, step=10, label="块间重叠（字符）",
                                                         info="相邻文本块重叠的字符数，有助于保持上下文连贯")
                            kb_embed_model = gr.Textbox(label="Embedding 模型（留空用默认）",
                                                        placeholder="如 BAAI/bge-small-zh-v1.5 或本地路径",
                                                        value="",
                                                        info="HuggingFace 模型名或本地路径")
                            with gr.Row():
                                kb_build_btn = gr.Button("🔨 构建索引", variant="primary")
                                kb_build_stop = gr.Button("⏹️ 停止", variant="stop")
                            kb_build_result = gr.Textbox(label="构建日志", lines=10, interactive=False,
                                                         placeholder="点击「构建索引」开始...")

                        with gr.Column(scale=2):
                            gr.Markdown("### 索引状态")
                            kb_stats_btn = gr.Button("🔄 刷新状态")
                            kb_stats_display = gr.Markdown("*点击「刷新状态」查看*")

                    kb_build_btn.click(lambda cs, co, em, kb: _kb_build_index(cs, co, em, kb),
                                       [kb_chunk_size, kb_chunk_overlap, kb_embed_model, kb_selected_state],
                                       [kb_build_result])
                    kb_build_stop.click(_kb_stop_build, outputs=[kb_build_result])
                    kb_stats_btn.click(lambda kb: _kb_get_stats(kb), [kb_selected_state], [kb_stats_display])

        # ==================== Tab 4: 文本工具（聊天压缩 + 翻译）====================
        with gr.Tab("📝 文本工具"):
            with gr.Tabs():
                # ---- 子 Tab: 聊天记录压缩 ----
                with gr.Tab("📄 聊天记录压缩"):
                    gr.Markdown(_make_title("聊天记录 txt 文件 → 精简格式化"))
                    with gr.Accordion("📖 使用方法", open=False):
                        gr.Markdown("**步骤：** ① 把截图 OCR 生成的 TXT（或任意聊天记录 TXT）放进文件夹 → ② 点「开始压缩」→ ③ 得到 `.compressed.txt` 文件\n\n"
                                    "> 💡 压缩会合并连续相同说话人的时间戳，移除系统消息等冗余内容，保留所有对话实质\n"
                                    "> 📎 **典型流程：** 截图 → 截图识别 → 本功能压缩 → 得到干净对话文本")
                    with gr.Row():
                        with gr.Column(scale=2):
                            ct_input = gr.Textbox(label="输入文件/文件夹",
                                                  value=s["compress"].get("input_path", str(config.DATA_DIR / "screenshots" / "texts")),
                                                  placeholder="粘贴聊天记录 TXT 文件或文件夹路径",
                                                  info="可以是单个 .txt 文件，也可以是装多个 .txt 的文件夹")
                            with gr.Accordion("⚙️ 高级设置", open=False):
                                ct_model_type, ct_model, ct_fetch_btn, ct_fetch_st, ct_thinking = _make_model_selector(
                                    "文本模型", s["compress"].get("model", config.VISION_MODEL_THINKING),
                                    "纯文本任务，用普通文本模型即可，不必用视觉模型")
                                ct_temp = gr.Slider(0.0, 0.5, value=s["compress"].get("temperature", 0.2), step=0.05, label="温度",
                                                    info="精简任务建议低温 0.1-0.3，保持稳定")
                                ct_chunk = gr.Slider(5000, 50000, value=s["compress"].get("chunk_size", config.DEFAULT_CHUNK_SIZE), step=1000,
                                                     label="分块大小（字符）",
                                                     info="每块给 LLM 处理的文字量。越大单次成本越高，但分段更连贯")
                                ct_iw = gr.Slider(1, 4, value=s["compress"].get("internal_workers", 2), step=1, label="内部并发数",
                                                  info="同时处理几块文本")
                                ct_maxtok = gr.Slider(1000, 8000, value=s["compress"].get("max_tokens", 4000), step=500, label="最大 Token 数",
                                                      info="输出长度上限")
                            with gr.Row():
                                ct_btn = gr.Button("开始压缩", variant="primary")
                                ct_stop = gr.Button("停止", variant="stop")
                        with gr.Column(scale=3):
                            ct_output = gr.Textbox(label="处理日志", lines=15, elem_classes="output-text",
                                                   placeholder="处理完成后这里会显示压缩进度...")
                    ct_btn.click(_compress_text, [ct_input, ct_model, ct_temp, ct_chunk, ct_iw, ct_maxtok, provider_info, ct_thinking], [ct_output])
                    def _stop_compress():
                        from text_tools.compress_chat import request_stop
                        request_stop()
                        return "⏹️ 已请求停止..."
                    ct_stop.click(_stop_compress, outputs=[ct_output])
                    _bind_model_fetch(ct_fetch_btn, ct_model_type, ct_model, ct_fetch_st,
                                     provider_info, config.VISION_MODEL_THINKING)

                # ---- 子 Tab: 文本翻译 ----
                with gr.Tab("🌐 文本翻译"):
                    gr.Markdown(_make_title("长篇文本翻译 — 按章节切分，断点续传"))
                    with gr.Accordion("📖 使用方法", open=False):
                        gr.Markdown("**步骤：** ① 把要翻译的文本文件路径填好 → ② 点「开始翻译」→ ③ 去 `outputs/translation/` 找译文\n\n"
                                    "> 💡 自动识别「第X章」「Chapter」等章节标记，按章节切分翻译\n"
                                    "> 💡 支持**断点续传**：中途中断后重新点开始，会接着上次进度继续\n"
                                    "> 💡 目标语言在 `.env` 中设置（默认 `English`）")
                    with gr.Row():
                        with gr.Column(scale=2):
                            tr_input = gr.Textbox(label="输入文件",
                                                  value=s["translate"].get("input_file", str(config.DATA_DIR / "texts" / "input.txt")),
                                                  placeholder="粘贴要翻译的 .txt 文件完整路径",
                                                  info="文件编码需为 UTF-8")
                            tr_output_file = gr.Textbox(label="输出文件",
                                                        value=s["translate"].get("output_file", ""),
                                                        placeholder="留空则自动保存到 outputs/translation/translation.txt",
                                                        info="指定译文保存路径，留空自动生成")
                            with gr.Accordion("⚙️ 高级设置", open=False):
                                tr_model_type, tr_model, tr_fetch_btn, tr_fetch_st, tr_thinking = _make_model_selector(
                                    "文本模型", s["translate"].get("model", config.TEXT_MODEL),
                                    "纯文本翻译任务，使用文本模型")
                                tr_batch = gr.Slider(1, 20, value=s["translate"].get("batch_size", 10), step=1, label="每批章节数",
                                                     info="每批同时翻译多少章。越大越快但可能超时")
                                tr_workers = gr.Slider(1, 4, value=s["translate"].get("workers", 2), step=1, label="并行线程数",
                                                       info="同时运行几个翻译任务")
                            with gr.Row():
                                tr_btn = gr.Button("开始翻译", variant="primary")
                                tr_stop = gr.Button("停止", variant="stop")
                        with gr.Column(scale=3):
                            tr_output = gr.Textbox(label="翻译日志", lines=15, elem_classes="output-text",
                                                   placeholder="处理完成后这里会显示翻译进度、已翻章节数...")
                    tr_btn.click(_translate, [tr_input, tr_output_file, tr_model, tr_batch, tr_workers, provider_info, tr_thinking], [tr_output])
                    def _stop_translate():
                        from text_tools.translate import request_stop
                        request_stop()
                        return "⏹️ 已请求停止..."
                    tr_stop.click(_stop_translate, outputs=[tr_output])
                    _bind_model_fetch(tr_fetch_btn, tr_model_type, tr_model, tr_fetch_st,
                                     provider_info, config.TEXT_MODEL)

        # ==================== Tab 8: LLM 压测 ====================
        with gr.Tab("⚡ LLM 压测"):
            gr.Markdown(_make_title("大模型推理性能测试 — 生成专业压测报告"))
            with gr.Accordion("📖 测试说明", open=False):
                gr.Markdown("**测试流程：** ① 填写 API 连接信息 → ② 设置测试参数 → ③ 点「开始测试」→ ④ 查看报告图表\n\n"
                            "| 测试项 | 说明 |\n|---|---|\n"
                            "| TTFT | 首 Token 延迟（Time to First Token），越小越好 |\n"
                            "| ITTL | Token 间延迟（Inter-Token Latency），影响生成流畅度 |\n"
                            "| Prefill | 预填充吞吐量，衡量 prompt 处理速度 |\n"
                            "| Decode | 输出吞吐量，衡量生成速度 |\n\n"
                            "> ⏱️ 压测会占用模型全部资源，建议在空闲时运行\n"
                            "> 📈 测试报告自动保存到 `outputs/benchmarks/`")
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 连接配置")
                    bm_url = gr.Textbox(label="API Base URL", value=s["benchmark"].get("url", config.OPENAI_BASE_URL),
                                        info="被测 API 地址，如 http://localhost:1234/v1")
                    bm_model_type, bm_model, bm_fetch_btn, bm_fetch_st, bm_thinking = _make_model_selector(
                        "被测模型", s["benchmark"].get("model", config.BENCHMARK_MODEL),
                        "被测模型，需与 API 中实际名称一致", show_thinking=False)
                    bm_key = gr.Textbox(label="API Key", value=s["benchmark"].get("api_key", config.OPENAI_API_KEY), type="password",
                                        info="本地服务如 LM Studio 填任意值即可")
                    gr.Markdown("### 测试参数")
                    bm_outtok = gr.Slider(64, 2048, value=s["benchmark"].get("output_tokens", 512), step=64, label="生成 Token 数",
                                          info="每次请求让模型生成多少 token")
                    bm_concur = gr.Slider(1, 8, value=s["benchmark"].get("concurrency", config.DEFAULT_WORKERS), step=1, label="并发数",
                                          info="同时发几个请求。越大越能测出吞吐上限")
                    bm_lengths = gr.Textbox(label="测试 Prompt 长度（逗号分隔）",
                                            value=s["benchmark"].get("lengths_str", "512,1024,2048,4096"),
                                            info="不同输入长度分别测试")
                    with gr.Accordion("⚙️ 更多设置", open=False):
                        bm_timeout = gr.Slider(10, 120, value=s["benchmark"].get("timeout", config.REQUEST_TIMEOUT_SHORT), step=5, label="超时（秒）",
                                               info="单个请求超时时间")
                    with gr.Row():
                        bm_btn = gr.Button("开始测试", variant="primary")
                        bm_stop = gr.Button("停止", variant="stop")
                with gr.Column(scale=3):
                    bm_text = gr.Textbox(label="测试进度", lines=12, elem_classes="output-text",
                                         placeholder="点击「开始测试」后，实时显示测试进度和结果...")
                    bm_plot = gr.Image(label="测试报告图表")
            bm_btn.click(_benchmark, [bm_url, bm_model, bm_key, bm_concur, bm_timeout, bm_outtok, bm_lengths],
                        [bm_text, bm_plot])
            def _stop_benchmark():
                from benchmarks.speedtest import request_stop
                request_stop()
                return "⏹️ 已请求停止..."
            bm_stop.click(_stop_benchmark, outputs=[bm_text])
            _bind_model_fetch_local(bm_fetch_btn, bm_model_type, bm_model, bm_fetch_st,
                             bm_url, bm_key, s["benchmark"].get("model", config.BENCHMARK_MODEL))

        # ==================== Tab 8: 设置 ====================
        with gr.Tab("⚙️ 设置"):
            gr.Markdown(_make_title("全局配置 — 在线编辑 .env 文件"))
            gr.Markdown("在此修改 API 地址、模型名称、并发参数等。点「保存」后写入 `.env` 文件，**部分配置需重启应用才能生效**。")

            current_env = _read_env()
            setting_inputs: dict = {}

            # 两列布局：左列（API/模型/并发），右列（图片/文本/更新/目录）
            _LEFT_SECTIONS = {"🔗 API 连接", "🤖 模型名称", "⚡ 并发与超时"}

            def _render_field(key, label, default_val, kind):
                cur_val = current_env.get(key, default_val)
                if kind == "password":
                    return gr.Textbox(label=label, value=cur_val, type="password")
                elif kind == "bool":
                    bool_default = cur_val.lower() == "true"
                    return gr.Dropdown(
                        label=label, value=str(bool_default),
                        choices=["True", "False"],
                        info="True = 启动时自动 git pull 检查更新；False = 仅手动更新"
                    )
                elif kind in ("int", "float"):
                    return gr.Textbox(label=label, value=str(cur_val), placeholder=default_val)
                else:
                    return gr.Textbox(label=label, value=cur_val, placeholder=default_val)

            with gr.Row():
                with gr.Column():
                    for section_name, fields in SETTINGS_SCHEMA:
                        if section_name in _LEFT_SECTIONS:
                            with gr.Accordion(section_name, open=False):
                                for key, label, default_val, kind in fields:
                                    setting_inputs[key] = _render_field(key, label, default_val, kind)
                with gr.Column():
                    for section_name, fields in SETTINGS_SCHEMA:
                        if section_name not in _LEFT_SECTIONS:
                            with gr.Accordion(section_name, open=False):
                                for key, label, default_val, kind in fields:
                                    setting_inputs[key] = _render_field(key, label, default_val, kind)

            save_btn = gr.Button("💾 保存设置", variant="primary", scale=0)
            save_msg = gr.Textbox(label="", interactive=False, container=False, show_label=False)

            # 恢复默认设置
            gr.Markdown("---")
            gr.Markdown(_make_title("🔄 恢复默认设置"))
            gr.Markdown("清除所有工具的参数记忆（输入目录、高级选项等），下次启动时恢复为默认值。")
            with gr.Row():
                reset_btn = gr.Button("🔄 恢复默认设置", variant="stop", scale=0)
                def _on_reset():
                    config.clear_state()
                    return "✅ 已清除所有保存的参数。请重启应用以加载默认值。"
                reset_btn.click(_on_reset, outputs=[save_msg])

            # 按固定顺序收集所有输入组件及其对应的 key
            _input_keys = []
            _input_widgets = []
            for _, fields in SETTINGS_SCHEMA:
                for key, _, _, _ in fields:
                    _input_keys.append(key)
                    _input_widgets.append(setting_inputs[key])

            def _on_save(*values):
                updates = dict(zip(_input_keys, values))
                return _save_env(updates)

            save_btn.click(_on_save, _input_widgets, [save_msg])

            # ---- API 连接测试 ----
            gr.Markdown("---")
            gr.Markdown(_make_title("🔗 API 连接测试"))

            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    test_base = gr.Textbox(label="API 地址", value=config.OPENAI_BASE_URL,
                                           info="与上方「API 连接」分组中的地址保持一致")
                with gr.Column(scale=2):
                    test_key = gr.Textbox(label="API 密钥", value=config.OPENAI_API_KEY, type="password",
                                          info="本地服务填任意值即可")

            with gr.Row(equal_height=True):
                with gr.Column(scale=4):
                    test_model_type, test_model, test_fetch_btn, test_fetch_st, test_thinking = _make_model_selector(
                        "测试模型（可选）", config.VISION_MODEL,
                        "检查该模型是否可用，留空则只测试连接", show_thinking=False)
                with gr.Column(scale=1, min_width=120):
                    test_btn = gr.Button("🔗 测试连接", variant="secondary")
            test_msg = gr.Textbox(label="测试结果", interactive=False, lines=4, elem_classes="output-text",
                                  placeholder="点击「测试连接」查看结果...")

            test_btn.click(_test_api_connection, [test_base, test_key, test_model], [test_msg])
            _bind_model_fetch_local(test_fetch_btn, test_model_type, test_model, test_fetch_st,
                             test_base, test_key, config.VISION_MODEL)

            # ---- 手动更新区域 ----
            gr.Markdown("---")
            gr.Markdown(_make_title("📦 手动更新"))
            gr.Markdown(f"当前版本: `{_get_current_version()}`")

            with gr.Row():
                check_btn = gr.Button("🔍 检查更新", variant="secondary", scale=1)
                update_btn = gr.Button("⬇️ 立即更新", variant="primary", scale=1)
            update_output = gr.Textbox(label="更新日志", lines=6, elem_classes="output-text",
                                       placeholder="检查/更新结果会显示在这里...")

            def _on_check():
                has_update, local, remote = _check_update()
                if has_update is None:
                    return "⚠️ 无法检查更新（网络异常或非 git 环境）"
                if has_update:
                    return f"🔔 发现新版本! 本地: {local} → 远程: {remote}"
                return f"✅ 当前已是最新版本 (本地: {local})"

            check_btn.click(_on_check, [], [update_output])
            update_btn.click(_do_update, [], [update_output])

        # ==================== 供应商切换 → 自动刷新所有模型列表 ====================
        def _refresh_all_models(prov):
            """供应商切换后，一次性刷新所有 Tab 的模型下拉框"""
            url = prov.get("base_url", "") if isinstance(prov, dict) else ""
            key = prov.get("api_key", "") if isinstance(prov, dict) else ""
            categorized, msg = _fetch_models(url, key)
            all_models = categorized.get("全部模型", [])
            cat_choices = list(categorized.keys())

            def _upd(default):
                val = default if default in all_models else (all_models[0] if all_models else "")
                return gr.update(choices=all_models, value=val), gr.update(choices=cat_choices, value="全部模型")

            rn_m, rn_t = _upd(config.RENAME_MODEL)
            de_m, de_t = _upd(config.VISION_MODEL)
            ei_m, ei_t = _upd(config.VISION_MODEL_THINKING)
            ct_m, ct_t = _upd(config.VISION_MODEL_THINKING)
            tr_m, tr_t = _upd(config.TEXT_MODEL)
            kb_m, kb_t = _upd(config.TEXT_MODEL)
            return (rn_m, rn_t, de_m, de_t, ei_m, ei_t, ct_m, ct_t, tr_m, tr_t, kb_m, kb_t)

        _model_outputs = [rn_model, rn_model_type, de_model, de_model_type,
                          ei_model, ei_model_type, ct_model, ct_model_type,
                          tr_model, tr_model_type, kb_model, kb_model_type]

        prov_save_btn.click(
            _on_prov_save_and_refresh,
            inputs=[prov_edit_mode, prov_name, prov_name, prov_url, prov_key, providers_state],
            outputs=[prov_msg, providers_state, provider_select, prov_info_text, provider_info],
        ).then(_refresh_all_models, inputs=[provider_info], outputs=_model_outputs)

        prov_del_btn.click(
            _on_prov_delete_and_refresh,
            inputs=[providers_state],
            outputs=[prov_msg, providers_state, provider_select, prov_info_text, provider_info],
        ).then(_refresh_all_models, inputs=[provider_info], outputs=_model_outputs)

        provider_select.change(
            _on_provider_change,
            inputs=[provider_select, providers_state],
            outputs=[prov_info_text, provider_info, providers_state],
        ).then(_refresh_all_models, inputs=[provider_info], outputs=_model_outputs)

    return app


if __name__ == "__main__":
    import traceback
    try:
        update_msg = _check_and_update_on_startup()
        if update_msg:
            print(update_msg)
        logger.info(f"LocalAITools 启动中... API={config.OPENAI_BASE_URL}")
        build_ui().launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True,
                          theme=gr.themes.Soft(), css=CSS, js=JS_ONLOAD)
    except Exception:
        print("\n❌ 启动失败：\n")
        traceback.print_exc()
        print("\n按 Enter 键退出...")
        input()
