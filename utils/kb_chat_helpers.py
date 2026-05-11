"""KB chat and document management helpers."""
import json
import re
import time
import os
import logging
from pathlib import Path
import gradio as gr
import config
import history

logger = logging.getLogger("LocalAITools")

_KB_HISTORY_DIR = config.OUTPUT_DIR / "kb_chats"
_KB_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _kb_save_chat(chat_messages, chat_name):
    """保存聊天记录到文件"""
    if not chat_messages:
        return "❌ 没有聊天记录可保存"
    logger.info(f"[保存聊天] 名称={chat_name} 消息数={len(chat_messages)}")
    if not chat_name or not chat_name.strip():
        chat_name = f"chat_{time.strftime('%Y%m%d_%H%M%S')}"
    chat_name = chat_name.strip()
    safe_name = re.sub(r'[\\/*?:"<>|]', '', chat_name)
    path = _KB_HISTORY_DIR / f"{safe_name}.json"
    data = {
        "name": chat_name,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "messages": chat_messages,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"✅ 已保存: {path.name}"


def _kb_get_saved_chat(selection):
    """根据名称查找已保存的聊天文件，返回 (path, data) 或 (None, None)"""
    for f in sorted(_KB_HISTORY_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            name = _kb_clean_name(data.get("name", f.stem))
            if name == selection or f.stem == selection:
                return f, data
        except Exception:
            continue
    return None, None


def _kb_clean_name(raw: str) -> str:
    """去掉旧版本可能附带的 ' (YYYY-MM-DD HH:MM:SS, N条)' 后缀"""
    return re.sub(r'\s*\(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\s*\d+条\)\s*$', '', raw)

def _kb_list_chats():
    """列出所有保存的聊天记录"""
    files = sorted(_KB_HISTORY_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return [], "暂无保存的聊天记录"
    choices = []
    for f in files[:50]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
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


def _kb_import_url(url, selected_kb=None):
    """从 URL 抓取网页内容并导入知识库"""
    if not url or not url.strip():
        return "❌ 请输入 URL"
    logger.info(f"[URL 导入] url={url}")
    from text_tools.kb_manager import import_url
    docs_dir = Path(selected_kb) if selected_kb else None
    return import_url(url.strip(), docs_dir)

