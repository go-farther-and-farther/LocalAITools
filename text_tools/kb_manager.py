#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库文档管理器 — 上传、列出、删除文档，构建/重建 FAISS 索引
"""

import os
import shutil
import threading
from pathlib import Path
from typing import List, Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

_stop_flag = threading.Event()


def request_stop():
    _stop_flag.set()


# ==================== 路径 ====================

def get_docs_dir() -> Path:
    """知识库源文档目录"""
    d = config.DATA_DIR / "knowledge_docs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_index_dir() -> Path:
    """FAISS 索引目录"""
    d = Path(config.FAISS_INDEX_PATH)
    if not d.is_absolute():
        d = config.ROOT_DIR / d
    d.mkdir(parents=True, exist_ok=True)
    return d


# ==================== 文档操作 ====================

SUPPORTED_EXTS = {".txt", ".md", ".csv", ".json", ".jsonl", ".log", ".py", ".rst"}


def list_documents(docs_dir: Path = None) -> List[Dict]:
    """列出知识库目录中的所有文档，返回 [{name, size, modified, type}]"""
    docs_dir = docs_dir or get_docs_dir()
    files = []
    for f in sorted(docs_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
            stat = f.stat()
            files.append({
                "name": f.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "type": f.suffix.lower(),
                "path": str(f),
            })
    return files


def upload_documents(file_paths: List[str], docs_dir: Path = None) -> List[str]:
    """上传文件到知识库目录，返回操作结果消息列表"""
    docs_dir = docs_dir or get_docs_dir()
    results = []
    for fp in file_paths:
        src = Path(fp)
        if not src.exists():
            results.append(f"❌ 文件不存在: {src.name}")
            continue
        if src.suffix.lower() not in SUPPORTED_EXTS:
            results.append(f"⚠️ 不支持的格式，跳过: {src.name}")
            continue
        dst = docs_dir / src.name
        if dst.exists():
            # 覆盖
            dst.unlink()
        shutil.copy2(str(src), str(dst))
        results.append(f"✅ 已上传: {src.name}")
    return results


def delete_document(filename: str, docs_dir: Path = None) -> str:
    """删除指定文档"""
    docs_dir = docs_dir or get_docs_dir()
    target = docs_dir / filename
    if not target.exists():
        return f"❌ 文件不存在: {filename}"
    try:
        target.unlink()
        return f"✅ 已删除: {filename}"
    except Exception as e:
        return f"❌ 删除失败: {filename} — {e}"


def delete_all_documents(docs_dir: Path = None) -> str:
    """清空知识库目录"""
    docs_dir = docs_dir or get_docs_dir()
    count = 0
    for f in docs_dir.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
            f.unlink()
            count += 1
    return f"✅ 已删除 {count} 个文档"


# ==================== 索引构建 ====================

HF_MIRROR = "https://hf-mirror.com"


def _get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device(0)
    except (ImportError, Exception):
        pass
    return "cpu"


def _load_embeddings_safe(model_path: str, log_fn=None):
    """加载 Embedding 模型，失败时自动切换 HuggingFace 镜像重试。返回 embeddings 或错误字符串。"""
    from langchain_huggingface import HuggingFaceEmbeddings
    device = _get_device()
    if log_fn:
        log_fn(f"📐 Embedding 设备: {device}")
    kwargs = dict(
        model_name=model_path,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True},
    )
    try:
        return HuggingFaceEmbeddings(**kwargs)
    except Exception as e:
        err = str(e).lower()
        if "ssl" in err or "certificate" in err or "connect" in err or "timeout" in err:
            if log_fn:
                log_fn(f"⚠️ HuggingFace 连接失败，自动切换镜像: {HF_MIRROR}")
            os.environ["HF_ENDPOINT"] = HF_MIRROR
            try:
                return HuggingFaceEmbeddings(**kwargs)
            except Exception as e2:
                return f"❌ 镜像也失败了: {e2}\n请手动下载模型后在设置中填写本地路径"
        return f"❌ 加载 Embedding 模型失败: {e}"


def build_index(
    docs_dir: Path = None,
    index_dir: Path = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    embedding_model: str = None,
    progress_callback=None,
) -> str:
    """从文档目录构建 FAISS 索引"""
    docs_dir = docs_dir or get_docs_dir()
    index_dir = index_dir or get_index_dir()
    if chunk_size is None:
        chunk_size = config.KB_CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = config.KB_CHUNK_OVERLAP

    def _log(msg):
        if progress_callback:
            progress_callback(msg)

    # 收集文档
    docs = list_documents(docs_dir)
    if not docs:
        return "❌ 知识库目录中没有文档，请先上传"

    _log(f"找到 {len(docs)} 个文档")

    # 读取并分块
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "],
    )

    all_docs = []
    for doc_info in docs:
        _log(f"读取: {doc_info['name']}")
        try:
            text = Path(doc_info["path"]).read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            _log(f"⚠️ 读取失败: {doc_info['name']} — {e}")
            continue
        if not text.strip():
            continue
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            all_docs.append(Document(
                page_content=chunk,
                metadata={"source": doc_info["name"], "chunk": i}
            ))

    if not all_docs:
        return "❌ 文档内容为空，无法构建索引"

    _log(f"共 {len(all_docs)} 个文本块，开始构建索引...")

    # 构建 Embedding
    model_path = embedding_model or config.EMBEDDING_MODEL_PATH or "BAAI/bge-small-zh-v1.5"
    embeddings = _load_embeddings_safe(model_path, _log)
    if isinstance(embeddings, str):
        return embeddings  # 错误消息

    _log("Embedding 模型已加载，正在向量化...")

    _stop_flag.clear()

    # 分批构建（避免一次性内存过大）
    from langchain_community.vectorstores import FAISS
    batch_size = 100
    vectorstore = None
    for i in range(0, len(all_docs), batch_size):
        if _stop_flag.is_set():
            return "⏹️ 索引构建已停止"
        batch = all_docs[i:i + batch_size]
        _log(f"处理第 {i+1}-{min(i+batch_size, len(all_docs))} / {len(all_docs)} 块...")
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

    if vectorstore is None:
        return "❌ 索引构建失败"

    # 保存
    _log("保存索引...")
    vectorstore.save_local(str(index_dir))

    # 重置 chapter_summary 中的缓存单例
    _reset_kb_cache()

    total_chunks = len(all_docs)
    _log(f"✅ 索引构建完成！共 {total_chunks} 个文本块，来自 {len(docs)} 个文档")
    return f"✅ 索引构建完成！\n\n- 文档数: {len(docs)}\n- 文本块数: {total_chunks}\n- 索引路径: {index_dir}"


def _reset_kb_cache():
    """重置 chapter_summary 中的 lazy 单例，使新索引生效"""
    try:
        from text_tools import chapter_summary
        chapter_summary._embeddings = None
        chapter_summary._vector = None
        chapter_summary._bm25 = None
        chapter_summary._all_docs = []
        chapter_summary._all_texts = []
    except Exception:
        pass


# ==================== 索引状态 ====================

def get_index_stats(index_dir: Path = None) -> Dict:
    """获取索引统计信息"""
    index_dir = index_dir or get_index_dir()
    index_file = index_dir / "index.faiss"
    docstore_file = index_dir / "index.pkl"

    stats = {
        "exists": index_file.exists(),
        "path": str(index_dir),
        "docs_dir": str(get_docs_dir()),
        "doc_count": 0,
        "index_size": 0,
    }

    if index_file.exists():
        stats["index_size"] = index_file.stat().st_size
        # 尝试读取文档数
        try:
            model_path = config.EMBEDDING_MODEL_PATH or "BAAI/bge-small-zh-v1.5"
            embeddings = _load_embeddings_safe(model_path)
            if isinstance(embeddings, str):
                raise RuntimeError(embeddings)
            from langchain_community.vectorstores import FAISS
            vs = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
            stats["doc_count"] = len(vs.docstore._dict)
        except Exception:
            stats["doc_count"] = -1  # 无法读取

    docs = list_documents()
    stats["source_file_count"] = len(docs)
    stats["source_total_size"] = sum(d["size"] for d in docs)

    return stats


def get_index_stats_quick(index_dir: Path = None) -> Dict:
    """快速获取索引统计（不加载 Embedding 模型）"""
    index_dir = index_dir or get_index_dir()
    index_file = index_dir / "index.faiss"

    stats = {
        "exists": index_file.exists(),
        "path": str(index_dir),
        "docs_dir": str(get_docs_dir()),
        "index_size": index_file.stat().st_size if index_file.exists() else 0,
    }

    docs = list_documents()
    stats["source_file_count"] = len(docs)
    stats["source_total_size"] = sum(d["size"] for d in docs)

    return stats
