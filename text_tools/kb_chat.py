import os
import sys
import threading
from pathlib import Path
from typing import List, Optional

import jieba
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

_stop_flag = threading.Event()

def request_stop():
    """请求停止当前正在执行的知识库查询任务"""
    _stop_flag.set()

from langchain_core.prompts import PromptTemplate
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# ---- lazy singletons ----
_embeddings = None
_vector = None
_bm25 = None
_all_docs: List = []
_all_texts: List[str] = []


def _init_kb():
    global _embeddings, _vector, _bm25, _all_docs, _all_texts
    if _vector is not None:
        return

    model_path = config.EMBEDDING_MODEL_PATH or "BAAI/bge-small-zh-v1.5"
    _embeddings = _load_embeddings(model_path)
    _vector = FAISS.load_local(
        config.FAISS_INDEX_PATH, _embeddings, allow_dangerous_deserialization=True
    )
    _all_docs = list(_vector.docstore._dict.values())
    _all_texts = [doc.page_content for doc in _all_docs]
    tokenized_corpus = [list(jieba.cut(text)) for text in _all_texts]
    _bm25 = BM25Okapi(tokenized_corpus)


HF_MIRROR = "https://hf-mirror.com"


def _get_device() -> str:
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


def _load_embeddings(model_path: str) -> HuggingFaceEmbeddings:
    """加载 Embedding 模型，失败时自动切换 HuggingFace 镜像重试"""
    device = _get_device()
    print(f"📐 Embedding 设备: {device}")
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
            print(f"⚠️ HuggingFace 连接失败，自动切换镜像: {HF_MIRROR}")
            os.environ["HF_ENDPOINT"] = HF_MIRROR
            return HuggingFaceEmbeddings(**kwargs)
        raise


def _normalize_scores(scores: List[float]) -> List[float]:
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def _hybrid_retrieve(
    query: str, k: int = 50, vector_weight: float = 1, bm25_weight: float = 1
) -> List:
    _init_kb()
    # vector
    vec_results = _vector.similarity_search_with_score(query, k=k * 2)
    vec_docs = [doc for doc, _ in vec_results]
    vec_scores = [1 / (1 + score) for _, score in vec_results]
    # bm25
    tokenized = list(jieba.cut(query))
    bm25_raw = _bm25.get_scores(tokenized)
    top_bm25_idx = np.argsort(bm25_raw)[::-1][: k * 2]
    bm25_docs = [_all_docs[i] for i in top_bm25_idx]
    bm25_scores = [bm25_raw[i] for i in top_bm25_idx]
    # merge
    nvec = _normalize_scores(vec_scores)
    nbm25 = _normalize_scores(bm25_scores)
    score_map: dict = {}
    for doc, s in zip(vec_docs, nvec):
        c = doc.page_content
        entry = score_map.setdefault(c, {"doc": doc, "score": 0.0})
        entry["score"] += s * vector_weight
    for doc, s in zip(bm25_docs, nbm25):
        c = doc.page_content
        entry = score_map.setdefault(c, {"doc": doc, "score": 0.0})
        entry["score"] += s * bm25_weight
    sorted_items = sorted(score_map.values(), key=lambda x: x["score"], reverse=True)
    return [it["doc"] for it in sorted_items[:k]]


PROMPT_R1 = """
你是一个问答机器人。
你的任务是根据下述给定的已知信息详细地回答用户问题。
确保你的回复完全依据下述已知信息。不要编造答案。
如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。

已知信息:
{info}

用户问：
{question}

请用中文回答用户问题。
"""

PROMPT_RN = """
你是一个问答机器人。
你的任务是根据下述给定的已知信息，并结合之前你已经给出的分析结果，进一步详细、准确地回答用户问题。
请仔细参考上一轮你的回答，利用新增的已知信息进行补充、修正或深化。
如果新增信息与上一轮回答存在矛盾，以新增信息为准并修正之前的结论。
不要编造答案，一切以已知信息和上一轮回答为依据。

【上一轮你的回答】
{previous_answer}

【新增的已知信息】
{info}

用户问：
{question}

请用中文回答用户问题。
"""

PROMPT_CHAT = """
你是一个知识库问答助手。根据检索到的相关文档内容回答用户问题。

要求：
- 回答完全依据下方已知信息，不要编造
- 如果信息不足，坦诚说明
- 结合对话历史理解上下文和指代关系
- 回答简洁准确，重点突出

{chat_history}

【已知信息】
{info}

用户问：
{question}
"""


def query_knowledge_base(
    query: str,
    keyword: str = "",
    model: Optional[str] = None,
    k: int = 50,
    batch_size: int = 20,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
    progress_callback=None,
) -> str:
    """混合检索 + 多轮迭代回答。progress_callback(msg: str) 用于推送进度。"""
    _init_kb()

    def _log(msg: str):
        if progress_callback:
            progress_callback(msg)

    _log(f"混合检索中... (共 {len(_all_texts)} 个文档块)")
    docs = _hybrid_retrieve(query, k=k, vector_weight=vector_weight, bm25_weight=bm25_weight)
    _log(f"检索到 {len(docs)} 个片段")

    if keyword.strip():
        docs = [d for d in docs if keyword.strip() in d.page_content]
        _log(f"关键词「{keyword}」过滤后剩余 {len(docs)} 个片段")

    if not docs:
        return "未找到相关文档片段，请调整查询或关键词。"

    batches = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]
    _log(f"分 {len(batches)} 轮迭代处理")

    llm = ChatOpenAI(
        model=model or config.TEXT_MODEL,
        streaming=True,
        base_url=config.OPENAI_BASE_URL,
        api_key=config.OPENAI_API_KEY,
        extra_body=config.get_llm_extra_body()
    )

    _stop_flag.clear()
    answers: List[str] = []
    for round_idx, batch in enumerate(batches, start=1):
        if _stop_flag.is_set():
            _log("⏹️ 已请求停止查询")
            break
        info_text = "\n\n".join(d.page_content for d in batch)
        if round_idx == 1:
            prompt = PromptTemplate.from_template(PROMPT_R1).format(
                info=info_text, question=query
            )
        else:
            prompt = PromptTemplate.from_template(PROMPT_RN).format(
                previous_answer=answers[-1], info=info_text, question=query
            )
        _log(f"第 {round_idx}/{len(batches)} 轮 LLM 调用...")
        answers.append(llm.invoke(prompt).content)

    # assemble output
    if not answers:
        return "查询已被停止，未生成任何回答。"
    lines = [f"## 最终回答\n\n{answers[-1]}"]
    if len(answers) > 1:
        lines.append("\n---\n## 各轮迭代记录")
        for i, ans in enumerate(answers, 1):
            lines.append(f"\n### 第 {i} 轮\n{ans[:500]}{'...' if len(ans) > 500 else ''}")
    return "\n\n".join(lines)


def prepare_kb_context(
    query: str,
    keyword: str = "",
    k: int = 50,
    batch_size: int = 20,
    chat_history: Optional[List[dict]] = None,
    web_context: Optional[str] = None,
    progress_callback=None,
):
    """准备知识库上下文（检索 + 构建prompt），不调用LLM。
    返回 (prompt_str, None) 用于 prompt 模式，或 (None, lc_messages) 用于消息模式。"""
    _init_kb()

    def _log(msg: str):
        if progress_callback:
            progress_callback(msg)

    _log(f"混合检索中... (共 {len(_all_texts)} 个文档块)")
    docs = _hybrid_retrieve(query, k=k)
    _log(f"检索到 {len(docs)} 个片段")

    if keyword.strip():
        docs = [d for d in docs if keyword.strip() in d.page_content]
        _log(f"关键词「{keyword}」过滤后剩余 {len(docs)} 个片段")

    if not docs:
        _log("未检索到相关片段，切换为通用对话模式..." + (" (含网络搜索结果)" if web_context else ""))
        lc_msgs = []
        if web_context:
            from langchain_core.messages import SystemMessage
            lc_msgs.append(SystemMessage(content=f"以下是联网搜索的最新结果，请基于这些信息回答用户问题：\n\n{web_context}"))
        for m in (chat_history or []):
            if m["role"] == "user":
                from langchain_core.messages import HumanMessage
                lc_msgs.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                from langchain_core.messages import AIMessage
                lc_msgs.append(AIMessage(content=m["content"]))
        from langchain_core.messages import HumanMessage
        lc_msgs.append(HumanMessage(content=query.strip()))
        return None, lc_msgs

    # 构建对话历史文本
    history_lines = []
    if chat_history:
        for msg in chat_history[-10:]:
            role = "用户" if msg["role"] == "user" else "助手"
            history_lines.append(f"{role}：{msg['content']}")
    chat_history_text = "\n".join(history_lines) if history_lines else "（无历史对话）"

    # 合并所有检索片段（含网络搜索结果）
    all_info = "\n\n".join(d.page_content for d in docs[:batch_size * 2])
    if web_context:
        all_info = f"【联网搜索结果】\n{web_context}\n\n【知识库检索结果】\n{all_info}"

    _log("正在生成回答..." + (" (含网络搜索)" if web_context else ""))
    prompt = PromptTemplate.from_template(PROMPT_CHAT).format(
        chat_history=chat_history_text,
        info=all_info,
        question=query,
    )
    return prompt, None


def query_knowledge_base_chat(
    query: str,
    keyword: str = "",
    model: Optional[str] = None,
    k: int = 50,
    batch_size: int = 20,
    chat_history: Optional[List[dict]] = None,
    progress_callback=None,
    web_context: Optional[str] = None,
) -> str:
    """多轮对话式知识库问答（阻塞版，内部使用 prepare_kb_context + LLM）"""
    prompt, messages = prepare_kb_context(
        query=query, keyword=keyword, k=k, batch_size=batch_size,
        chat_history=chat_history, web_context=web_context,
        progress_callback=progress_callback,
    )
    llm = ChatOpenAI(
        model=model or config.TEXT_MODEL,
        streaming=True,
        base_url=config.OPENAI_BASE_URL,
        api_key=config.OPENAI_API_KEY,
        extra_body=config.get_llm_extra_body(),
    )
    _stop_flag.clear()
    try:
        if prompt is not None:
            return llm.invoke(prompt).content
        else:
            return llm.invoke(messages).content
    except Exception as e:
        return f"❌ 调用失败: {e}"


# ==================== CLI ====================
if __name__ == "__main__":
    result = query_knowledge_base(
        query=sys.argv[1] if len(sys.argv) > 1 else "白雨珺有哪些朋友？",
        keyword=sys.argv[2] if len(sys.argv) > 2 else "白雨珺",
    )
    print(result)
    out_dir = config.OUTPUT_DIR / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime

    fname = out_dir / f"kb_answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    fname.write_text(result, encoding="utf-8")
    print(f"\n已保存至: {fname}")
