import os
import sys
from pathlib import Path
from langchain_core.prompts import PromptTemplate
import jieba
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from typing import List
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from contextlib import redirect_stdout
from datetime import datetime

# ================== 准备工作：加载向量库 & 构建 BM25 ==================
model_path = config.EMBEDDING_MODEL_PATH if config.EMBEDDING_MODEL_PATH else "BAAI/bge-small-zh-v1.5"

embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
    cache_folder=str(config.DATA_DIR / "models")
)

vector = FAISS.load_local(config.FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

all_docs = list(vector.docstore._dict.values())
all_texts = [doc.page_content for doc in all_docs]
tokenized_corpus = [list(jieba.cut(text)) for text in all_texts]
bm25 = BM25Okapi(tokenized_corpus)

print(f"✅ BM25 索引已构建，共 {len(all_texts)} 个文档块。")

def normalize_scores(scores: List[float]) -> List[float]:
    """Min-Max 归一化"""
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]

def hybrid_retrieve(query: str, k: int = 50, vector_weight: float = 1, bm25_weight: float = 1) -> List:
    """
    混合检索，返回加权排序后的 Document 列表
    """
    # 1. 向量检索
    vector_results_with_scores = vector.similarity_search_with_score(query, k=k*2)
    vector_docs = [doc for doc, _ in vector_results_with_scores]
    vector_scores = [1 / (1 + score) for _, score in vector_results_with_scores]

    # 2. BM25 检索
    tokenized_query = list(jieba.cut(query))
    bm25_raw_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_raw_scores)[::-1][:k*2]
    bm25_docs = [all_docs[i] for i in top_bm25_indices]
    bm25_scores = [bm25_raw_scores[i] for i in top_bm25_indices]

    # 3. 归一化
    norm_vector_scores = normalize_scores(vector_scores)
    norm_bm25_scores = normalize_scores(bm25_scores)

    # 4. 加权合并
    doc_score_map = {}
    for doc, score in zip(vector_docs, norm_vector_scores):
        content = doc.page_content
        weighted = score * vector_weight
        if content not in doc_score_map:
            doc_score_map[content] = {"doc": doc, "score": weighted}
        else:
            doc_score_map[content]["score"] += weighted

    for doc, score in zip(bm25_docs, norm_bm25_scores):
        content = doc.page_content
        weighted = score * bm25_weight
        if content not in doc_score_map:
            doc_score_map[content] = {"doc": doc, "score": weighted}
        else:
            doc_score_map[content]["score"] += weighted

    sorted_items = sorted(doc_score_map.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_items[:k]]

# ================== 定义 LLM ==================
llm = ChatOpenAI(
    model=config.TEXT_MODEL,
    streaming=True,
    base_url=config.OPENAI_BASE_URL,
    api_key=config.OPENAI_API_KEY
)

# ================== 定义两套 Prompt 模板 ==================
# 第一轮模板（无历史答案）
prompt_template_round1 = """
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

# 第二轮及以后模板（含上一轮答案）
prompt_template_roundN = """
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

# ================== 主流程：分批迭代回答 ==================
query = "白雨珺有哪些朋友？"
keyword = "白雨珺"

# 1. 混合检索获取 300 个片段（用于分成三批）
all_retrieved_docs = hybrid_retrieve(query, k=50)
print(f"📚 混合检索共获取 {len(all_retrieved_docs)} 个文档片段")

# 2. 过滤包含关键词的文档（可选，确保相关性）
filtered_docs = [doc for doc in all_retrieved_docs if keyword in doc.page_content]
print(f"🎯 其中包含「{keyword}」的片段数：{len(filtered_docs)}")

# 3. 按每批 100 个进行切片
batch_size = 20
batches = [filtered_docs[i:i+batch_size] for i in range(0, len(filtered_docs), batch_size)]
# if len(batches) > 3:
#     batches = batches[:3]   # 只取前三批（保证正好三次迭代）
print(f"🔁 将分 {len(batches)} 轮进行迭代处理，每批最多 {batch_size} 个片段")

# 4. 存储每轮的回答
answers = []

# 5. 迭代处理每一批
for round_idx, batch_docs in enumerate(batches, start=1):
    print(f"\n{'='*50}")
    print(f"第 {round_idx} 轮处理，当前批次文档数：{len(batch_docs)}")

    # 将本批文档内容拼接为字符串（每个片段之间用换行分隔）
    info_text = "\n\n".join([doc.page_content for doc in batch_docs])

    if round_idx == 1:
        # 第一轮：使用基础模板
        prompt = PromptTemplate.from_template(prompt_template_round1).format(
            info=info_text,
            question=query
        )
    else:
        # 后续轮次：将上一轮的回答作为重要参考
        previous = answers[-1]  # 上一轮的完整回答
        prompt = PromptTemplate.from_template(prompt_template_roundN).format(
            previous_answer=previous,
            info=info_text,
            question=query
        )

    # 调用 LLM
    response = llm.invoke(prompt)
    answer = response.content
    answers.append(answer)

    print(f"✅ 第 {round_idx} 轮回答生成完毕")
    print(f"📝 回答预览（前300字）：\n{answer[:300]}...")

# ================== 输出最终三个版本 ==================
print("\n" + "="*50)
print("📌 多轮迭代回答汇总：")
for i, ans in enumerate(answers, 1):
    print(f"\n--- 版本 {i} ---")
    print(ans)

output_dir = str(config.OUTPUT_DIR / "summaries")
os.makedirs(output_dir, exist_ok=True)
filename = os.path.join(output_dir, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# 提示保存成功（不写入文件，而是输出到控制台）
print(f"✅ 内容已保存至：{filename}", file=sys.stderr)