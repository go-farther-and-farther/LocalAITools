#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LocalAITools - Gradio Web 界面
启动方式：python app.py  或  双击 run.bat
然后浏览器打开 http://localhost:7860
"""

import sys
import io
import logging
from pathlib import Path
from contextlib import redirect_stdout

sys.path.insert(0, str(Path(__file__).parent))
import config

import gradio as gr


def _make_title(text):
    return f"## {text}"


def _capture_log(fn, *args, **kwargs):
    """捕获 logging 输出为字符串"""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        fn(*args, **kwargs)
    finally:
        root.removeHandler(handler)
    return stream.getvalue() or "✅ 处理完成"


# ============================================================
# Tab 1: 图片重命名
# ============================================================
def _rename_images(input_dir, model, workers, dry_run):
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
    recent_history = deque(maxlen=5)
    results = []
    total = len(images)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one_image, img, model or config.RENAME_MODEL, dry_run, recent_history): img
                   for img in images}
        for i, future in enumerate(as_completed(futures)):
            try:
                old_name, new_phrase = future.result()
                results.append(f"{old_name} → {new_phrase}")
            except Exception as e:
                results.append(f"❌ 错误: {e}")

    return f"处理完成：{total} 张图片\n\n" + "\n".join(results)


# ============================================================
# Tab 2: 图片质量评分与分类（合并原 Tab 2 + Tab 3）
# ============================================================
def _detect_and_classify(input_dir):
    from image_tools.detect_ai_errors import process_and_classify
    return _capture_log(process_and_classify, input_dir)


# ============================================================
# Tab 3: 聊天截图识别
# ============================================================
def _explain_images(input_dir, vision_model, temperature, workers, internal_workers, max_tokens):
    from image_tools.ocr_chat_screenshots import process_folder

    input_path = Path(input_dir)
    if not input_path.is_dir():
        return "❌ 请输入有效的文件夹路径"

    output_dir = input_path / "chat_text_output"
    output_dir.mkdir(exist_ok=True)

    buf = io.StringIO()
    with redirect_stdout(buf):
        process_folder(
            input_path,
            output_dir,
            vision_model or config.VISION_MODEL_THINKING,
            temperature,
            workers,
            max_tokens,
            internal_workers
        )

    return f"✅ 输出目录: {output_dir}\n\n{buf.getvalue()}"


# ============================================================
# Tab 4: 聊天记录压缩
# ============================================================
def _compress_text(input_path, model, temperature, chunk_size, internal_workers, max_tokens):
    from text_tools.compress_chat import process_single_text_file, process_folder

    p = Path(input_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        if p.is_file():
            process_single_text_file(
                p, None,
                model or config.VISION_MODEL_THINKING,
                temperature, max_tokens, chunk_size, internal_workers
            )
        elif p.is_dir():
            process_folder(
                p, None,
                model or config.VISION_MODEL_THINKING,
                temperature, max_tokens, chunk_size, internal_workers
            )
        else:
            return "❌ 路径无效"

    return f"✅ 处理完成\n\n{buf.getvalue()}"


# ============================================================
# Tab 5: 文本翻译
# ============================================================
def _translate(input_file, output_file, model, batch_size, workers):
    from text_tools.translate import translate_book_parallel

    if not Path(input_file).is_file():
        return "❌ 输入文件不存在"

    output_file = output_file or str(config.OUTPUT_DIR / "translation" / "translation.txt")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    buf = io.StringIO()
    with redirect_stdout(buf):
        translate_book_parallel(
            input_file, output_file,
            model or config.TEXT_MODEL,
            batch_size, workers,
            resume=True
        )

    return f"✅ 输出: {output_file}\n\n{buf.getvalue()}"


# ============================================================
# Tab 6: 知识库问答
# ============================================================
def _query_kb(query, keyword, model, k, batch_size):
    from text_tools.chapter_summary import query_knowledge_base

    if not query.strip():
        return "❌ 请输入查询问题"

    progress_lines = []

    def on_progress(msg: str):
        progress_lines.append(msg)

    answer = query_knowledge_base(
        query=query,
        keyword=keyword,
        model=model or None,
        k=int(k),
        batch_size=int(batch_size),
        progress_callback=on_progress,
    )

    progress_text = "\n".join(f"⏳ {l}" for l in progress_lines)
    return f"{progress_text}\n\n{answer}"


# ============================================================
# Tab 7: LLM 压测
# ============================================================
def _benchmark(url, model, api_key, concurrency, timeout, output_tokens, lengths_str):
    from benchmarks.speedtest import run_benchmark

    if lengths_str.strip():
        lengths = [int(x.strip()) for x in lengths_str.split(",")]
    else:
        lengths = [512, 1024, 2048, 4096]

    save_json = str(config.OUTPUT_DIR / "benchmarks" / "benchmark_results.json")
    save_plot = str(config.OUTPUT_DIR / "benchmarks" / "throughput_chart.png")

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
            save_plot=save_plot
        )

    text_output = buf.getvalue()
    plot_path = Path(save_plot)
    if plot_path.exists():
        return text_output, str(plot_path)
    return text_output, None


# ============================================================
# 构建 Gradio 界面
# ============================================================
def build_ui():
    css = """
    .output-text { font-size: 0.9em; max-height: 500px; overflow-y: auto; }
    .hint { font-size: 0.85em; color: #888; margin-top: -8px; margin-bottom: 12px; }
    footer { visibility: hidden; }
    """
    with gr.Blocks(title="LocalAITools", css=css, theme=gr.themes.Soft()) as app:
        gr.Markdown("# LocalAITools - 本地 AI 工具箱")
        gr.Markdown("""所有工具均调用本地大模型（兼容 OpenAI API），请在 `.env` 中配置 API 地址和密钥。

| Tab | 功能 | 需要什么模型 | 输入 → 输出 |
|-----|------|-------------|-------------|
| 🖼️ 图片重命名 | AI 看图起中文名 | 视觉模型 | 图片文件夹 → 文件重命名 |
| 🔍 质量评分分类 | 四维评分 + 自动分拣 | 视觉模型 | 图片文件夹 → HighQuality / LowQuality_Errors |
| 💬 截图识别 | 聊天截图 → 文字 | 视觉模型（Thinking 最佳） | 截图 → TXT 文件 |
| 📝 聊天压缩 | 合并冗余时间戳 | 文本模型 | TXT → 精简 TXT |
| 🌐 文本翻译 | 长篇章节翻译 | 文本模型 | TXT → 翻译 TXT |
| 📚 知识库问答 | RAG 混合检索 | 文本模型 + Embedding 模型 | 问题 → 答案 |
| ⚡ LLM 压测 | API 吞吐量测试 | 被测模型 | 参数 → 图表 |""")

        # ==================== Tab 1: 图片重命名 ====================
        with gr.Tab("🖼️ 图片重命名"):
            gr.Markdown(_make_title("图片 AI 重命名 — 用中文短句替代杂乱的文件名"))
            gr.Markdown("**使用步骤：** ① 把图片放进文件夹 → ② 点「开始重命名」→ ③ 文件名变成中文描述短语\n\n"
                        "> 💡 建议先勾选「试运行」预览效果，满意后再取消勾选正式改名。")
            with gr.Row():
                with gr.Column(scale=2):
                    rn_input = gr.Textbox(label="图片文件夹",
                                          value=str(config.DATA_DIR / "images"),
                                          placeholder="粘贴图片所在文件夹的完整路径",
                                          info="支持 .jpg / .jpeg / .png / .webp / .avif / .gif")
                    with gr.Accordion("⚙️ 高级设置", open=False):
                        rn_model = gr.Textbox(label="模型名称", value=config.RENAME_MODEL,
                                              info="需要视觉模型，能理解图片内容")
                        rn_workers = gr.Slider(1, 8, value=config.DEFAULT_WORKERS, step=1,
                                               label="并行线程数", info="越大越快，但可能触发 API 限流")
                    rn_dry = gr.Checkbox(label="试运行（只预览不实际改名）", value=False,
                                         info="强烈建议第一次使用时先试运行，看看效果")
                    rn_btn = gr.Button("开始重命名", variant="primary")
                with gr.Column(scale=3):
                    rn_output = gr.Textbox(label="处理结果", lines=15, elem_classes="output-text",
                                           placeholder="处理完成后这里会显示每张图片的新名字...")
            rn_btn.click(_rename_images, [rn_input, rn_model, rn_workers, rn_dry], [rn_output])

        # ==================== Tab 2: 图片质量评分与分类 ====================
        with gr.Tab("🔍 图片质量评分与分类"):
            gr.Markdown(_make_title("AI 评分 + 自动分拣 — 四维度评估，高分/低分图片自动归类"))
            gr.Markdown("**使用步骤：** ① 把图片放进文件夹 → ② 点「开始评分分类」→ ③ 优质图片自动移至 `HighQuality/`，劣质/错误图片移至 `LowQuality_Errors/`\n\n"
                        "**评分标准：** 真实感 / 艺术性 / 细节协调 / 清晰度，每项综合评分 0.0-10.0 分\n"
                        f"> 📊 前 **{config.TOP_PERCENT*100:.0f}%** 高分 → `{config.HIGH_QUALITY_FOLDER}/`，后 **{config.BOTTOM_PERCENT*100:.0f}%** 低分 + 错误 → `{config.LOW_QUALITY_ERRORS_FOLDER}/`\n"
                        "> 💡 每张图会生成同名 `.txt` 评分文件，方便核对")
            with gr.Row():
                with gr.Column(scale=2):
                    de_input = gr.Textbox(label="图片文件夹",
                                          value=str(config.DATA_DIR / "images"),
                                          placeholder="粘贴图片所在文件夹的完整路径",
                                          info="处理完成后会在该文件夹下生成 HighQuality/ 和 LowQuality_Errors/ 子文件夹")
                    de_btn = gr.Button("开始评分分类", variant="primary")
                with gr.Column(scale=3):
                    de_output = gr.Textbox(label="处理日志", lines=15, elem_classes="output-text",
                                           placeholder="处理完成后这里会显示每张图的评分和分类结果...")
            de_btn.click(_detect_and_classify, [de_input], [de_output])

        # ==================== Tab 3: 聊天截图识别 ====================
        with gr.Tab("💬 聊天截图识别"):
            gr.Markdown(_make_title("聊天记录长截图 → 文字提取 — 切片 + VLM 识别"))
            gr.Markdown("**使用步骤：** ① 把微信/QQ 聊天长截图放进文件夹 → ② 点「开始识别」→ ③ 去 `chat_text_output/` 找生成的 TXT 文件\n\n"
                        "> 💡 长图会被自动切成 2000px 高的薄片，每片之间有 400px 重叠防止漏字\n"
                        "> 💡 建议使用 **Thinking 模型**（如 qwen3.6-35b-a3b-Thinking），识别更准确")
            with gr.Row():
                with gr.Column(scale=2):
                    ei_input = gr.Textbox(label="截图文件夹",
                                          value=str(config.DATA_DIR / "screenshots"),
                                          placeholder="粘贴聊天截图所在文件夹的完整路径",
                                          info="支持 .jpg / .jpeg / .png / .bmp / .webp")
                    with gr.Accordion("⚙️ 高级设置", open=False):
                        ei_model = gr.Textbox(label="视觉模型", value=config.VISION_MODEL_THINKING,
                                              info="需要视觉模型，推荐带 Thinking 能力的模型")
                        ei_temp = gr.Slider(0.0, 1.0, value=0.3, step=0.1, label="温度",
                                            info="越低越稳定，越高越有创意。OCR 任务建议 0.2-0.4")
                        ei_workers = gr.Slider(1, 4, value=1, step=1, label="并行图片数",
                                               info="同时处理几张图。注意：过大可能导致显存不足")
                        ei_iworkers = gr.Slider(1, 4, value=2, step=1, label="单图内部并发",
                                                info="每张图内部分两半并行处理")
                        ei_maxtok = gr.Slider(1000, 8000, value=5000, step=500, label="最大 Token 数",
                                              info="输出上限，长截图可适当调大")
                    ei_btn = gr.Button("开始识别", variant="primary")
                with gr.Column(scale=3):
                    ei_output = gr.Textbox(label="处理日志", lines=15, elem_classes="output-text",
                                           placeholder="处理完成后这里会显示识别进度和结果...")
            ei_btn.click(_explain_images, [ei_input, ei_model, ei_temp, ei_workers, ei_iworkers, ei_maxtok], [ei_output])

        # ==================== Tab 4: 聊天记录压缩 ====================
        with gr.Tab("📝 聊天记录压缩"):
            gr.Markdown(_make_title("聊天记录 txt 文件 → 精简格式化"))
            gr.Markdown("**使用步骤：** ① 把截图 OCR 生成的 TXT（或任意聊天记录 TXT）放进文件夹 → ② 点「开始压缩」→ ③ 得到 `.compressed.txt` 文件\n\n"
                        "> 💡 压缩会合并连续相同说话人的时间戳，移除系统消息等冗余内容，保留所有对话实质\n"
                        "> 📎 **典型流程：** 截图 → Tab 3 识别 → 本 Tab 压缩 → 得到干净对话文本")
            with gr.Row():
                with gr.Column(scale=2):
                    ct_input = gr.Textbox(label="输入文件/文件夹",
                                          value=str(config.DATA_DIR / "screenshots" / "texts"),
                                          placeholder="粘贴聊天记录 TXT 文件或文件夹路径",
                                          info="可以是单个 .txt 文件，也可以是装多个 .txt 的文件夹")
                    with gr.Accordion("⚙️ 高级设置", open=False):
                        ct_model = gr.Textbox(label="文本模型", value=config.VISION_MODEL_THINKING,
                                              info="纯文本任务，用普通文本模型即可，不必用视觉模型")
                        ct_temp = gr.Slider(0.0, 0.5, value=0.2, step=0.05, label="温度",
                                            info="精简任务建议低温 0.1-0.3，保持稳定")
                        ct_chunk = gr.Slider(5000, 50000, value=config.DEFAULT_CHUNK_SIZE, step=1000,
                                             label="分块大小（字符）",
                                             info="每块给 LLM 处理的文字量。越大单次成本越高，但分段更连贯")
                        ct_iw = gr.Slider(1, 4, value=2, step=1, label="内部并发数",
                                          info="同时处理几块文本")
                        ct_maxtok = gr.Slider(1000, 8000, value=4000, step=500, label="最大 Token 数",
                                              info="输出长度上限")
                    ct_btn = gr.Button("开始压缩", variant="primary")
                with gr.Column(scale=3):
                    ct_output = gr.Textbox(label="处理日志", lines=15, elem_classes="output-text",
                                           placeholder="处理完成后这里会显示压缩进度...")
            ct_btn.click(_compress_text, [ct_input, ct_model, ct_temp, ct_chunk, ct_iw, ct_maxtok], [ct_output])

        # ==================== Tab 5: 文本翻译 ====================
        with gr.Tab("🌐 文本翻译"):
            gr.Markdown(_make_title("长篇文本翻译 — 按章节切分，断点续传"))
            gr.Markdown("**使用步骤：** ① 把要翻译的文本文件路径填好 → ② 点「开始翻译」→ ③ 去 `outputs/translation/` 找译文\n\n"
                        "> 💡 自动识别「第X章」「Chapter」等章节标记，按章节切分翻译\n"
                        "> 💡 支持**断点续传**：中途中断后重新点开始，会接着上次进度继续\n"
                        "> 💡 目标语言在 `.env` 中设置（默认 `English`）")
            with gr.Row():
                with gr.Column(scale=2):
                    tr_input = gr.Textbox(label="输入文件",
                                          value=str(config.DATA_DIR / "texts" / "input.txt"),
                                          placeholder="粘贴要翻译的 .txt 文件完整路径",
                                          info="文件编码需为 UTF-8")
                    tr_output_file = gr.Textbox(label="输出文件",
                                                value="",
                                                placeholder="留空则自动保存到 outputs/translation/translation.txt",
                                                info="指定译文保存路径，留空自动生成")
                    with gr.Accordion("⚙️ 高级设置", open=False):
                        tr_model = gr.Textbox(label="模型名称", value=config.TEXT_MODEL,
                                              info="纯文本翻译任务，使用文本模型")
                        tr_batch = gr.Slider(1, 20, value=10, step=1, label="每批章节数",
                                             info="每批同时翻译多少章。越大越快但可能超时")
                        tr_workers = gr.Slider(1, 4, value=2, step=1, label="并行线程数",
                                               info="同时运行几个翻译任务")
                    tr_btn = gr.Button("开始翻译", variant="primary")
                with gr.Column(scale=3):
                    tr_output = gr.Textbox(label="翻译日志", lines=15, elem_classes="output-text",
                                           placeholder="处理完成后这里会显示翻译进度、已翻章节数...")
            tr_btn.click(_translate, [tr_input, tr_output_file, tr_model, tr_batch, tr_workers], [tr_output])

        # ==================== Tab 6: 知识库问答 ====================
        with gr.Tab("📚 知识库问答"):
            gr.Markdown(_make_title("RAG 知识库问答 — FAISS + BM25 混合检索，多轮迭代回答"))
            gr.Markdown("**使用步骤：** ① 预先用文档构建好 FAISS 索引（`.env` 中配置 `FAISS_INDEX_PATH`）→ ② 输入问题 → ③ 点「开始查询」→ ④ LLM 基于检索到的相关段落多轮迭代生成答案\n\n"
                        "> 🔍 **混合检索：** 向量相似度 + BM25 关键词匹配，比纯向量检索更准\n"
                        "> 🔄 **多轮迭代：** 查一批 → 回答 → 带着上一轮答案查下一批 → 不断修正完善\n"
                        "> ⚠️ 首次查询需要加载 Embedding 模型和索引，可能等待几十秒\n"
                        "> 💡 关键词过滤选填：如果你知道答案一定包含某个词，填上可大幅提高精度")
            with gr.Row():
                with gr.Column(scale=2):
                    kb_query = gr.Textbox(label="查询问题",
                                          placeholder="输入你想问的问题...",
                                          lines=2,
                                          info="问题越具体，答案越准确")
                    kb_keyword = gr.Textbox(label="关键词过滤（可选）",
                                            placeholder="留空则不按关键词过滤。如填「白雨珺」则只检索包含该词的段落",
                                            value="",
                                            info="过滤掉不包含该关键词的段落，能显著提高答案精度")
                    with gr.Accordion("⚙️ 高级设置", open=False):
                        kb_model = gr.Textbox(label="模型名称", value=config.TEXT_MODEL,
                                              info="用于最终回答生成的文本模型")
                        kb_k = gr.Slider(10, 200, value=50, step=10, label="检索片段数",
                                         info="从知识库中检索多少个相关段落。越多越全面但越慢")
                        kb_batch = gr.Slider(5, 50, value=20, step=5, label="每批处理数",
                                              info="每轮迭代喂给 LLM 的段落数。越小迭代轮数越多但答案越精细")
                    kb_btn = gr.Button("开始查询", variant="primary")
                with gr.Column(scale=3):
                    kb_output = gr.Textbox(label="回答结果", lines=18, elem_classes="output-text",
                                           placeholder="查询结果会显示在这里，包括进度和最终答案...")
            kb_btn.click(_query_kb, [kb_query, kb_keyword, kb_model, kb_k, kb_batch], [kb_output])

        # ==================== Tab 7: LLM 压测 ====================
        with gr.Tab("⚡ LLM 压测"):
            gr.Markdown(_make_title("LLM 吞吐量压测 — 测试 API 性能并生成图表"))
            gr.Markdown("**使用步骤：** ① 确认 API 地址和模型名称 → ② 点「开始压测」→ ③ 查看吞吐量图表，找到模型性能瓶颈\n\n"
                        "> 📊 **测量指标：** TTFT（首 Token 延迟）、ITTL（Token 间延迟）、Prefill 吞吐量、Decode 吞吐量\n"
                        "> ⏱️ 压测会占用模型全部资源，建议在空闲时运行\n"
                        "> 📈 结果自动保存到 `outputs/benchmarks/`")
            with gr.Row():
                with gr.Column(scale=2):
                    bm_url = gr.Textbox(label="API Base URL", value=config.OPENAI_BASE_URL,
                                        info="被测 API 地址，如 http://localhost:1234/v1")
                    bm_model = gr.Textbox(label="模型名称", value=config.BENCHMARK_MODEL,
                                          info="被测模型，需与 API 中实际名称一致")
                    bm_key = gr.Textbox(label="API Key", value=config.OPENAI_API_KEY, type="password",
                                        info="本地服务如 LM Studio 填任意值即可")
                    with gr.Accordion("⚙️ 高级设置", open=False):
                        bm_concur = gr.Slider(1, 8, value=config.DEFAULT_WORKERS, step=1, label="并发数",
                                              info="同时发几个请求。越大越能测出吞吐上限，但可能超时")
                        bm_timeout = gr.Slider(10, 120, value=config.REQUEST_TIMEOUT_SHORT, step=5, label="超时（秒）",
                                               info="单个请求超时时间")
                        bm_outtok = gr.Slider(64, 2048, value=512, step=64, label="每次生成 Token 数",
                                              info="每次请求让模型生成多少 token")
                        bm_lengths = gr.Textbox(label="测试长度列表（逗号分隔）",
                                                value="512,1024,2048,4096",
                                                info="不同输入长度分别测试，逗号分隔")
                    bm_btn = gr.Button("开始压测", variant="primary")
                with gr.Column(scale=3):
                    bm_text = gr.Textbox(label="压测日志", lines=10, elem_classes="output-text",
                                         placeholder="压测完成后这里会显示各长度的 TTFT、ITTL、吞吐量数据...")
                    bm_plot = gr.Image(label="吞吐量图表")
            bm_btn.click(_benchmark, [bm_url, bm_model, bm_key, bm_concur, bm_timeout, bm_outtok, bm_lengths],
                        [bm_text, bm_plot])

    return app


if __name__ == "__main__":
    import traceback
    try:
        build_ui().launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
    except Exception:
        print("\n❌ 启动失败：\n")
        traceback.print_exc()
        print("\n按 Enter 键退出...")
        input()
