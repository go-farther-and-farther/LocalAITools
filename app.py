#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LocalAITools - Gradio Web 界面
启动方式：python app.py
然后浏览器打开 http://localhost:7860
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config

import gradio as gr


def _make_title(text):
    return f"## {text}"


# ============================================================
# Tab 1: 图片重命名
# ============================================================
def _rename_images(input_dir, model, workers, dry_run):
    from image_tools.rename_images import process_one_image, get_shared_llm
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os

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
# Tab 2: 图片质量检测
# ============================================================
def _detect_errors(input_dir):
    from image_tools.detect_ai_errors import process_and_classify
    import io
    import logging

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
    logging.getLogger().addHandler(handler)

    try:
        process_and_classify(input_dir)
    finally:
        logging.getLogger().removeHandler(handler)

    return log_stream.getvalue() or "✅ 处理完成"


# ============================================================
# Tab 3: 图片重新分类
# ============================================================
def _reclassify(input_dir):
    from image_tools.reclassify_by_txt import process_directory
    import io
    import logging

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
    logging.getLogger().addHandler(handler)

    try:
        process_directory(input_dir)
    finally:
        logging.getLogger().removeHandler(handler)

    return log_stream.getvalue() or "✅ 重新分类完成"


# ============================================================
# Tab 4: 聊天截图识别
# ============================================================
def _explain_images(input_dir, vision_model, temperature, workers, internal_workers, max_tokens):
    from image_tools.explain_images_txt import process_folder
    import io
    import sys as _sys
    from contextlib import redirect_stdout

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
# Tab 5: 聊天记录压缩
# ============================================================
def _compress_text(input_path, model, temperature, chunk_size, internal_workers, max_tokens):
    from text_tools.explain_txt import process_single_text_file, process_folder
    import io
    from contextlib import redirect_stdout

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
# Tab 6: 文本翻译
# ============================================================
def _translate(input_file, output_file, model, batch_size, workers):
    from text_tools.translate import translate_book_parallel

    if not Path(input_file).is_file():
        return "❌ 输入文件不存在"

    output_file = output_file or str(config.OUTPUT_DIR / "translation" / "translation.txt")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    import io
    from contextlib import redirect_stdout

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
# Tab 7: LLM 压测
# ============================================================
def _benchmark(url, model, api_key, concurrency, timeout, output_tokens, lengths_str):
    from benchmarks.speedtest import run_benchmark
    import io
    from contextlib import redirect_stdout

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

    # Return text + plot image
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
    footer { visibility: hidden; }
    """
    with gr.Blocks(title="LocalAITools", css=css, theme=gr.themes.Soft()) as app:
        gr.Markdown("# LocalAITools - 本地 AI 工具箱")
        gr.Markdown("所有工具均调用本地大模型（兼容 OpenAI API），请在 `.env` 中配置 API 地址和密钥。")

        with gr.Tab("🖼️ 图片重命名"):
            gr.Markdown(_make_title("图片 AI 重命名 — 用中文短句替代杂乱的文件名"))
            with gr.Row():
                with gr.Column(scale=2):
                    rn_input = gr.Textbox(label="图片文件夹", value=str(config.DATA_DIR / "images"))
                    rn_model = gr.Textbox(label="模型名称", value=config.RENAME_MODEL)
                    rn_workers = gr.Slider(1, 8, value=config.DEFAULT_WORKERS, step=1, label="并行线程数")
                    rn_dry = gr.Checkbox(label="试运行（不实际改名）", value=False)
                    rn_btn = gr.Button("开始重命名", variant="primary")
                with gr.Column(scale=3):
                    rn_output = gr.Textbox(label="处理结果", lines=15, elem_classes="output-text")
            rn_btn.click(_rename_images, [rn_input, rn_model, rn_workers, rn_dry], [rn_output])

        with gr.Tab("🔍 图片质量检测"):
            gr.Markdown(_make_title("AI 图片质量评分 — 自动评分 + 错误检测"))
            with gr.Row():
                with gr.Column(scale=2):
                    de_input = gr.Textbox(label="图片文件夹", value=str(config.DATA_DIR / "images"))
                    de_btn = gr.Button("开始检测", variant="primary")
                with gr.Column(scale=3):
                    de_output = gr.Textbox(label="处理日志", lines=15, elem_classes="output-text")
            de_btn.click(_detect_errors, [de_input], [de_output])

        with gr.Tab("📂 图片重新分类"):
            gr.Markdown(_make_title("按评分重新分类图片 — 将高质量/低质量图片分入子文件夹"))
            with gr.Row():
                with gr.Column(scale=2):
                    rc_input = gr.Textbox(label="图片文件夹", value=str(config.DATA_DIR / "images"))
                    rc_btn = gr.Button("开始分类", variant="primary")
                with gr.Column(scale=3):
                    rc_output = gr.Textbox(label="处理日志", lines=15, elem_classes="output-text")
            rc_btn.click(_reclassify, [rc_input], [rc_output])

        with gr.Tab("💬 聊天截图识别"):
            gr.Markdown(_make_title("聊天记录长截图 → 文字提取 — 切片 + VLM 识别"))
            with gr.Row():
                with gr.Column(scale=2):
                    ei_input = gr.Textbox(label="截图文件夹", value=str(config.DATA_DIR / "screenshots"))
                    ei_model = gr.Textbox(label="视觉模型", value=config.VISION_MODEL_THINKING)
                    ei_temp = gr.Slider(0.0, 1.0, value=0.3, step=0.1, label="温度")
                    ei_workers = gr.Slider(1, 4, value=1, step=1, label="并行图片数")
                    ei_iworkers = gr.Slider(1, 4, value=2, step=1, label="单图内部并发")
                    ei_maxtok = gr.Slider(1000, 8000, value=5000, step=500, label="最大 Token 数")
                    ei_btn = gr.Button("开始识别", variant="primary")
                with gr.Column(scale=3):
                    ei_output = gr.Textbox(label="处理日志", lines=15, elem_classes="output-text")
            ei_btn.click(_explain_images, [ei_input, ei_model, ei_temp, ei_workers, ei_iworkers, ei_maxtok], [ei_output])

        with gr.Tab("📝 聊天记录压缩"):
            gr.Markdown(_make_title("聊天记录 txt 文件 → 精简格式化"))
            with gr.Row():
                with gr.Column(scale=2):
                    ct_input = gr.Textbox(label="输入文件/文件夹", value=str(config.DATA_DIR / "screenshots" / "texts"))
                    ct_model = gr.Textbox(label="文本模型", value=config.VISION_MODEL_THINKING)
                    ct_temp = gr.Slider(0.0, 0.5, value=0.2, step=0.05, label="温度")
                    ct_chunk = gr.Slider(5000, 50000, value=config.DEFAULT_CHUNK_SIZE, step=1000, label="分块大小（字符）")
                    ct_iw = gr.Slider(1, 4, value=2, step=1, label="内部并发数")
                    ct_maxtok = gr.Slider(1000, 8000, value=4000, step=500, label="最大 Token 数")
                    ct_btn = gr.Button("开始压缩", variant="primary")
                with gr.Column(scale=3):
                    ct_output = gr.Textbox(label="处理日志", lines=15, elem_classes="output-text")
            ct_btn.click(_compress_text, [ct_input, ct_model, ct_temp, ct_chunk, ct_iw, ct_maxtok], [ct_output])

        with gr.Tab("🌐 文本翻译"):
            gr.Markdown(_make_title("长篇文本翻译 — 按章节切分，断点续传"))
            with gr.Row():
                with gr.Column(scale=2):
                    tr_input = gr.Textbox(label="输入文件", value=str(config.DATA_DIR / "texts" / "input.txt"))
                    tr_output_file = gr.Textbox(label="输出文件（留空自动生成）", value="")
                    tr_model = gr.Textbox(label="模型名称", value=config.TEXT_MODEL)
                    tr_batch = gr.Slider(1, 20, value=10, step=1, label="每批章节数")
                    tr_workers = gr.Slider(1, 4, value=2, step=1, label="并行线程数")
                    tr_btn = gr.Button("开始翻译", variant="primary")
                with gr.Column(scale=3):
                    tr_output = gr.Textbox(label="翻译日志", lines=15, elem_classes="output-text")
            tr_btn.click(_translate, [tr_input, tr_output_file, tr_model, tr_batch, tr_workers], [tr_output])

        with gr.Tab("⚡ LLM 压测"):
            gr.Markdown(_make_title("LLM 吞吐量压测 — 测试 API 性能并生成图表"))
            with gr.Row():
                with gr.Column(scale=2):
                    bm_url = gr.Textbox(label="API Base URL", value=config.OPENAI_BASE_URL)
                    bm_model = gr.Textbox(label="模型名称", value=config.BENCHMARK_MODEL)
                    bm_key = gr.Textbox(label="API Key", value=config.OPENAI_API_KEY, type="password")
                    bm_concur = gr.Slider(1, 8, value=config.DEFAULT_WORKERS, step=1, label="并发数")
                    bm_timeout = gr.Slider(10, 120, value=config.REQUEST_TIMEOUT_SHORT, step=5, label="超时（秒）")
                    bm_outtok = gr.Slider(64, 2048, value=512, step=64, label="每次生成 Token 数")
                    bm_lengths = gr.Textbox(label="测试长度列表（逗号分隔）", value="512,1024,2048,4096")
                    bm_btn = gr.Button("开始压测", variant="primary")
                with gr.Column(scale=3):
                    bm_text = gr.Textbox(label="压测日志", lines=10, elem_classes="output-text")
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
