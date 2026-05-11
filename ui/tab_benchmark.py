"""LLM benchmark tab."""
import logging
import threading
import queue
import time as _time
import gradio as gr
import config
import history
from services import benchmark_services
from ui.common import _make_title, _make_model_selector, _bind_model_fetch_local

logger = logging.getLogger("LocalAITools")


def _benchmark(url, model, api_key, concurrency, timeout, output_tokens, lengths_str):
    """Generator that runs the benchmark via the service layer and yields progress updates."""
    logger.info(f"[LLM压测] UI 触发 URL={url} 模型={model} 并发={concurrency} 长度={lengths_str}")

    result_queue = queue.Queue()

    def _run():
        try:
            text = benchmark_services.run_benchmark(
                url=url, model=model, api_key=api_key,
                concurrency=concurrency, timeout=timeout,
                output_tokens=output_tokens, lengths_str=lengths_str,
            )
            result_queue.put(text)
        except Exception as e:
            result_queue.put(f"\n❌ 压测出错: {e}\n")

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    lengths = [x.strip() for x in lengths_str.split(",") if x.strip()] or ["512", "1024", "2048", "4096"]
    last_text = "⏳ 压测启动中..."
    while t.is_alive():
        yield last_text, None
        last_text = f"⏳ 压测进行中...\n（{len(lengths)} 个长度 × {concurrency} 并发，完成后自动显示结果）"
        _time.sleep(0.8)

    text_output = result_queue.get()

    config.save_state("benchmark", url=url, model=model, api_key=api_key,
                      concurrency=concurrency, timeout=timeout,
                      output_tokens=output_tokens, lengths_str=lengths_str)
    history.add_entry("LLM压测", model or url, "压测完成")

    plot_path = benchmark_services.get_plot_path()
    yield text_output, plot_path


def render_tab_benchmark(s):
    """Render the LLM benchmark tab. Returns component dict."""
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

    # ---- 历史记录 ----
    gr.Markdown("---")
    gr.Markdown(_make_title("历史记录"))
    with gr.Row():
        bm_hist_btn = gr.Button("加载历史", variant="secondary")
        bm_csv_btn = gr.Button("导出 CSV", variant="secondary")
    bm_hist_status = gr.Textbox(visible=False, max_lines=1)
    bm_hist_table = gr.Dataframe(
        headers=["时间", "模型", "tokens/s", "总token数", "耗时(秒)", "Prompt长度"],
        label="最近 20 次压测记录",
        interactive=False,
        wrap=True,
    )

    # ---- 事件绑定 ----
    bm_evt = bm_btn.click(_benchmark, [bm_url, bm_model, bm_key, bm_concur, bm_timeout, bm_outtok, bm_lengths],
                          [bm_text, bm_plot])
    # Auto-refresh history table after benchmark completes
    bm_evt.then(benchmark_services.get_history_rows, outputs=[bm_hist_table])

    def _stop_benchmark():
        from benchmarks.speedtest import request_stop
        request_stop()
        return "⏹️ 已请求停止..."

    bm_stop.click(_stop_benchmark, outputs=[bm_text], cancels=[bm_evt])
    _bind_model_fetch_local(bm_fetch_btn, bm_model_type, bm_model, bm_fetch_st,
                            bm_url, bm_key, s["benchmark"].get("model", config.BENCHMARK_MODEL))

    bm_hist_btn.click(benchmark_services.get_history_rows, outputs=[bm_hist_table])

    def _do_export_csv():
        path = benchmark_services.export_bench_csv()
        if path:
            return gr.update(value=f"已导出: {path}")
        return gr.update(value="暂无历史记录可导出")

    bm_csv_btn.click(_do_export_csv, outputs=[bm_hist_status])

    return {"bm_model": bm_model, "bm_model_type": bm_model_type, "bm_url": bm_url, "bm_key": bm_key}
