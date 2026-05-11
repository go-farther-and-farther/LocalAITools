"""Text tools tab: chat compression and translation."""
import logging
from pathlib import Path
import config
import history
import gradio as gr
from ui.common import _make_title, _apply_provider, _make_model_selector, _bind_model_fetch
from services.text_services import compress_text, translate_text, get_chapter_progress as _svc_get_chapter_progress

logger = logging.getLogger("LocalAITools")


# ============================================================
# Thin wrappers — call service layer, then save state / history
# ============================================================

def _compress_text(input_path, model, temperature, chunk_size, internal_workers, max_tokens,
                   provider, thinking=True, progress=gr.Progress()):
    _apply_provider(provider)
    result = compress_text(
        input_path, model, temperature, chunk_size, internal_workers, max_tokens,
        thinking=thinking,
        progress_callback=lambda c, t: progress(c / t, desc=f"压缩中 {c}/{t}"),
    )
    config.save_state("compress", input_path=input_path, model=model, temperature=temperature,
                      chunk_size=chunk_size, internal_workers=internal_workers, max_tokens=max_tokens,
                      thinking=thinking)
    history.add_entry("聊天压缩", input_path, "压缩完成")
    return result


def _get_chapter_progress(input_file):
    """Read input file and return chapter progress table for Gradio."""
    rows = _svc_get_chapter_progress(input_file)
    if not rows:
        return gr.update(value=[], visible=True)
    return gr.update(value=rows, visible=True)


def _translate(input_file, output_file, model, batch_size, workers, provider, thinking=True, glossary_text="", progress=gr.Progress()):
    _apply_provider(provider)
    result = translate_text(
        input_file, output_file, model, batch_size, workers,
        thinking=thinking, glossary_text=glossary_text,
        progress_callback=lambda c, t: progress(c / t, desc=f"翻译中 {c}/{t}"),
    )
    config.save_state("translate", input_file=input_file, output_file=output_file,
                      model=model, batch_size=batch_size, workers=workers,
                      glossary_text=glossary_text, thinking=thinking)
    history.add_entry("文本翻译", input_file, "翻译完成")
    return result


# ============================================================
# UI rendering
# ============================================================

def render_tab_text_tools(s, provider_info):
    """Render the text tools tab (compress + translate). Returns component dict."""
    with gr.Tabs():
        # ---- Sub-tab: Chat compression ----
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
                            "纯文本任务，用普通文本模型即可，不必用视觉模型",
                            thinking_default=s["compress"].get("thinking", True))
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

        # ---- Sub-tab: Translation ----
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
                    tr_glossary = gr.Textbox(label="术语表（可选）",
                                             value=s["translate"].get("glossary_text", ""),
                                             placeholder="每行一条：原文=译文\n例如：哈利波特=Harry Potter\n霍格沃茨=Hogwarts",
                                             info="翻译时强制按术语表翻译指定词汇，留空则不启用",
                                             lines=4, max_lines=8)
                    with gr.Accordion("⚙️ 高级设置", open=False):
                        tr_model_type, tr_model, tr_fetch_btn, tr_fetch_st, tr_thinking = _make_model_selector(
                            "文本模型", s["translate"].get("model", config.TEXT_MODEL),
                            "纯文本翻译任务，使用文本模型",
                            thinking_default=s["translate"].get("thinking", True))
                        tr_batch = gr.Slider(1, 20, value=s["translate"].get("batch_size", 10), step=1, label="每批章节数",
                                             info="每批同时翻译多少章。越大越快但可能超时")
                        tr_workers = gr.Slider(1, 4, value=s["translate"].get("workers", 2), step=1, label="并行线程数",
                                               info="同时运行几个翻译任务")
                    with gr.Row():
                        tr_btn = gr.Button("开始翻译", variant="primary")
                        tr_stop = gr.Button("停止", variant="stop")
                        tr_progress_btn = gr.Button("查看章节进度")
                with gr.Column(scale=3):
                    tr_output = gr.Textbox(label="翻译日志", lines=15, elem_classes="output-text",
                                           placeholder="处理完成后这里会显示翻译进度、已翻章节数...")
                    tr_progress_table = gr.Dataframe(
                        headers=["章节序号", "章节标题", "状态", "字数"],
                        datatype=["number", "str", "str", "number"],
                        interactive=False,
                        visible=False,
                        label="章节进度",
                    )
            tr_btn.click(_translate, [tr_input, tr_output_file, tr_model, tr_batch, tr_workers, provider_info, tr_thinking, tr_glossary], [tr_output])

            def _stop_translate():
                from text_tools.translate import request_stop
                request_stop()
                return "⏹️ 已请求停止..."

            tr_stop.click(_stop_translate, outputs=[tr_output])
            tr_progress_btn.click(_get_chapter_progress, inputs=[tr_input], outputs=[tr_progress_table])
            _bind_model_fetch(tr_fetch_btn, tr_model_type, tr_model, tr_fetch_st,
                              provider_info, config.TEXT_MODEL)

    return {"ct_model": ct_model, "ct_model_type": ct_model_type, "tr_model": tr_model, "tr_model_type": tr_model_type}
