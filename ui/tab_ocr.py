"""Chat screenshot OCR tab."""
import config
import history
import gradio as gr
from ui.common import _make_title, _make_model_selector, _bind_model_fetch, _apply_provider
from services.ocr_services import extract_chat_text as _svc_ocr


def _explain_images(input_dir, vision_model, temperature, workers, internal_workers, max_tokens,
                    max_size, provider, thinking=True, progress=gr.Progress()):
    _apply_provider(provider)
    result = _svc_ocr(input_dir, vision_model, temperature, workers, internal_workers, max_tokens,
                      max_size, thinking, progress_callback=lambda c, t: progress(c / t, desc=f"识别中 {c}/{t}"))
    config.save_state("ocr", input_dir=input_dir, model=vision_model, temperature=temperature,
                      workers=workers, internal_workers=internal_workers, max_tokens=max_tokens,
                      max_size=max_size, thinking=thinking)
    history.add_entry("截图识别", input_dir, "文字提取完成")
    return result


def render_tab_ocr(s, provider_info):
    """Render the OCR tab. Returns component dict."""
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
                    "需要视觉模型，推荐带 Thinking 能力的模型",
                    thinking_default=s["ocr"].get("thinking", False))
                ei_temp = gr.Slider(0.0, 1.0, value=s["ocr"].get("temperature", 0.3), step=0.1, label="温度",
                                    info="越低越稳定，越高越有创意。OCR 任务建议 0.2-0.4")
                ei_workers = gr.Slider(1, 4, value=s["ocr"].get("workers", 4), step=1, label="并行图片数",
                                       info="同时处理几张图。注意：过大可能导致显存不足")
                ei_iworkers = gr.Slider(1, 4, value=s["ocr"].get("internal_workers", 2), step=1, label="单图内部并发",
                                        info="每张图内部分两半并行处理")
                ei_maxtok = gr.Slider(1000, 8000, value=s["ocr"].get("max_tokens", 5000), step=500, label="最大 Token 数",
                                      info="输出上限，长截图可适当调大")
                ei_maxsz = gr.Slider(512, 4096, value=s["ocr"].get("max_size", config.IMAGE_MAX_SIZE), step=256,
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

    return {"ei_model": ei_model, "ei_model_type": ei_model_type}
