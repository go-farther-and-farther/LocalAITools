#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LocalAITools - Gradio Web 界面
启动方式：python app.py  或  双击 启动.bat
然后浏览器打开 http://localhost:7860
"""

import os
import sys
import logging
from pathlib import Path

# 控制台日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("LocalAITools")

sys.path.insert(0, str(Path(__file__).parent))
import config
import gradio as gr

from ui.common import _fetch_models
from ui.providers import render_provider_bar
from ui.tab_welcome import render_tab_welcome
from ui.tab_image_tools import render_tab_image_tools
from ui.tab_ocr import render_tab_ocr
from ui.tab_chat import render_tab_chat
from ui.tab_kb import render_tab_kb
from ui.tab_text_tools import render_tab_text_tools
from ui.tab_benchmark import render_tab_benchmark
from ui.tab_settings import render_tab_settings, check_and_update_on_startup

ROOT_DIR = Path(__file__).parent
_STATIC_DIR = ROOT_DIR / "static"
CSS = (_STATIC_DIR / "chat.css").read_text(encoding="utf-8")
JS_ONLOAD = (_STATIC_DIR / "chat.js").read_text(encoding="utf-8")


def build_ui():
    s = {k: config.load_state(k) for k in ["rename", "score", "classify", "ai_classify", "ocr", "compress", "translate", "benchmark"]}

    _prov_list, _prov_active = config.load_providers()
    _active_prov = config.get_active_provider()

    # Auto-start local model if configured
    from services.local_model import auto_start as _auto_start_local, get_status as _local_status
    _local_msg = _auto_start_local()
    if _local_msg:
        logger.info(_local_msg)

    # Inject local model as a provider if available
    _local = _local_status()
    if _local["available"] and _local["model_path"]:
        _local_url = _local.get("base_url") or "http://127.0.0.1:8081/v1"
        _exists = any(p["name"] == "本地模型" for p in _prov_list)
        if not _exists:
            _prov_list.append({"name": "本地模型", "base_url": _local_url, "api_key": "local"})
        elif _local["running"]:
            for p in _prov_list:
                if p["name"] == "本地模型":
                    p["base_url"] = _local_url
                    break

    with gr.Blocks(title="LocalAITools") as app:
        # ---- Top bar: title + provider + buttons ----
        prov = render_provider_bar(_active_prov, _prov_list, _prov_active)

        # ---- Tabs ----
        with gr.Tab("🏠 开始使用"):
            render_tab_welcome(app)

        with gr.Tab("🖼️ 图片工具") as _tab_image:
            t_image = render_tab_image_tools(s, prov["provider_info"])

        with gr.Tab("💬 聊天截图识别") as _tab_ocr:
            t_ocr = render_tab_ocr(s, prov["provider_info"])

        with gr.Tab("💬 聊天") as kb_chat_tab:
            t_chat = render_tab_chat(s, prov["provider_info"], kb_chat_tab)

        with gr.Tab("📚 知识库"):
            render_tab_kb(s)

        with gr.Tab("📝 文本工具") as _tab_text:
            t_text = render_tab_text_tools(s, prov["provider_info"])

        with gr.Tab("⚡ LLM 压测") as _tab_bench:
            t_bench = render_tab_benchmark(s)

        with gr.Tab("⚙️ 设置"):
            render_tab_settings()

        # ---- Provider switch → refresh all model lists ----
        def _refresh_all_models(prov_info):
            url = prov_info.get("base_url", "") if isinstance(prov_info, dict) else ""
            key = prov_info.get("api_key", "") if isinstance(prov_info, dict) else ""
            try:
                categorized, msg = _fetch_models(url, key)
            except Exception:
                categorized = {}
            all_models = categorized.get("全部模型", [])
            cat_choices = list(categorized.keys()) if categorized else []

            def _upd(default):
                if not all_models:
                    return gr.update(), gr.update()
                val = default if default in all_models else all_models[0]
                return gr.update(choices=all_models, value=val), gr.update(choices=cat_choices, value="全部模型")

            rn_m, rn_t = _upd(config.RENAME_MODEL)
            de_m, de_t = _upd(config.VISION_MODEL)
            ei_m, ei_t = _upd(config.VISION_MODEL_THINKING)
            ct_m, ct_t = _upd(config.VISION_MODEL_THINKING)
            tr_m, tr_t = _upd(config.TEXT_MODEL)
            kb_m, kb_t = _upd(config.TEXT_MODEL)
            bm_m, bm_t = _upd(config.BENCHMARK_MODEL)
            return (rn_m, rn_t, de_m, de_t, ei_m, ei_t, ct_m, ct_t, tr_m, tr_t, kb_m, kb_t, bm_m, bm_t,
                    gr.update(value=url), gr.update(value=key))

        _model_outputs = [
            t_image["rn_model"], t_image["rn_model_type"],
            t_image["de_model"], t_image["de_model_type"],
            t_ocr["ei_model"], t_ocr["ei_model_type"],
            t_text["ct_model"], t_text["ct_model_type"],
            t_text["tr_model"], t_text["tr_model_type"],
            t_chat["kb_model"], t_chat["kb_model_type"],
            t_bench["bm_model"], t_bench["bm_model_type"],
            t_bench["bm_url"], t_bench["bm_key"],
        ]

        prov["prov_save_btn"].click(
            prov["_on_prov_save_and_refresh"],
            inputs=[prov["prov_edit_mode"], prov["prov_old_name"], prov["prov_name"],
                    prov["prov_url"], prov["prov_key"], prov["providers_state"]],
            outputs=[prov["prov_msg"], prov["providers_state"], prov["provider_select"],
                     prov["prov_info_text"], prov["provider_info"]],
        ).then(_refresh_all_models, inputs=[prov["provider_info"]], outputs=_model_outputs)

        prov["prov_del_btn"].click(
            prov["_on_prov_delete_and_refresh"],
            inputs=[prov["providers_state"]],
            outputs=[prov["prov_msg"], prov["providers_state"], prov["provider_select"],
                     prov["prov_info_text"], prov["provider_info"]],
        ).then(_refresh_all_models, inputs=[prov["provider_info"]], outputs=_model_outputs)

        prov["provider_select"].change(
            prov["_on_provider_change"],
            inputs=[prov["provider_select"], prov["providers_state"]],
            outputs=[prov["prov_info_text"], prov["provider_info"], prov["providers_state"]],
        ).then(_refresh_all_models, inputs=[prov["provider_info"]], outputs=_model_outputs)

        # ---- Chat tab: refresh conversation list on select (in-memory only, no disk I/O) ----
        def _on_chat_tab_select(all_chats, active_name):
            all_chats = dict(all_chats) if all_chats else {}
            choices = sorted(all_chats.keys())
            return gr.update(choices=choices), ""

        kb_chat_tab.select(_on_chat_tab_select,
                           [t_chat["kb_all_chats"], t_chat["kb_active_name"]],
                           [t_chat["kb_chat_list"], t_chat["kb_chat_status"]])

    return app


if __name__ == "__main__":
    import traceback
    try:
        update_msg = check_and_update_on_startup()
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
