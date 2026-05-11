"""Shared UI components and helpers."""
import os
import sys
import io
import logging
from pathlib import Path
from typing import Any, Callable
import config
import gradio as gr

logger = logging.getLogger("LocalAITools")


def _make_title(text: str) -> str:
    return f"## {text}"


def _capture_log(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """捕获 logging 输出为字符串，异常时返回错误信息"""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        fn(*args, **kwargs)
    except Exception as e:
        import traceback
        stream.write(f"❌ 处理出错: {e}\n")
        stream.write(traceback.format_exc())
    finally:
        root.removeHandler(handler)
    return stream.getvalue() or "✅ 处理完成"


def _apply_provider(prov: dict | None) -> None:
    """将供应商信息写入环境变量，使 config 模块读取到正确的 API 地址和密钥"""
    if prov and isinstance(prov, dict):
        if prov.get("base_url"):
            os.environ["OPENAI_BASE_URL"] = prov["base_url"]
            config.OPENAI_BASE_URL = prov["base_url"]
        if prov.get("api_key") is not None:
            os.environ["OPENAI_API_KEY"] = prov["api_key"]
            config.OPENAI_API_KEY = prov["api_key"]


def _test_api_connection(base_url: str, api_key: str, model: str) -> str:
    """测试 API 连接是否正常"""
    logger.info(f"[测试连接] URL={base_url} 模型={model}")
    if not base_url.strip():
        return "❌ 请填写 API 地址"
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url.strip(), api_key=api_key.strip() or "no-key", timeout=10)
        models = client.models.list()
        model_names = [m.id for m in models]
        snippet = ", ".join(model_names[:8])
        lines = [f"✅ 连接成功！找到 {len(model_names)} 个模型"]
        if model_names:
            lines.append(f"可用模型示例: {snippet}{'...' if len(model_names) > 8 else ''}")
        if model.strip() and model.strip() not in model_names:
            lines.append(f"⚠️ 注意: 当前配置的模型「{model.strip()}」不在可用列表中")
        return "\n".join(lines)
    except Exception as e:
        msg = str(e)
        if "Connection" in msg or "refused" in msg.lower():
            return f"❌ 无法连接到 {base_url}\n请确认：\n1. LM Studio 或其他 API 服务是否已启动\n2. 地址和端口是否正确"
        return f"❌ 连接失败: {msg}"


def _fetch_models(base_url: str, api_key: str) -> tuple[dict, str]:
    """从 API 获取可用模型列表，按类型分类"""
    if not base_url or not base_url.strip():
        return {}, "❌ 请填写 API 地址"
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url.strip(), api_key=(api_key or "").strip() or "no-key", timeout=10)
        models = client.models.list()
        model_ids = sorted([m.id for m in models])
    except Exception as e:
        return {}, f"❌ 获取失败: {e}"

    if not model_ids:
        return {}, "⚠️ 未找到任何模型"

    def _classify(mid):
        low = mid.lower()
        if any(k in low for k in config.MODEL_KEYWORDS_EMBED):
            return 'embed'
        if any(k in low for k in config.MODEL_KEYWORDS_VLM):
            return 'vlm'
        if any(k in low for k in config.MODEL_KEYWORDS_CHAT):
            return 'chat'
        return 'other'

    categorized = {'全部模型': [], 'chat 模型': [], 'vlm 视觉模型': [], 'embed 模型': [], '其他': []}
    for mid in model_ids:
        categorized['全部模型'].append(mid)
        cat = _classify(mid)
        if cat == 'embed':
            categorized['embed 模型'].append(mid)
        elif cat == 'vlm':
            categorized['vlm 视觉模型'].append(mid)
        elif cat == 'chat':
            categorized['chat 模型'].append(mid)
        else:
            categorized['其他'].append(mid)

    # 移除空分类
    categorized = {k: v for k, v in categorized.items() if v}
    msg = f"✅ 获取到 {len(model_ids)} 个模型"
    return categorized, msg


def _make_model_selector(label: str, default_value: str, info: str = "", show_thinking: bool = True, thinking_default: bool = True) -> tuple[gr.Dropdown, gr.Dropdown, gr.Button, gr.Textbox, gr.Checkbox]:
    """创建模型选择器组件（类型筛选 + 下拉选择 + 重新获取按钮 + 思考模式开关）"""
    with gr.Row(equal_height=True):
        model_type = gr.Dropdown(
            choices=["全部模型", "chat 模型", "vlm 视觉模型", "embed 模型", "其他"],
            value="全部模型",
            label="分类",
            scale=1,
            container=False,
        )
        model_select = gr.Dropdown(
            choices=[default_value] if default_value else [],
            value=default_value,
            label=label,
            info=info,
            filterable=True,
            allow_custom_value=True,
            scale=3,
            container=False,
        )
        fetch_btn = gr.Button("🔄", scale=0, variant="secondary", min_width=40)
    thinking_toggle = gr.Checkbox(
        label="思考模式（Thinking）",
        value=thinking_default,
        visible=show_thinking,
        info="开启后模型会先推理再回答，关闭可加快速度",
    )
    fetch_status = gr.Textbox(visible=False, max_lines=1)
    return model_type, model_select, fetch_btn, fetch_status, thinking_toggle


def _bind_model_fetch_impl(fetch_btn: gr.Button, model_type: gr.Dropdown, model_select: gr.Dropdown, fetch_status: gr.Textbox,
                           fetch_fn: Callable[..., tuple[dict, str]], fetch_inputs: list[gr.components.Component], default_value: str) -> Callable[..., Any]:
    """Shared implementation for model fetch binding."""
    _models_cache = gr.State({})

    def _handle_result(categorized, msg):
        all_models = categorized.get("全部模型", [])
        return (
            gr.update(choices=list(categorized.keys()), value="全部模型"),
            gr.update(choices=all_models, value=default_value if default_value in all_models else (all_models[0] if all_models else "")),
            msg,
            categorized,
        )

    def _do_fetch(*inputs):
        categorized, msg = fetch_fn(*inputs)
        return _handle_result(categorized, msg)

    fetch_btn.click(
        _do_fetch,
        inputs=fetch_inputs,
        outputs=[model_type, model_select, fetch_status, _models_cache],
    )

    def _filter_models(type_val, cache):
        if not cache:
            return gr.update()
        if type_val in cache:
            return gr.update(choices=cache[type_val])
        return gr.update(choices=cache.get("全部模型", []))

    model_type.change(
        _filter_models,
        inputs=[model_type, _models_cache],
        outputs=[model_select],
    )

    return _do_fetch


def _bind_model_fetch(fetch_btn: gr.Button, model_type: gr.Dropdown, model_select: gr.Dropdown, fetch_status: gr.Textbox, provider_info: gr.State, default_value: str) -> Callable[..., Any]:
    """绑定模型获取和筛选事件。provider_info 是 gr.State，存储当前供应商的 {base_url, api_key}。"""
    def _fetch_from_provider(prov):
        url = prov.get("base_url", "") if isinstance(prov, dict) else ""
        key = prov.get("api_key", "") if isinstance(prov, dict) else ""
        return _fetch_models(url, key)

    return _bind_model_fetch_impl(
        fetch_btn, model_type, model_select, fetch_status,
        _fetch_from_provider, [provider_info], default_value,
    )


def _bind_model_fetch_local(fetch_btn: gr.Button, model_type: gr.Dropdown, model_select: gr.Dropdown, fetch_status: gr.Textbox, url_component: gr.Textbox, key_component: gr.Textbox, default_value: str) -> Callable[..., Any]:
    """绑定模型获取事件（使用 Tab 自有的 URL/Key 输入组件）。"""
    return _bind_model_fetch_impl(
        fetch_btn, model_type, model_select, fetch_status,
        _fetch_models, [url_component, key_component], default_value,
    )
