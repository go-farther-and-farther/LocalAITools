"""Supplier management components and logic."""
import logging
import config
import gradio as gr
from services import llama_server

logger = logging.getLogger("LocalAITools")


def _on_restart():
    """Restart the application."""
    print("\n🔄 用户请求重启...")
    import subprocess, sys, os
    from pathlib import Path

    script = str(Path(__file__).resolve().parent.parent / "app.py")
    cmd = f'timeout /t 1 /nobreak >nul & "{sys.executable}" "{script}"'
    subprocess.Popen(cmd, shell=True, cwd=str(Path(script).parent))
    os._exit(0)


def _provider_display(prov: dict) -> str:
    """Return a display string for a provider's connection info."""
    ptype = prov.get("type", "openai_compatible")
    if ptype == "llama_server":
        model = prov.get("model_path", "")
        model_name = model.split("\\")[-1].split("/")[-1] if model else "未设置"
        status = llama_server.get_status()
        icon = "🟢" if status["running"] and status.get("model_name") == (model.split("\\")[-1].split("/")[-1] if model else "") else "🔴"
        return f"{icon} llama.cpp: `{model_name}`"
    return f"📡 `{prov.get('base_url', '')}`"


def render_provider_bar(active_prov, prov_list, prov_active):
    """Render the top provider bar and editor. Returns a dict of components."""
    with gr.Row(equal_height=True, elem_classes="top-bar"):
        gr.Markdown("# LocalAITools - 本地 AI 工具箱")
        provider_select = gr.Dropdown(
            choices=[p["name"] for p in prov_list],
            value=prov_active,
            label="供应商",
            container=False,
            scale=2,
        )
        prov_add_btn = gr.Button("添加", size="sm", scale=0, min_width=50)
        prov_edit_btn = gr.Button("编辑", size="sm", scale=0, min_width=50)
        prov_del_btn = gr.Button("删除", size="sm", scale=0, min_width=50)
        restart_btn_top = gr.Button("🔄", size="sm", scale=0, min_width=40)

    # Provider info + llama-server control row
    with gr.Row(equal_height=True):
        prov_info_text = gr.Markdown(
            _provider_display(active_prov),
            elem_classes="hint",
        )
        # llama-server start/stop buttons (visible only for llama_server type)
        _is_llama = active_prov.get("type") == "openai_compatible" or not active_prov.get("type")
        llama_start_btn = gr.Button("▶️ 启动服务", size="sm", variant="primary",
                                     scale=0, min_width=80, visible=not _is_llama)
        llama_stop_btn = gr.Button("⏹ 停止", size="sm", variant="stop",
                                    scale=0, min_width=60, visible=not _is_llama)

    llama_status_msg = gr.Textbox("", interactive=False, container=False, show_label=False,
                                   visible=False, max_lines=2)

    restart_msg = gr.Textbox("", interactive=False, container=False, show_label=False, visible=False)
    restart_btn_top.click(_on_restart, outputs=[restart_msg])

    # Supplier editor (hidden by default)
    with gr.Accordion("添加/编辑供应商", open=False, visible=True) as prov_editor:
        with gr.Row():
            prov_name = gr.Textbox(label="名称", placeholder="如：Qwen3.6-35B-A3B", scale=1)
            prov_type = gr.Dropdown(
                label="类型",
                choices=["openai_compatible", "llama_server"],
                value="openai_compatible",
                scale=1,
            )

        # --- openai_compatible fields ---
        with gr.Group(visible=True) as group_openai:
            prov_url = gr.Textbox(label="API Base URL", placeholder="http://localhost:1234/v1", scale=2)
            prov_key = gr.Textbox(label="API Key", type="password", scale=2)

        # --- llama_server fields ---
        with gr.Group(visible=False) as group_llama:
            prov_server_path = gr.Textbox(
                label="llama 安装目录（自动检索 llama-server.exe）",
                placeholder=r"D:\Software files\llama\llama-b9116-bin-win-vulkan-x64",
            )
            with gr.Row():
                prov_model_path = gr.Dropdown(
                    label="GGUF 模型",
                    choices=[],
                    value="",
                    filterable=True,
                    allow_custom_value=True,
                    scale=3,
                )
                scan_models_btn = gr.Button("🔍 扫描模型", scale=0, variant="secondary", min_width=80)
                prov_mmproj_path = gr.Textbox(
                    label="mmproj 路径（视觉模型可选）",
                    placeholder=r"留空则自动查找同目录下的 mmproj 文件",
                    scale=2,
                )
            scan_result = gr.Textbox(visible=False, max_lines=2)
            with gr.Row():
                prov_ctx_size = gr.Number(label="Context Size", value=64000, precision=0, scale=1)
                prov_port = gr.Number(label="端口", value=1234, precision=0, scale=1)
                prov_expert_kv = gr.Number(label="Expert Used Count（MoE 模型, 0=不设置）",
                                            value=0, precision=0, scale=1)
            prov_reasoning = gr.Checkbox(label="启用推理模式 (Reasoning)", value=False)

            def _scan_models(root_dir):
                if not root_dir or not root_dir.strip():
                    return gr.update(), "请先填写 llama 安装目录"
                from services.llama_server import scan_gguf_models
                models = scan_gguf_models(root_dir.strip())
                if not models:
                    return gr.update(choices=[], value=""), "未找到 .gguf 模型文件"
                return gr.update(choices=models, value=models[0]), f"找到 {len(models)} 个模型"

            scan_models_btn.click(
                _scan_models,
                inputs=[prov_server_path],
                outputs=[prov_model_path, scan_result],
            )

        with gr.Row():
            prov_save_btn = gr.Button("💾 保存", variant="primary", scale=1)
            prov_cancel_btn = gr.Button("取消", variant="secondary", scale=1)
        prov_msg = gr.Markdown("")

    # State
    provider_info = gr.State({"base_url": active_prov["base_url"], "api_key": active_prov["api_key"]})
    providers_state = gr.State({"list": prov_list, "active": prov_active})
    prov_edit_mode = gr.State(None)
    prov_old_name = gr.State("")

    # ---- Event functions ----
    def _on_provider_change(name, providers):
        if not name:
            return gr.update(), {}, providers, gr.update(), gr.update(), gr.update()
        for p in providers["list"]:
            if p["name"] == name:
                providers["active"] = name
                config.save_providers(providers["list"], name)
                ptype = p.get("type", "openai_compatible")
                info_md = _provider_display(p)
                is_llama = ptype == "llama_server"
                # Update base_url for llama_server if running
                info = {"base_url": p.get("base_url", ""), "api_key": p.get("api_key", "")}
                if is_llama:
                    ls = llama_server.get_status()
                    if ls["running"]:
                        info["base_url"] = ls["base_url"]
                return (
                    info_md, info, providers,
                    gr.update(visible=is_llama),  # llama_start_btn
                    gr.update(visible=is_llama),   # llama_stop_btn
                    gr.update(visible=is_llama),   # llama_status_msg
                )
        return gr.update(), {}, providers, gr.update(), gr.update(), gr.update()

    def _on_prov_add(providers):
        return (
            gr.update(open=True),  # prov_editor
            "",   # prov_name
            "openai_compatible",  # prov_type
            "",   # prov_url
            "",   # prov_key
            "",   # prov_server_path
            gr.update(choices=[], value=""),  # prov_model_path
            "",   # prov_mmproj_path
            64000,  # prov_ctx_size
            1234,   # prov_port
            0,      # prov_expert_kv
            False,  # prov_reasoning
            "add",  # prov_edit_mode
            "",     # prov_msg
            "",     # prov_old_name
            gr.update(visible=True),   # group_openai
            gr.update(visible=False),  # group_llama
        )

    def _on_prov_edit(providers):
        name = providers.get("active", "")
        for p in providers["list"]:
            if p["name"] == name:
                ptype = p.get("type", "openai_compatible")
                is_llama = ptype == "llama_server"
                # Scan models for the dropdown if editing a llama_server provider
                model_choices = []
                model_value = p.get("model_path", "")
                if is_llama and p.get("llama_server_path"):
                    model_choices = llama_server.scan_gguf_models(p["llama_server_path"])
                    if model_value and model_value not in model_choices:
                        model_choices.insert(0, model_value)
                return (
                    gr.update(open=True),  # prov_editor
                    p["name"],             # prov_name
                    ptype,                 # prov_type
                    p.get("base_url", ""), # prov_url
                    p.get("api_key", ""),  # prov_key
                    p.get("llama_server_path", ""),  # prov_server_path
                    gr.update(choices=model_choices, value=model_value) if is_llama else model_value,  # prov_model_path
                    p.get("mmproj_path", ""),         # prov_mmproj_path
                    p.get("ctx_size", 64000),         # prov_ctx_size
                    p.get("port", 1234),              # prov_port
                    p.get("expert_kv", 0),            # prov_expert_kv
                    p.get("reasoning", False),         # prov_reasoning
                    "edit",                           # prov_edit_mode
                    "",                               # prov_msg
                    name,                             # prov_old_name
                    gr.update(visible=not is_llama),  # group_openai
                    gr.update(visible=is_llama),      # group_llama
                )
        return (
            gr.update(open=True), "", "openai_compatible", "", "", "", "", "", 64000, 1234, 0, False,
            "edit", "", "", gr.update(visible=True), gr.update(visible=False),
        )

    def _on_prov_type_change(ptype):
        is_llama = ptype == "llama_server"
        return gr.update(visible=not is_llama), gr.update(visible=is_llama)

    prov_type.change(
        _on_prov_type_change,
        inputs=[prov_type],
        outputs=[group_openai, group_llama],
    )

    def _on_prov_save(mode, old_name, name, ptype, url, key,
                      server_path, model_path, mmproj_path, ctx_size, port, expert_kv, reasoning,
                      providers):
        if not name.strip():
            return "❌ 名称不能为空", providers
        if ptype == "openai_compatible" and not url.strip():
            return "❌ URL 不能为空", providers
        if ptype == "llama_server" and not model_path.strip():
            return "❌ 模型路径不能为空", providers

        # Build provider dict
        prov = {"name": name.strip(), "type": ptype}
        if ptype == "openai_compatible":
            prov["base_url"] = url.strip()
            prov["api_key"] = key.strip()
        else:
            prov["base_url"] = f"http://127.0.0.1:{int(port)}/v1"
            prov["api_key"] = "local"
            prov["llama_server_path"] = server_path.strip()
            prov["model_path"] = model_path.strip()
            prov["mmproj_path"] = mmproj_path.strip()
            prov["ctx_size"] = int(ctx_size) if ctx_size else 64000
            prov["port"] = int(port) if port else 1234
            prov["expert_kv"] = int(expert_kv) if expert_kv else 0
            prov["reasoning"] = bool(reasoning)

        if mode == "add":
            if any(p["name"] == name.strip() for p in providers["list"]):
                return f"❌ 已存在同名供应商「{name.strip()}」", providers
            providers["list"].append(prov)
            providers["active"] = name.strip()
        elif mode == "edit":
            for i, p in enumerate(providers["list"]):
                if p["name"] == old_name:
                    providers["list"][i] = prov
                    break
            if providers["active"] == old_name:
                providers["active"] = name.strip()
        config.save_providers(providers["list"], providers["active"])
        return f"✅ 已保存供应商「{name.strip()}」", providers

    def _on_prov_delete(providers):
        name = providers.get("active", "")
        _no_llama = gr.update(visible=False)
        if len(providers["list"]) <= 1:
            return "❌ 至少保留一个供应商", providers, gr.update(), gr.update(), gr.update(), _no_llama, _no_llama, gr.update(visible=False)
        # Stop llama-server if it belongs to this provider
        for p in providers["list"]:
            if p["name"] == name and p.get("type") == "llama_server":
                llama_server.stop_server()
                break
        providers["list"] = [p for p in providers["list"] if p["name"] != name]
        providers["active"] = providers["list"][0]["name"]
        config.save_providers(providers["list"], providers["active"])
        new_active = providers["list"][0]
        names = [p["name"] for p in providers["list"]]
        info_md = _provider_display(new_active)
        is_llama = new_active.get("type") == "llama_server"
        return (
            f"✅ 已删除供应商「{name}」", providers,
            gr.update(choices=names, value=providers["active"]),
            info_md, gr.update(),
            gr.update(visible=is_llama),
            gr.update(visible=is_llama),
            gr.update(visible=is_llama),
        )

    def _on_prov_save_and_refresh(mode, old_name, name, ptype, url, key,
                                   server_path, model_path, mmproj_path, ctx_size, port, expert_kv,
                                   reasoning, providers):
        effective_old = old_name if old_name else name
        msg, providers = _on_prov_save(mode, effective_old, name, ptype, url, key,
                                        server_path, model_path, mmproj_path, ctx_size, port,
                                        expert_kv, reasoning, providers)
        prov_names = [p["name"] for p in providers["list"]]
        active_prov = None
        for p in providers["list"]:
            if p["name"] == providers["active"]:
                active_prov = p
                break
        info_md = _provider_display(active_prov) if active_prov else ""
        prov_info = {}
        if active_prov:
            prov_info = {"base_url": active_prov.get("base_url", ""), "api_key": active_prov.get("api_key", "")}
        is_llama = active_prov and active_prov.get("type") == "llama_server"
        return (
            msg, providers,
            gr.update(choices=prov_names, value=providers["active"]),
            info_md, prov_info,
            gr.update(visible=is_llama),   # llama_start_btn
            gr.update(visible=is_llama),   # llama_stop_btn
            gr.update(visible=is_llama),   # llama_status_msg
        )

    def _on_prov_delete_and_refresh(providers):
        msg, providers, new_select, info_md, _, llama_vis1, llama_vis2, llama_vis3 = _on_prov_delete(providers)
        active_prov = None
        for p in providers["list"]:
            if p["name"] == providers["active"]:
                active_prov = p
                break
        prov_info = {}
        if active_prov:
            prov_info = {"base_url": active_prov.get("base_url", ""), "api_key": active_prov.get("api_key", "")}
        is_llama = active_prov and active_prov.get("type") == "llama_server"
        return (
            msg, providers, new_select, info_md, prov_info,
            gr.update(visible=is_llama),   # llama_start_btn
            gr.update(visible=is_llama),   # llama_stop_btn
            gr.update(visible=is_llama),   # llama_status_msg
        )

    # ---- llama-server start/stop ----
    def _on_llama_start(providers):
        name = providers.get("active", "")
        for p in providers["list"]:
            if p["name"] == name and p.get("type") == "llama_server":
                result = llama_server.launch_server(
                    server_path=p["llama_server_path"],
                    model_path=p["model_path"],
                    mmproj_path=p.get("mmproj_path", ""),
                    port=p.get("port", 1234),
                    ctx_size=p.get("ctx_size", 64000),
                    expert_kv=p.get("expert_kv", 0),
                    reasoning=p.get("reasoning", False),
                )
                return result
        return "当前供应商不是 llama_server 类型"

    def _on_llama_stop():
        return llama_server.stop_server()

    llama_start_btn.click(_on_llama_start, inputs=[providers_state], outputs=[llama_status_msg])
    llama_stop_btn.click(_on_llama_stop, outputs=[llama_status_msg])

    # Bind events
    prov_add_btn.click(
        _on_prov_add,
        inputs=[providers_state],
        outputs=[prov_editor, prov_name, prov_type, prov_url, prov_key,
                 prov_server_path, prov_model_path, prov_mmproj_path,
                 prov_ctx_size, prov_port, prov_expert_kv, prov_reasoning,
                 prov_edit_mode, prov_msg, prov_old_name,
                 group_openai, group_llama],
    )
    prov_edit_btn.click(
        _on_prov_edit,
        inputs=[providers_state],
        outputs=[prov_editor, prov_name, prov_type, prov_url, prov_key,
                 prov_server_path, prov_model_path, prov_mmproj_path,
                 prov_ctx_size, prov_port, prov_expert_kv, prov_reasoning,
                 prov_edit_mode, prov_msg, prov_old_name,
                 group_openai, group_llama],
    )

    return {
        "provider_select": provider_select,
        "prov_save_btn": prov_save_btn,
        "prov_del_btn": prov_del_btn,
        "prov_info_text": prov_info_text,
        "provider_info": provider_info,
        "providers_state": providers_state,
        "prov_edit_mode": prov_edit_mode,
        "prov_old_name": prov_old_name,
        "prov_name": prov_name,
        "prov_type": prov_type,
        "prov_url": prov_url,
        "prov_key": prov_key,
        "prov_server_path": prov_server_path,
        "prov_model_path": prov_model_path,
        "prov_mmproj_path": prov_mmproj_path,
        "prov_ctx_size": prov_ctx_size,
        "prov_port": prov_port,
        "prov_expert_kv": prov_expert_kv,
        "prov_reasoning": prov_reasoning,
        "prov_msg": prov_msg,
        "group_openai": group_openai,
        "group_llama": group_llama,
        "llama_start_btn": llama_start_btn,
        "llama_stop_btn": llama_stop_btn,
        "llama_status_msg": llama_status_msg,
        # Functions for external binding
        "_on_prov_save_and_refresh": _on_prov_save_and_refresh,
        "_on_prov_delete_and_refresh": _on_prov_delete_and_refresh,
        "_on_provider_change": _on_provider_change,
    }
