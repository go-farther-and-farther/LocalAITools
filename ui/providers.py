"""Supplier management components and logic."""
import logging
import config
import gradio as gr

logger = logging.getLogger("LocalAITools")


def _on_restart():
    """Restart the application."""
    print("\n🔄 用户请求重启...")
    import subprocess, sys, os
    from pathlib import Path

    script = str(Path(__file__).resolve().parent.parent / "app.py")
    # Use shell "timeout 1 && python app.py" so the old process can die first
    cmd = f'timeout /t 1 /nobreak >nul & "{sys.executable}" "{script}"'
    subprocess.Popen(cmd, shell=True, cwd=str(Path(script).parent))
    os._exit(0)


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
    prov_info_text = gr.Markdown(
        f"📡 `{active_prov['base_url']}`",
        elem_classes="hint",
    )
    restart_msg = gr.Textbox("", interactive=False, container=False, show_label=False, visible=False)
    restart_btn_top.click(_on_restart, outputs=[restart_msg])

    # Supplier editor (hidden by default)
    with gr.Accordion("添加/编辑供应商", open=False, visible=True) as prov_editor:
        with gr.Row():
            prov_name = gr.Textbox(label="名称", placeholder="如：硅基流动", scale=1)
            prov_url = gr.Textbox(label="API Base URL", placeholder="https://api.siliconflow.cn/v1", scale=2)
            prov_key = gr.Textbox(label="API Key", type="password", scale=2)
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
            return gr.update(), {}, providers
        for p in providers["list"]:
            if p["name"] == name:
                providers["active"] = name
                config.save_providers(providers["list"], name)
                info_md = f"📡 `{p['base_url']}`"
                return info_md, {"base_url": p["base_url"], "api_key": p["api_key"]}, providers
        return gr.update(), {}, providers

    def _on_prov_add(providers):
        return gr.update(open=True), "", "", "", "add", "", ""

    def _on_prov_edit(providers):
        name = providers.get("active", "")
        for p in providers["list"]:
            if p["name"] == name:
                return gr.update(open=True), name, p["base_url"], p["api_key"], "edit", "", name
        return gr.update(open=True), "", "", "", "edit", "", ""

    def _on_prov_save(mode, old_name, name, url, key, providers):
        if not name.strip():
            return "❌ 名称不能为空", providers
        if not url.strip():
            return "❌ URL 不能为空", providers
        if mode == "add":
            if any(p["name"] == name.strip() for p in providers["list"]):
                return f"❌ 已存在同名供应商「{name.strip()}」", providers
            providers["list"].append({"name": name.strip(), "base_url": url.strip(), "api_key": key.strip()})
            providers["active"] = name.strip()
        elif mode == "edit":
            for p in providers["list"]:
                if p["name"] == old_name:
                    p["name"] = name.strip()
                    p["base_url"] = url.strip()
                    p["api_key"] = key.strip()
                    break
            if providers["active"] == old_name:
                providers["active"] = name.strip()
        config.save_providers(providers["list"], providers["active"])
        return f"✅ 已保存供应商「{name.strip()}」", providers

    def _on_prov_delete(providers):
        name = providers.get("active", "")
        if len(providers["list"]) <= 1:
            return "❌ 至少保留一个供应商", providers, gr.update(), gr.update()
        providers["list"] = [p for p in providers["list"] if p["name"] != name]
        providers["active"] = providers["list"][0]["name"]
        config.save_providers(providers["list"], providers["active"])
        new_active = providers["list"][0]
        names = [p["name"] for p in providers["list"]]
        info_md = f"📡 `{new_active['base_url']}`"
        return f"✅ 已删除供应商「{name}」", providers, gr.update(choices=names, value=providers["active"]), info_md

    def _on_prov_save_and_refresh(mode, old_name, name, url, key, providers):
        # Use old_name from state if available, fallback to name textbox
        effective_old = old_name if old_name else name
        msg, providers = _on_prov_save(mode, effective_old, name, url, key, providers)
        prov_names = [p["name"] for p in providers["list"]]
        active_prov = None
        for p in providers["list"]:
            if p["name"] == providers["active"]:
                active_prov = p
                break
        info_md = f"📡 `{active_prov['base_url']}`" if active_prov else ""
        prov_info = {"base_url": active_prov["base_url"], "api_key": active_prov["api_key"]} if active_prov else {}
        return msg, providers, gr.update(choices=prov_names, value=providers["active"]), info_md, prov_info

    def _on_prov_delete_and_refresh(providers):
        msg, providers, new_select, info_md = _on_prov_delete(providers)
        active_prov = None
        for p in providers["list"]:
            if p["name"] == providers["active"]:
                active_prov = p
                break
        prov_info = {"base_url": active_prov["base_url"], "api_key": active_prov["api_key"]} if active_prov else {}
        return msg, providers, new_select, info_md, prov_info

    # Bind events
    prov_add_btn.click(
        _on_prov_add,
        inputs=[providers_state],
        outputs=[prov_editor, prov_name, prov_url, prov_key, prov_edit_mode, prov_msg, prov_old_name],
    )
    prov_edit_btn.click(
        _on_prov_edit,
        inputs=[providers_state],
        outputs=[prov_editor, prov_name, prov_url, prov_key, prov_edit_mode, prov_msg, prov_old_name],
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
        "prov_url": prov_url,
        "prov_key": prov_key,
        "prov_msg": prov_msg,
        # Functions for external binding
        "_on_prov_save_and_refresh": _on_prov_save_and_refresh,
        "_on_prov_delete_and_refresh": _on_prov_delete_and_refresh,
        "_on_provider_change": _on_provider_change,
    }
