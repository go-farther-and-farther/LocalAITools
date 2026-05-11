"""Settings tab: .env editing, API test, version update, config backup/restore."""
import os
import re
import subprocess
import logging
import zipfile
import time
from pathlib import Path
import config
import gradio as gr
from ui.common import _make_title, _make_model_selector, _bind_model_fetch_local, _test_api_connection

logger = logging.getLogger("LocalAITools")

ROOT_DIR = Path(__file__).parent.parent
ENV_PATH = ROOT_DIR / ".env"
EXAMPLE_PATH = ROOT_DIR / ".env.example"

# Config items displayed on settings page, grouped
SETTINGS_SCHEMA = [
    ("🔗 API 连接", [
        ("OPENAI_BASE_URL", "API 地址", "http://localhost:1234/v1", "text"),
        ("OPENAI_API_KEY", "API 密钥", "lm-studio", "password"),
    ]),
    ("🤖 模型名称", [
        ("VISION_MODEL", "视觉模型（图片识别、质量检测）", "qwen/qwen3.6-27b", "text"),
        ("VISION_MODEL_THINKING", "视觉模型-思考版（截图 OCR 推荐）", "qwen/qwen3.6-35b-a3b-Thinking", "text"),
        ("RENAME_MODEL", "图片重命名模型", "qwen/Qwen3.6-27b", "text"),
        ("TEXT_MODEL", "文本模型（翻译、压缩、知识库）", "qwen/qwen3.5-9b", "text"),
        ("BENCHMARK_MODEL", "压测模型", "qwen3.6-35b-a3b@iq2_xxs", "text"),
    ]),
    ("⚡ 并发与超时", [
        ("DEFAULT_WORKERS", "默认并行线程数", "2", "int"),
        ("RETRY_TIMES", "失败重试次数", "2", "int"),
        ("REQUEST_TIMEOUT", "请求超时-长任务（秒）", "300", "int"),
        ("REQUEST_TIMEOUT_SHORT", "请求超时-短任务（秒）", "60", "int"),
    ]),
    ("🖼️ 图片处理", [
        ("IMAGE_MAX_SIZE", "输入图片最大边长（像素）", "2048", "int"),
        ("SLICE_HEIGHT", "长截图切片高度（像素）", "2000", "int"),
        ("OVERLAP", "切片重叠高度（像素）", "400", "int"),
        ("TOP_PERCENT", "高质量比例（小数）", "0.05", "float"),
        ("BOTTOM_PERCENT", "低质量比例（小数）", "0.05", "float"),
        ("HIGH_QUALITY_FOLDER", "高质量目录名", "HighQuality", "text"),
        ("LOW_QUALITY_ERRORS_FOLDER", "低质量目录名", "LowQuality_Errors", "text"),
    ]),
    ("📝 文本 / 翻译", [
        ("DEFAULT_CHUNK_SIZE", "聊天分块大小（字符）", "20480", "int"),
        ("OVERLAP_MESSAGES", "块间重叠消息数", "2", "int"),
        ("SOURCE_LANG", "翻译源语言", "Chinese", "text"),
        ("TARGET_LANG", "翻译目标语言", "English", "text"),
    ]),
    ("🔄 更新", [
        ("AUTO_UPDATE", "启动时自动检查更新", "true", "bool"),
    ]),
    ("📂 目录与索引", [
        ("DATA_DIR", "数据输入目录", "data", "text"),
        ("OUTPUT_DIR", "输出目录", "outputs", "text"),
        ("FAISS_INDEX_PATH", "FAISS 索引路径", "faiss_index", "text"),
        ("EMBEDDING_MODEL_PATH", "Embedding 模型路径（留空=在线下载）", "", "text"),
        ("KB_CHUNK_SIZE", "知识库文本块大小（字符）", "500", "int"),
        ("KB_CHUNK_OVERLAP", "知识库块间重叠（字符）", "50", "int"),
    ]),
]


def _read_env() -> dict:
    """Read .env file and return {KEY: value}.

    Handles:
    - Quoted values (single or double): strips surrounding quotes
    - Inline comments: if value is not quoted and contains ' #', treats
      everything after ' #' as a comment (trimmed away)
    """
    if not ENV_PATH.exists():
        return {}
    result = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(\w+)\s*=\s*(.*)$", line)
        if m:
            value = m.group(2).strip()
            # Strip surrounding quotes (single or double)
            if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"')
                                    or (value[0] == "'" and value[-1] == "'")):
                value = value[1:-1]
            else:
                # Only strip inline comments when value is NOT quoted
                comment_idx = value.find(" #")
                if comment_idx != -1:
                    value = value[:comment_idx].strip()
            result[m.group(1)] = value
    return result


def _save_env(updates: dict) -> str:
    """Write updates to .env file, preserving comments and other lines."""
    logger.info(f"[保存设置] {list(updates.keys())}")
    try:
        if ENV_PATH.exists():
            lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
        elif EXAMPLE_PATH.exists():
            lines = EXAMPLE_PATH.read_text(encoding="utf-8").splitlines()
        else:
            lines = []

        updated_keys = set()
        new_lines = []
        for line in lines:
            m = re.match(r"^(\w+)\s*=\s*(.*)$", line.strip())
            if m:
                key = m.group(1)
                if key in updates:
                    new_lines.append(f"{key}={updates[key]}")
                    updated_keys.add(key)
                    continue
            new_lines.append(line)

        for key, val in updates.items():
            if key not in updated_keys:
                new_lines.append(f"{key}={val}")

        ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return "✅ 设置已保存。部分配置需要重启应用才能生效。"
    except Exception as e:
        return f"❌ 保存失败: {e}"


def _get_current_version():
    """Get current version from git."""
    try:
        r = subprocess.run(
            ["git", "describe", "--tags", "--always"],
            cwd=ROOT_DIR, capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _check_update():
    """Check if remote has updates. Returns (has_update, local, remote)."""
    try:
        subprocess.run(
            ["git", "fetch", "origin"],
            cwd=ROOT_DIR, capture_output=True, text=True, timeout=30
        )
        local = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        remote = subprocess.run(
            ["git", "rev-parse", "origin/main"],
            cwd=ROOT_DIR, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        return local != remote, local[:8], remote[:8]
    except Exception:
        return None, None, None


def _do_update():
    """Execute git pull and return result text."""
    try:
        ver_before = _get_current_version()
        r = subprocess.run(
            ["git", "pull", "origin", "main"],
            cwd=ROOT_DIR, capture_output=True, text=True, timeout=60
        )
        if r.returncode != 0:
            return f"❌ 更新失败:\n{r.stderr}"
        ver_after = _get_current_version()
        if r.stdout.strip() == "Already up to date.":
            return f"✅ 已是最新版本 ({ver_after})"
        return f"✅ 更新成功!\n\n{ver_before} → {ver_after}\n\n```\n{r.stdout}\n```\n\n⚠️ 部分功能需重启应用才能生效。"
    except Exception as e:
        return f"❌ 更新出错: {e}"


def check_and_update_on_startup():
    """Auto-check for updates on startup (if AUTO_UPDATE enabled)."""
    if not config.AUTO_UPDATE:
        return None
    has_update, local, remote = _check_update()
    if has_update:
        return f"📢 发现新版本! 本地: {local} → 远程: {remote}，正在自动更新...\n{_do_update()}"
    elif has_update is False:
        return f"✅ 当前已是最新版本 ({local})"
    return "⚠️ 无法检查更新（网络异常或非 git 环境）"


# ---- Config backup / restore ----
BACKUP_DIR = ROOT_DIR / "outputs" / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# Files to include in a backup (relative to ROOT_DIR, may not all exist)
_BACKUP_FILES = [
    ".env",
    "data/state.json",
    "outputs/history.json",
]

# Directories whose contents are included entirely
_BACKUP_DIRS = [
    "outputs/kb_chats",
]


def _create_backup() -> tuple[str, str | None]:
    """Create a ZIP backup of config and state files.

    Returns (status_message, zip_path_or_None).
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    zip_name = f"backup_{timestamp}.zip"
    zip_path = BACKUP_DIR / zip_name

    added = []
    skipped = []

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add individual files
            for rel in _BACKUP_FILES:
                full = ROOT_DIR / rel
                if full.exists():
                    zf.write(full, rel)
                    added.append(rel)
                else:
                    skipped.append(rel)

            # Add entire directories
            for rel_dir in _BACKUP_DIRS:
                dir_path = ROOT_DIR / rel_dir
                if dir_path.is_dir():
                    for f in sorted(dir_path.rglob("*")):
                        if f.is_file():
                            arc_name = f.relative_to(ROOT_DIR).as_posix()
                            zf.write(f, arc_name)
                            added.append(arc_name)
                    if not any(dir_path.rglob("*")):
                        skipped.append(rel_dir + "/ (empty)")
                else:
                    skipped.append(rel_dir + "/ (not found)")

        size_kb = zip_path.stat().st_size / 1024
        lines = [f"✅ 备份成功: {zip_name}  ({size_kb:.1f} KB)",
                 f"📁 {zip_path}",
                 f"\n包含 {len(added)} 个文件:"]
        for a in added:
            lines.append(f"  - {a}")
        if skipped:
            lines.append(f"\n跳过 {len(skipped)} 项（不存在）:")
            for s in skipped:
                lines.append(f"  - {s}")
        return "\n".join(lines), str(zip_path)
    except Exception as e:
        return f"❌ 备份失败: {e}", None


def _restore_backup(zip_file) -> str:
    """Restore config from an uploaded ZIP file.

    Validates that the ZIP contains at least one expected file before extracting.
    """
    if zip_file is None:
        return "❌ 请先上传备份 ZIP 文件"

    # Gradio gives us a temp file path (str) or a file-like object
    zip_path = Path(zip_file) if isinstance(zip_file, str) else Path(zip_file.name)

    if not zip_path.exists():
        return f"❌ 文件不存在: {zip_path}"
    if not zipfile.is_zipfile(zip_path):
        return "❌ 不是有效的 ZIP 文件"

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = set(zf.namelist())

            # Validate: check that at least one expected file is inside
            expected_set = set(_BACKUP_FILES)
            for rel_dir in _BACKUP_DIRS:
                # Any path starting with the dir counts
                for n in names:
                    if n.startswith(rel_dir + "/") or n.startswith(rel_dir + "\\"):
                        expected_set.add(n)
                        break

            found = expected_set & names
            if not found:
                return ("❌ ZIP 中未找到任何可识别的配置文件。\n"
                        f"期望包含: {', '.join(_BACKUP_FILES)}")

            restored = []
            for name in names:
                # Security: prevent path traversal
                target = (ROOT_DIR / name).resolve()
                if not str(target).startswith(str(ROOT_DIR.resolve())):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                restored.append(name)

        return (f"✅ 恢复成功！已还原 {len(restored)} 个文件:\n"
                + "\n".join(f"  - {r}" for r in restored)
                + "\n\n⚠️ 部分配置需要重启应用才能生效。")
    except Exception as e:
        return f"❌ 恢复失败: {e}"


def render_tab_settings():
    """Render the settings tab. Returns component dict for external wiring."""
    gr.Markdown(_make_title("全局配置 — 在线编辑 .env 文件"))
    gr.Markdown("在此修改 API 地址、模型名称、并发参数等。点「保存」后写入 `.env` 文件，**部分配置需重启应用才能生效**。")

    current_env = _read_env()
    setting_inputs: dict = {}

    _LEFT_SECTIONS = {"🔗 API 连接", "🤖 模型名称", "⚡ 并发与超时"}

    def _render_field(key, label, default_val, kind):
        cur_val = current_env.get(key, default_val)
        if kind == "password":
            return gr.Textbox(label=label, value=cur_val, type="password")
        elif kind == "bool":
            bool_default = cur_val.lower() == "true"
            return gr.Dropdown(
                label=label, value=str(bool_default),
                choices=["True", "False"],
                info="True = 启动时自动 git pull 检查更新；False = 仅手动更新"
            )
        elif kind in ("int", "float"):
            return gr.Textbox(label=label, value=str(cur_val), placeholder=default_val)
        else:
            return gr.Textbox(label=label, value=cur_val, placeholder=default_val)

    with gr.Row():
        with gr.Column():
            for section_name, fields in SETTINGS_SCHEMA:
                if section_name in _LEFT_SECTIONS:
                    with gr.Accordion(section_name, open=False):
                        for key, label, default_val, kind in fields:
                            setting_inputs[key] = _render_field(key, label, default_val, kind)
        with gr.Column():
            for section_name, fields in SETTINGS_SCHEMA:
                if section_name not in _LEFT_SECTIONS:
                    with gr.Accordion(section_name, open=False):
                        for key, label, default_val, kind in fields:
                            setting_inputs[key] = _render_field(key, label, default_val, kind)

    save_btn = gr.Button("💾 保存设置", variant="primary", scale=0)
    save_msg = gr.Textbox(label="", interactive=False, container=False, show_label=False)

    # Reset defaults
    gr.Markdown("---")
    gr.Markdown(_make_title("🔄 恢复默认设置"))
    gr.Markdown("清除所有工具的参数记忆（输入目录、高级选项等），下次启动时恢复为默认值。")
    with gr.Row():
        reset_btn = gr.Button("🔄 恢复默认设置", variant="stop", scale=0)

        def _on_reset():
            config.clear_state()
            return "✅ 已清除所有保存的参数。请重启应用以加载默认值。"

        reset_btn.click(_on_reset, outputs=[save_msg])

    # Collect inputs in fixed order
    _input_keys = []
    _input_widgets = []
    for _, fields in SETTINGS_SCHEMA:
        for key, _, _, _ in fields:
            _input_keys.append(key)
            _input_widgets.append(setting_inputs[key])

    def _on_save(*values):
        updates = dict(zip(_input_keys, values))
        return _save_env(updates)

    save_btn.click(_on_save, _input_widgets, [save_msg])

    # ---- Theme color ----
    gr.Markdown("---")
    gr.Markdown(_make_title("🎨 主题色彩"))
    gr.Markdown("选择主色调，即时预览并自动保存到浏览器。")

    gr.HTML("""
    <div style="display:flex;align-items:center;gap:16px;padding:8px 0;flex-wrap:wrap;">
      <label style="font-weight:500;">主色调</label>
      <input type="color" id="theme_picker" value="#4f8ef7"
             style="width:48px;height:36px;border:1px solid #ccc;border-radius:6px;cursor:pointer;" />
      <span id="theme_hex" style="font-family:monospace;font-size:0.9em;">#4f8ef7</span>
      <button id="theme_reset" style="padding:4px 12px;border:1px solid #ccc;border-radius:6px;background:#fff;cursor:pointer;font-size:0.85em;">
        恢复默认
      </button>
    </div>
    <script>
    (function(){
        var picker = document.getElementById('theme_picker');
        var hexLabel = document.getElementById('theme_hex');
        var resetBtn = document.getElementById('theme_reset');
        var DEFAULT = '#4f8ef7';
        // Init from localStorage
        var saved = localStorage.getItem('theme_accent');
        if (saved) { picker.value = saved; hexLabel.textContent = saved; }
        picker.addEventListener('input', function(){
            var v = picker.value;
            hexLabel.textContent = v;
            if (window.setThemeColor) window.setThemeColor(v);
        });
        resetBtn.addEventListener('click', function(){
            picker.value = DEFAULT;
            hexLabel.textContent = DEFAULT;
            if (window.setThemeColor) window.setThemeColor(DEFAULT);
        });
    })();
    </script>
    """)

    # ---- Local model ----
    gr.Markdown("---")
    gr.Markdown(_make_title("🏠 内置本地模型"))
    gr.Markdown("无需安装 LM Studio 等外部服务，直接在应用内运行小模型。适合轻量任务（分类、重命名等）。")

    from services.local_model import get_status, start_server_simple, is_available, get_model_path

    status = get_status()
    _models_dir = Path(__file__).parent.parent / "models"

    if not status["available"]:
        gr.Markdown(
            "**状态：** ❌ 未安装 `llama-cpp-python`\n\n"
            "安装命令：\n```\npip install llama-cpp-python\n```\n"
            "> 安装后重启应用即可使用"
        )
    elif not status["model_path"]:
        gr.Markdown(
            f"**状态：** ⚠️ 已安装 llama-cpp-python，但未找到模型文件\n\n"
            f"请将 `.gguf` 模型文件放入 `{_models_dir}` 目录\n\n"
            f"推荐下载：\n"
            f"- [Qwen3.5-0.6B](https://huggingface.co/Qwen/Qwen3.5-0.6B-GGUF) (~400MB，速度快)\n"
            f"- [Qwen3.5-1.7B](https://huggingface.co/Qwen/Qwen3.5-1.7B-GGUF) (~1GB，更智能)"
        )
    else:
        _model_name = Path(status["model_path"]).name
        _model_size = Path(status["model_path"]).stat().st_size / (1024*1024)
        _status_text = "🟢 运行中" if status["running"] else "⚪ 未启动"
        gr.Markdown(
            f"**状态：** {_status_text}\n\n"
            f"**模型：** `{_model_name}` ({_model_size:.0f} MB)\n\n"
            f"**地址：** `{status['base_url'] or '未启动'}`"
        )
        with gr.Row():
            local_start_btn = gr.Button("▶️ 启动本地模型", variant="primary", scale=1)
            local_stop_btn = gr.Button("⏹ 停止", variant="stop", scale=0, min_width=80)
        local_status_msg = gr.Textbox(label="", interactive=False, container=False,
                                       show_label=False, max_lines=3)

        def _on_start_local():
            result = start_server_simple()
            return result

        def _on_stop_local():
            from services.local_model import _server_running, _server_port
            return "⚠️ 本地模型随应用关闭而停止（重启应用即可）"

        local_start_btn.click(_on_start_local, outputs=[local_status_msg])
        local_stop_btn.click(_on_stop_local, outputs=[local_status_msg])

    # ---- API connection test ----
    gr.Markdown("---")
    gr.Markdown(_make_title("🔗 API 连接测试"))

    with gr.Row(equal_height=True):
        with gr.Column(scale=2):
            test_base = gr.Textbox(label="API 地址", value=config.OPENAI_BASE_URL,
                                   info="与上方「API 连接」分组中的地址保持一致")
        with gr.Column(scale=2):
            test_key = gr.Textbox(label="API 密钥", value=config.OPENAI_API_KEY, type="password",
                                  info="本地服务填任意值即可")

    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            test_model_type, test_model, test_fetch_btn, test_fetch_st, test_thinking = _make_model_selector(
                "测试模型（可选）", config.VISION_MODEL,
                "检查该模型是否可用，留空则只测试连接", show_thinking=False)
        with gr.Column(scale=1, min_width=120):
            test_btn = gr.Button("🔗 测试连接", variant="secondary")
    test_msg = gr.Textbox(label="测试结果", interactive=False, lines=4, elem_classes="output-text",
                          placeholder="点击「测试连接」查看结果...")

    test_btn.click(_test_api_connection, [test_base, test_key, test_model], [test_msg])
    _bind_model_fetch_local(test_fetch_btn, test_model_type, test_model, test_fetch_st,
                            test_base, test_key, config.VISION_MODEL)

    # ---- Manual update ----
    gr.Markdown("---")
    gr.Markdown(_make_title("📦 手动更新"))
    gr.Markdown(f"当前版本: `{_get_current_version()}`")

    with gr.Row():
        check_btn = gr.Button("🔍 检查更新", variant="secondary", scale=1)
        update_btn = gr.Button("⬇️ 立即更新", variant="primary", scale=1)
    update_output = gr.Textbox(label="更新日志", lines=6, elem_classes="output-text",
                               placeholder="检查/更新结果会显示在这里...")

    def _on_check():
        has_update, local, remote = _check_update()
        if has_update is None:
            return "⚠️ 无法检查更新（网络异常或非 git 环境）"
        if has_update:
            return f"🔔 发现新版本! 本地: {local} → 远程: {remote}"
        return f"✅ 当前已是最新版本 (本地: {local})"

    check_btn.click(_on_check, [], [update_output])
    update_btn.click(_do_update, [], [update_output])

    # ---- Config backup / restore ----
    gr.Markdown("---")
    gr.Markdown(_make_title("📦 配置备份与恢复"))
    gr.Markdown(
        "备份：将 `.env`、工具参数、聊天记录等打包为 ZIP 文件。\n"
        "恢复：上传之前下载的 ZIP 文件，还原所有配置。"
    )

    with gr.Row():
        backup_btn = gr.Button("📥 创建备份", variant="primary", scale=1)
        restore_btn = gr.Button("📤 恢复备份", variant="secondary", scale=1)

    backup_msg = gr.Textbox(label="操作结果", interactive=False, lines=6,
                            elem_classes="output-text", placeholder="点击按钮执行备份或恢复...")
    backup_file = gr.File(label="下载备份文件", interactive=False, visible=True)
    restore_upload = gr.File(label="上传备份 ZIP", file_types=[".zip"], visible=True)

    backup_btn.click(_create_backup, [], [backup_msg, backup_file])
    restore_btn.click(_restore_backup, [restore_upload], [backup_msg])

    return {
        "save_btn": save_btn,
        "save_msg": save_msg,
    }
