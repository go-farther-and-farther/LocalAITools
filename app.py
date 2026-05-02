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
import subprocess
from pathlib import Path
from contextlib import redirect_stdout

sys.path.insert(0, str(Path(__file__).parent))
import config
import history

import gradio as gr

ROOT_DIR = Path(__file__).parent


def _make_title(text):
    return f"## {text}"


def _capture_log(fn, *args, **kwargs):
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


# ============================================================
# Tab 1: 图片重命名
# ============================================================
def _rename_images(input_dir, model, workers, dry_run, progress=gr.Progress()):
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
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one_image, img, model or config.RENAME_MODEL, dry_run, recent_history): img
                   for img in images}
        for i, future in enumerate(as_completed(futures)):
            try:
                old_name, new_phrase = future.result()
                results.append(f"{old_name} → {new_phrase}")
            except Exception as e:
                results.append(f"❌ 错误: {e}")
            completed += 1
            progress(completed / total, desc=f"重命名中 {completed}/{total}")

    result = f"处理完成：{total} 张图片\n\n" + "\n".join(results)
    history.add_entry("图片重命名", input_dir, f"处理 {total} 张图片")
    return result


# ============================================================
# Tab 2: 图片质量评分与分类（合并原 Tab 2 + Tab 3）
# ============================================================
def _detect_and_classify(input_dir, mode, top_percent, bottom_percent, custom_prompt, progress=gr.Progress()):
    from image_tools.detect_ai_errors import process_and_classify

    def on_progress(completed, total):
        progress(completed / total, desc=f"评分中 {completed}/{total}")

    result = _capture_log(process_and_classify, input_dir, mode, on_progress,
                           top_percent / 100, bottom_percent / 100, custom_prompt)
    mode_label = "漫展摄影" if mode == "photo" else "AI图片检测"
    history.add_entry(f"质量评分({mode_label})", input_dir, "评分分类完成")
    return result


# ============================================================
# Tab 3: 聊天截图识别
# ============================================================
def _explain_images(input_dir, vision_model, temperature, workers, internal_workers, max_tokens,
                    progress=gr.Progress()):
    from image_tools.ocr_chat_screenshots import process_folder

    input_path = Path(input_dir)
    if not input_path.is_dir():
        return "❌ 请输入有效的文件夹路径"

    output_dir = input_path / "chat_text_output"
    output_dir.mkdir(exist_ok=True)

    def on_progress(completed, total):
        progress(completed / total, desc=f"识别中 {completed}/{total}")

    buf = io.StringIO()
    with redirect_stdout(buf):
        process_folder(
            input_path,
            output_dir,
            vision_model or config.VISION_MODEL_THINKING,
            temperature,
            workers,
            max_tokens,
            internal_workers,
            progress_callback=on_progress
        )

    result = f"✅ 输出目录: {output_dir}\n\n{buf.getvalue()}"
    history.add_entry("截图识别", input_dir, "文字提取完成")
    return result


# ============================================================
# Tab 4: 聊天记录压缩
# ============================================================
def _compress_text(input_path, model, temperature, chunk_size, internal_workers, max_tokens,
                   progress=gr.Progress()):
    from text_tools.compress_chat import process_single_text_file, process_folder

    def on_progress(completed, total):
        progress(completed / total, desc=f"压缩中 {completed}/{total}")

    p = Path(input_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        if p.is_file():
            process_single_text_file(
                p, None,
                model or config.VISION_MODEL_THINKING,
                temperature, max_tokens, chunk_size, internal_workers,
                progress_callback=on_progress
            )
        elif p.is_dir():
            process_folder(
                p, None,
                model or config.VISION_MODEL_THINKING,
                temperature, max_tokens, chunk_size, internal_workers,
                progress_callback=on_progress
            )
        else:
            return "❌ 路径无效"

    result = f"✅ 处理完成\n\n{buf.getvalue()}"
    history.add_entry("聊天压缩", input_path, "压缩完成")
    return result


# ============================================================
# Tab 5: 文本翻译
# ============================================================
def _translate(input_file, output_file, model, batch_size, workers, progress=gr.Progress()):
    from text_tools.translate import translate_book_parallel

    if not Path(input_file).is_file():
        return "❌ 输入文件不存在"

    output_file = output_file or str(config.OUTPUT_DIR / "translation" / "translation.txt")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    def on_progress(completed, total):
        progress(completed / total, desc=f"翻译中 {completed}/{total}")

    buf = io.StringIO()
    with redirect_stdout(buf):
        translate_book_parallel(
            input_file, output_file,
            model or config.TEXT_MODEL,
            batch_size, workers,
            resume=True,
            progress_callback=on_progress
        )

    result = f"✅ 输出: {output_file}\n\n{buf.getvalue()}"
    history.add_entry("文本翻译", input_file, "翻译完成")
    return result


# ============================================================
# Tab 6: 知识库问答
# ============================================================
def _query_kb(query, keyword, model, k, batch_size, progress=gr.Progress()):
    from text_tools.chapter_summary import query_knowledge_base

    if not query.strip():
        return "❌ 请输入查询问题"

    progress_lines = []
    # query_knowledge_base 用轮数作为进度，预估约 3-5 轮
    round_count = [0]

    def on_progress(msg: str):
        progress_lines.append(msg)
        round_count[0] += 1
        # 模拟进度，每轮推进一部分
        pct = min(0.95, round_count[0] / max(3, 1))
        progress(pct, desc=f"检索回答中 (第{round_count[0]}轮)")

    answer = query_knowledge_base(
        query=query,
        keyword=keyword,
        model=model or None,
        k=int(k),
        batch_size=int(batch_size),
        progress_callback=on_progress,
    )

    progress(1.0, desc="完成")
    progress_text = "\n".join(f"⏳ {l}" for l in progress_lines)
    result = f"{progress_text}\n\n{answer}"
    history.add_entry("知识库问答", query[:50], "查询完成")
    return result


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
    history.add_entry("LLM压测", model or url, "压测完成")
    plot_path = Path(save_plot)
    if plot_path.exists():
        return text_output, str(plot_path)
    return text_output, None


# ============================================================
# 设置页：读写 .env
# ============================================================
ENV_PATH = Path(__file__).parent / ".env"
EXAMPLE_PATH = Path(__file__).parent / ".env.example"

# 需要显示在设置页的配置项及分组
SETTINGS_SCHEMA = [
    ("🔗 API 连接", [
        ("OPENAI_BASE_URL", "API 地址", "http://localhost:1234/v1", "text"),
        ("OPENAI_API_KEY", "API 密钥", "lm-studio", "password"),
    ]),
    ("🧠 推理设置", [
        ("ENABLE_THINKING", "启用思考模式（Thinking）", "true", "bool"),
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
    ]),
]


def _read_env() -> dict:
    """读取 .env 文件返回 {KEY: value}"""
    if not ENV_PATH.exists():
        return {}
    import re
    result = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(\w+)\s*=\s*(.*)$", line)
        if m:
            result[m.group(1)] = m.group(2).strip()
    return result


def _save_env(updates: dict) -> str:
    """将 updates 中的键值写入 .env 文件，保留原有注释和其他行"""
    try:
        import re
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

        # 追加新增的 key
        for key, val in updates.items():
            if key not in updated_keys:
                new_lines.append(f"{key}={val}")

        ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return "✅ 设置已保存。部分配置需要重启应用才能生效。"
    except Exception as e:
        return f"❌ 保存失败: {e}"


def _get_current_version():
    """获取当前版本号"""
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
    """检查远程是否有更新，返回 (has_update, local, remote)"""
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
    except Exception as e:
        return None, None, None

def _do_update():
    """执行 git pull 更新，返回结果文本"""
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

def _check_and_update_on_startup():
    """启动时自动检查更新（如果开启了 AUTO_UPDATE）"""
    if not config.AUTO_UPDATE:
        return None
    has_update, local, remote = _check_update()
    if has_update:
        return f"📢 发现新版本! 本地: {local} → 远程: {remote}，正在自动更新...\n{_do_update()}"
    elif has_update is False:
        return f"✅ 当前已是最新版本 ({local})"
    return "⚠️ 无法检查更新（网络异常或非 git 环境）"

def _test_api_connection(base_url, api_key, model):
    """测试 API 连接是否正常"""
    if not base_url.strip():
        return "❌ 请填写 API 地址"
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url.strip(), api_key=api_key.strip() or "no-key")
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

        # ==================== Tab 0: 开始使用（新手引导） ====================
        with gr.Tab("🏠 开始使用"):
            gr.Markdown(_make_title("欢迎使用 LocalAITools！"))
            gr.Markdown("一套调用本地大模型的 AI 工具集，**所有功能免费、数据不上传云端**。")

            gr.Markdown("---")

            gr.Markdown("### 🔌 第一步：获取 AI 模型服务")

            gr.Markdown("""本工具需要连接一个 AI 模型服务才能工作。推荐以下方式（任选一种）：

**🟢 方式一：LM Studio（推荐新手）**
1. 下载安装 [LM Studio](https://lmstudio.ai/)（支持 Windows / Mac）
2. 打开 LM Studio，在搜索框搜 `qwen3` 或 `qwen3.6`
3. 下载一个视觉模型（如 `qwen3.6-27b`，约 16 GB）
4. 切换到 **Local Server** 标签页，点击 **Start Server**
5. 默认地址就是 `http://localhost:1234/v1`，无需修改

**🟡 方式二：云端 API（免下载模型，需付费）**
1. [硅基流动 SiliconFlow](https://cloud.siliconflow.cn/) — 注册送额度，支持 Qwen 系列
2. [DeepSeek 开放平台](https://platform.deepseek.com/) — 便宜好用
3. [阿里云百炼](https://bailian.console.aliyun.com/) — Qwen 官方 API
4. 注册后在后台创建 API Key，填入下方的 API 地址和密钥

**🟢 方式三：Ollama（进阶用户）**
```bash
ollama serve          # 启动服务
ollama pull qwen3     # 下载模型
```
默认地址 `http://localhost:11434/v1`
""")

            gr.Markdown("---")

            gr.Markdown("### ⚙️ 第二步：填写配置")
            gr.Markdown("""切换到 **⚙️ 设置** 标签页，填写你的 API 信息：

- **API 地址**：LM Studio 默认 `http://localhost:1234/v1`，云端 API 填对应地址
- **API 密钥**：本地服务填任意值（如 `lm-studio`），云端 API 填真实的 Key
- 填好后点 **💾 保存设置**，然后点 **🔗 测试连接** 确认能连上

> 💡 如果使用 LM Studio，默认配置**无需修改**，直接测试连接即可！""")

            gr.Markdown("---")

            gr.Markdown("### 🎯 第三步：开始使用")
            gr.Markdown("""配置完成后，切换到对应的功能标签页即可使用：

| Tab | 功能 | 需要什么模型 | 输入 → 输出 |
|-----|------|-------------|-------------|
| 🖼️ 图片重命名 | AI 看图起中文名 | 视觉模型 | 图片文件夹 → 文件重命名 |
| 🔍 质量评分分类 | 评分 + 自动分拣 | 视觉模型 | 图片文件夹 → HighQuality / LowQuality_Errors |
| 💬 截图识别 | 聊天截图 → 文字 | 视觉模型（Thinking 最佳） | 截图 → TXT 文件 |
| 📝 聊天压缩 | 合并冗余时间戳 | 文本模型 | TXT → 精简 TXT |
| 🌐 文本翻译 | 长篇章节翻译 | 文本模型 | TXT → 翻译 TXT |
| 📚 知识库问答 | RAG 混合检索 | 文本 + Embedding | 问题 → 答案 |
| ⚡ LLM 压测 | API 吞吐量测试 | 被测模型 | 参数 → 图表 |

> 💡 每个标签页顶部都有「📖 使用方法」折叠面板，点开即可查看详细步骤。
""")

            gr.Markdown("---")
            gr.Markdown('<div style="text-align:center;color:#888;font-size:0.85em">'
                        '遇到问题？<a href="https://github.com/go-farther-and-farther/LocalAITools/issues" target="_blank">GitHub Issues</a>'
                        '</div>')

            # ---- 历史记录 ----
            gr.Markdown("---")
            with gr.Accordion("📋 处理历史", open=False):
                history_md = gr.Markdown("")

            def _load_history():
                entries = history.get_recent(20)
                if not entries:
                    return "暂无处理记录。\n\n每次使用工具处理文件后，这里会自动记录。"
                lines = ["| 时间 | 工具 | 输入 | 摘要 |",
                         "|------|------|------|------|"]
                for e in entries:
                    inp = str(e["input"])[:40]
                    lines.append(f"| {e['time']} | {e['tool']} | {inp} | {e['summary']} |")
                return "\n".join(lines)

            # 每次切换到该 Tab 时刷新历史
            app.load(_load_history, outputs=[history_md])

        # ==================== Tab 1: 图片重命名 ====================
        with gr.Tab("🖼️ 图片重命名"):
            gr.Markdown(_make_title("图片 AI 重命名 — 用中文短句替代杂乱的文件名"))
            with gr.Accordion("📖 使用方法", open=False):
                gr.Markdown("**步骤：** ① 把图片放进文件夹 → ② 点「开始重命名」→ ③ 文件名变成中文描述短语\n\n"
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
            gr.Markdown(_make_title("AI 评分 + 自动分拣 — 多维度评估，高分/低分图片自动归类"))
            with gr.Accordion("📖 使用方法", open=False):
                gr.Markdown("**步骤：** ① 选择检测模式 → ② 把图片放进文件夹 → ③ 点「开始评分分类」→ 优质图片自动移至 `HighQuality/`，劣质/错误图片移至 `LowQuality_Errors/`\n\n"
                            "**两种模式：**\n"
                            "- 🎨 **AI 图片错误检测** — 检测 AI 生成图的肢体错乱、面部畸形、结构崩坏等问题\n"
                            "- 📸 **漫展摄影筛选** — 检测跑焦模糊、过曝欠曝等拍摄问题，筛选可出片的 Cosplay 照片\n\n"
                            "> 💡 每张图会生成同名 `.txt` 评分文件，方便核对\n"
                            "> 🎛️ 下方的分拣比例和评分标准均可自由调整")
            with gr.Row():
                with gr.Column(scale=2):
                    de_mode = gr.Dropdown(
                        label="检测模式",
                        choices=[("🎨 AI 图片错误检测", "ai"), ("📸 漫展摄影筛选", "photo")],
                        value="ai",
                        info="AI 错误检测：找肢体畸形/结构崩坏 | 漫展摄影：找跑焦/过曝/欠曝"
                    )
                    de_input = gr.Textbox(label="图片文件夹",
                                          value=str(config.DATA_DIR / "images"),
                                          placeholder="粘贴图片所在文件夹的完整路径",
                                          info="处理完成后会在该文件夹下生成 HighQuality/ 和 LowQuality_Errors/ 子文件夹")

                    with gr.Accordion("🎛️ 分拣规则 & 提示词", open=False):
                        with gr.Row():
                            de_top = gr.Slider(1, 50, value=int(config.TOP_PERCENT * 100), step=1,
                                               label=f"高分比例（%）",
                                               info="评分前 N% 的图片移入 HighQuality 目录")
                            de_bottom = gr.Slider(1, 50, value=int(config.BOTTOM_PERCENT * 100), step=1,
                                                  label="低分比例（%）",
                                                  info="评分后 N% 的图片 + 所有 ERR 图片移入 LowQuality_Errors")
                        de_prompt = gr.Textbox(
                            label="自定义评分提示词（留空使用默认）",
                            value="",
                            lines=8,
                            placeholder="在此输入自定义评分标准...\n\n留空则使用当前模式的默认提示词。\n修改提示词可以：调整评分宽松度、改变关注重点、自定义输出格式等。\n\n提示：切换检测模式后请清空此框或重新填写对应模式的提示词。",
                            info="留空 = 使用默认提示词。填写后覆盖默认，适合有经验的用户微调评分标准。"
                        )

                    de_btn = gr.Button("开始评分分类", variant="primary")
                with gr.Column(scale=3):
                    de_output = gr.Textbox(label="处理日志", lines=15, elem_classes="output-text",
                                           placeholder="处理完成后这里会显示每张图的评分和分类结果...")
            de_btn.click(_detect_and_classify, [de_input, de_mode, de_top, de_bottom, de_prompt], [de_output])

        # ==================== Tab 3: 聊天截图识别 ====================
        with gr.Tab("💬 聊天截图识别"):
            gr.Markdown(_make_title("聊天记录长截图 → 文字提取 — 切片 + VLM 识别"))
            with gr.Accordion("📖 使用方法", open=False):
                gr.Markdown("**步骤：** ① 把微信/QQ 聊天长截图放进文件夹 → ② 点「开始识别」→ ③ 去 `chat_text_output/` 找生成的 TXT 文件\n\n"
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
            with gr.Accordion("📖 使用方法", open=False):
                gr.Markdown("**步骤：** ① 把截图 OCR 生成的 TXT（或任意聊天记录 TXT）放进文件夹 → ② 点「开始压缩」→ ③ 得到 `.compressed.txt` 文件\n\n"
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
            with gr.Accordion("📖 使用方法", open=False):
                gr.Markdown("**步骤：** ① 把要翻译的文本文件路径填好 → ② 点「开始翻译」→ ③ 去 `outputs/translation/` 找译文\n\n"
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
            with gr.Accordion("📖 使用方法", open=False):
                gr.Markdown("**步骤：** ① 预先用文档构建好 FAISS 索引（`.env` 中配置 `FAISS_INDEX_PATH`）→ ② 输入问题 → ③ 点「开始查询」→ ④ LLM 基于检索到的相关段落多轮迭代生成答案\n\n"
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
            with gr.Accordion("📖 使用方法", open=False):
                gr.Markdown("**步骤：** ① 确认 API 地址和模型名称 → ② 点「开始压测」→ ③ 查看吞吐量图表，找到模型性能瓶颈\n\n"
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

        # ==================== Tab 8: 设置 ====================
        with gr.Tab("⚙️ 设置"):
            gr.Markdown(_make_title("全局配置 — 在线编辑 .env 文件"))
            gr.Markdown("在此修改 API 地址、模型名称、并发参数等。点「保存」后写入 `.env` 文件，**部分配置需重启应用才能生效**。")

            current_env = _read_env()
            setting_inputs: dict = {}

            with gr.Column():
                for section_name, fields in SETTINGS_SCHEMA:
                    with gr.Accordion(section_name, open=False):
                        for key, label, default_val, kind in fields:
                            cur_val = current_env.get(key, default_val)
                            if kind == "password":
                                inp = gr.Textbox(label=label, value=cur_val, type="password")
                            elif kind == "bool":
                                bool_default = cur_val.lower() == "true"
                                if key == "ENABLE_THINKING":
                                    bool_info = "True = 模型先思考再回答（适合复杂任务）；False = 直接输出（更快，适合简单任务）"
                                else:
                                    bool_info = "True = 启动时自动 git pull 检查更新；False = 仅手动更新"
                                inp = gr.Dropdown(
                                    label=label, value=str(bool_default),
                                    choices=["True", "False"],
                                    info=bool_info
                                )
                            elif kind in ("int", "float"):
                                inp = gr.Textbox(label=label, value=str(cur_val), placeholder=default_val)
                            else:
                                inp = gr.Textbox(label=label, value=cur_val, placeholder=default_val)
                            setting_inputs[key] = inp

                save_btn = gr.Button("💾 保存设置", variant="primary", scale=0)
                save_msg = gr.Textbox(label="", interactive=False, container=False, show_label=False)

            # 按固定顺序收集所有输入组件及其对应的 key
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

            # ---- API 连接测试 ----
            gr.Markdown("---")
            gr.Markdown(_make_title("🔗 API 连接测试"))
            gr.Markdown("测试你的 API 地址和密钥是否能正常连接，并列出可用模型。")

            with gr.Group():
                test_base = gr.Textbox(label="API 地址", value=config.OPENAI_BASE_URL,
                                       info="与上方「API 连接」分组中的地址保持一致")
                test_key = gr.Textbox(label="API 密钥", value=config.OPENAI_API_KEY, type="password",
                                      info="本地服务填任意值即可")
                test_model = gr.Textbox(label="模型名称（可选）", value=config.VISION_MODEL,
                                        info="测试时仅检查该模型是否在可用列表中，留空则只列出模型")
                test_btn = gr.Button("🔗 测试连接", variant="secondary")
                test_msg = gr.Textbox(label="测试结果", interactive=False, lines=5, elem_classes="output-text",
                                      placeholder="测试结果会显示在这里...")

            test_btn.click(_test_api_connection, [test_base, test_key, test_model], [test_msg])

            # ---- 手动更新区域 ----
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

    return app


if __name__ == "__main__":
    import traceback
    try:
        update_msg = _check_and_update_on_startup()
        if update_msg:
            print(update_msg)
        build_ui().launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
    except Exception:
        print("\n❌ 启动失败：\n")
        traceback.print_exc()
        print("\n按 Enter 键退出...")
        input()
