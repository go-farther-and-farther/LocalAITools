"""Image tools tab: rename, score, classify, manual review."""
import config
import history
import gradio as gr
from ui.common import _make_title, _apply_provider, _make_model_selector, _bind_model_fetch
from services.image_services import rename_images as _svc_rename, classify_by_work as _svc_cls_work
from services.image_services import score_images as _svc_score, classify_images as _svc_cls_images
from services.image_services import ai_classify_images as _svc_ai_classify
from services.image_services import apply_rename_results as _svc_apply_rename, apply_classify_results as _svc_apply_classify


# ============================================================
# Thin UI wrappers — delegate to service layer
# ============================================================

def _rename_images(input_dir, model, workers, dry_run, keep_original, rename_mode, custom_prompt, context_count, max_size, provider, thinking=True, progress=gr.Progress()):
    _apply_provider(provider)
    result = _svc_rename(input_dir, model, workers, dry_run, keep_original, custom_prompt, context_count, max_size, thinking,
                         progress_callback=lambda c, t: progress(c / t, desc=f"重命名中 {c}/{t}"),
                         rename_mode=rename_mode)
    config.save_state("rename", input_dir=input_dir, model=model or config.RENAME_MODEL,
                      workers=workers, dry_run=dry_run, keep_original=keep_original,
                      rename_mode=rename_mode, custom_prompt=custom_prompt, context_count=context_count,
                      thinking=thinking)
    history.add_entry("图片重命名", input_dir, "重命名完成")
    return result


def _classify_by_work(input_dir, dry_run, min_count):
    return _svc_cls_work(input_dir, dry_run, int(min_count))


def _ai_classify_images(input_dir, checked_cats, custom_cats, custom_desc, model, max_size, dry_run, provider, thinking=True, workers=4, group_similar=False, similarity_threshold=20, max_samples_per_group=4, progress=gr.Progress()):
    _apply_provider(provider)
    # Build categories string from checked + custom
    _PRESET_DESC = {
        "照片": "真实拍摄的照片，如风景、人像、生活照",
        "动漫": "日本动漫、漫画风格的图片和同人插画",
        "游戏": "电子游戏相关的图片，包括游戏角色、截图、同人图",
        "绘画": "手绘、数字绘画、插画、原画等艺术作品",
        "聊天截图": "微信、QQ、Discord 等聊天软件的截图",
        "应用截图": "软件界面、系统设置、手机桌面等屏幕截图",
        "风景": "自然风光、城市景观、旅行摄影",
        "美食": "食物、饮品、餐厅相关图片",
        "文档": "扫描件、证件、课件、合同等文档照片",
        "其他": "不属于以上任何分类",
    }
    parts = []
    for name in (checked_cats or []):
        desc = _PRESET_DESC.get(name, "")
        parts.append(f"{name}：{desc}" if desc else name)
    if custom_cats:
        for c in custom_cats.split(","):
            c = c.strip()
            if c:
                parts.append(c)
    if custom_desc:
        for c in custom_desc.split(","):
            c = c.strip()
            if c:
                parts.append(c)
    if len(parts) < 2:
        return "❌ 请至少勾选两个分类"
    categories = ",".join(parts)

    result = _svc_ai_classify(
        input_dir, categories, model, max_size, dry_run, thinking,
        workers=workers, group_similar=group_similar,
        similarity_threshold=similarity_threshold,
        max_samples_per_group=max_samples_per_group,
        progress_callback=lambda c, t: progress(c / t, desc=f"分类中 {c}/{t}"),
    )
    config.save_state("ai_classify", input_dir=input_dir, categories=categories, model=model,
                      max_size=max_size, dry_run=dry_run, workers=workers, thinking=thinking,
                      checked_cats=checked_cats, custom_cats=custom_cats, custom_desc=custom_desc,
                      group_similar=group_similar, similarity_threshold=similarity_threshold,
                      max_samples_per_group=max_samples_per_group)
    history.add_entry("AI智能分类", input_dir, "分类完成")
    return result


def _score_images(input_dir, mode, custom_prompt, model, max_size, provider, thinking=True, progress=gr.Progress()):
    _apply_provider(provider)
    log_text, structured = _svc_score(input_dir, mode, custom_prompt, model, max_size, thinking,
                                      progress_callback=lambda c, t: progress(c / t, desc=f"评分中 {c}/{t}"))
    config.save_state("score", input_dir=input_dir, mode=mode, model=model, custom_prompt=custom_prompt,
                      max_size=max_size, thinking=thinking)
    mode_labels = {"ai":"AI检测","photo":"漫展摄影","general":"通用照片","portrait":"人像","landscape":"风景","document":"文档扫描","art":"绘画插图"}
    history.add_entry(f"质量评分({mode_labels.get(mode, mode)})", input_dir, "评分完成")
    return log_text, structured


def _classify_images(input_dir, classify_method, top_percent, bottom_percent, min_score, max_score, progress=gr.Progress()):
    result = _svc_cls_images(input_dir, classify_method, top_percent, bottom_percent, min_score, max_score)
    config.save_state("classify", input_dir=input_dir, classify_method=classify_method,
                      top_percent=top_percent, bottom_percent=bottom_percent,
                      min_score=min_score, max_score=max_score)
    history.add_entry("质量分类", input_dir, "分类完成")
    return result


_GALLERY_MAX = 200


def _populate_gallery(structured_results):
    """Build gallery items from structured score results (capped at _GALLERY_MAX)."""
    if not structured_results:
        return [], "暂无评分结果，请先执行评分"
    gallery_items = []
    cls_labels = {"high": "高质量", "mid": "中等", "low": "低质量"}
    for item in structured_results[:_GALLERY_MAX]:
        path = item["path"]
        score = item["score"]
        error = item["error"]
        cls = item["classification"]
        if error:
            caption = f"❌ {error}"
        elif score is not None:
            caption = f"{score} - {cls_labels.get(cls, cls)}"
        else:
            caption = "未知"
        gallery_items.append((path, caption))
    total = len(structured_results)
    info = f"共 {total} 张图片（按分数从高到低排列）"
    if total > _GALLERY_MAX:
        info += f"，画廊仅显示前 {_GALLERY_MAX} 张"
    return gallery_items, info


def _load_scored_to_gallery(input_dir):
    """Load scored results from existing .txt files and return gallery items."""
    from image_tools.detect_ai_errors import load_scored_results
    results = load_scored_results(input_dir)
    gallery_items, info_text = _populate_gallery(results)
    return gallery_items, info_text, results


def _export_csv(structured_results):
    """Export scored results to a CSV file in outputs/ directory."""
    import csv
    from datetime import datetime
    if not structured_results:
        return gr.update(visible=False), "❌ 无评分结果可导出，请先执行评分"

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"score_results_{timestamp}.csv"

    cls_labels = {"high": "高质量", "mid": "中等", "low": "低质量"}
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "score", "classification", "reason"])
        for item in structured_results:
            filename = Path(item["path"]).name
            score = item["score"] if item["score"] is not None else ""
            cls = cls_labels.get(item["classification"], item["classification"])
            reason = item["reason"] or (item["error"] or "")
            writer.writerow([filename, score, cls, reason])

    return gr.update(value=str(csv_path), visible=True), f"✅ 已导出至 {csv_path}"


# ============================================================
# UI rendering
# ============================================================

def render_tab_image_tools(s, provider_info):
    """Render the image tools tab (rename + score + review). Returns component dict."""
    with gr.Tabs():
        # ---- Sub-tab: AI Classify ----
        with gr.Tab("🤖 AI 智能分类"):
            gr.Markdown(_make_title("AI 智能分类 — 按内容自动归类图片"))
            gr.Markdown("用 AI 识别每张图片的内容，自动归入对应分类文件夹。适合整理大量混杂图片。")
            with gr.Accordion("📖 使用方法", open=False):
                gr.Markdown("**步骤：** ① 输入图片文件夹 → ② 勾选需要的分类 → ③ 点「开始分类」\n\n"
                            "> 💡 根据你的图片内容勾选分类，没有的类别取消勾选可以提高准确率\n"
                            "> 💡 可在「自定义补充」框添加额外分类（逗号分隔）\n"
                            "> 💡 建议先勾选「试运行」预览效果\n"
                            "> 📎 分类完成后，可切换到「图片重命名」对每个分类文件夹分别重命名")
            with gr.Row():
                with gr.Column(scale=2):
                    ai_cls_input = gr.Textbox(label="图片文件夹",
                                              value=s["ai_classify"].get("input_dir", str(config.DATA_DIR / "images")),
                                              placeholder="粘贴图片所在文件夹的完整路径")

                    # Predefined category checkboxes
                    _ALL_CATS = {
                        "照片": "真实拍摄的照片，如风景、人像、生活照",
                        "动漫": "日本动漫、漫画风格的图片和同人插画",
                        "游戏": "电子游戏相关的图片，包括游戏角色、截图、同人图",
                        "绘画": "手绘、数字绘画、插画、原画等艺术作品",
                        "聊天截图": "微信、QQ、Discord 等聊天软件的截图",
                        "应用截图": "软件界面、系统设置、手机桌面等屏幕截图",
                        "风景": "自然风光、城市景观、旅行摄影",
                        "美食": "食物、饮品、餐厅相关图片",
                        "文档": "扫描件、证件、课件、合同等文档照片",
                        "其他": "不属于以上任何分类（建议保留作为兜底）",
                    }
                    _default_checked = ["照片", "动漫", "游戏", "聊天截图", "应用截图", "其他"]
                    # Load saved state if available
                    _saved_checked = s["ai_classify"].get("checked_cats", None)
                    if _saved_checked and isinstance(_saved_checked, list):
                        # Filter out invalid choices (e.g. old "截图" split into "聊天截图"/"应用截图")
                        _valid_names = set(_ALL_CATS.keys())
                        _default_checked = [c for c in _saved_checked if c in _valid_names]
                        # If old "截图" was checked, replace with both new categories
                        if "截图" in _saved_checked:
                            for new_cat in ("聊天截图", "应用截图"):
                                if new_cat not in _default_checked:
                                    _default_checked.append(new_cat)

                    gr.Markdown("**选择分类（勾选你需要的类别）**")
                    ai_cls_checks = gr.CheckboxGroup(
                        choices=[(f"{name} — {desc}", name) for name, desc in _ALL_CATS.items()],
                        value=_default_checked,
                        label=None,
                        container=False,
                    )
                    with gr.Row():
                        ai_cls_custom = gr.Textbox(
                            label="自定义补充",
                            value=s["ai_classify"].get("custom_cats", ""),
                            placeholder="逗号分隔，如：建筑,动物,二次元",
                            scale=3,
                        )
                        ai_cls_custom_desc = gr.Textbox(
                            label="自定义描述（可选）",
                            value=s["ai_classify"].get("custom_desc", ""),
                            placeholder="格式：名称：描述",
                            scale=2,
                        )

                    with gr.Accordion("⚙️ 高级设置", open=False):
                        ai_cls_model_type, ai_cls_model, ai_cls_fetch_btn, ai_cls_fetch_st, ai_cls_thinking = _make_model_selector(
                            "视觉模型", s["ai_classify"].get("model", config.VISION_MODEL),
                            "需要视觉模型，能理解图片内容",
                            thinking_default=s["ai_classify"].get("thinking", False))
                        ai_cls_workers = gr.Slider(1, 8, value=s["ai_classify"].get("workers", config.DEFAULT_WORKERS), step=1,
                                                   label="并行线程数", info="越大越快，但可能触发 API 限流")
                        ai_cls_maxsz = gr.Slider(512, 4096, value=s["ai_classify"].get("max_size", config.IMAGE_MAX_SIZE), step=256,
                                                 label="图片最大边长（px）",
                                                 info="超过此值的图片会等比缩小")
                        ai_cls_threshold = gr.Slider(0, 64, value=s["ai_classify"].get("similarity_threshold", 20), step=1,
                                                     label="相似度阈值",
                                                     info="越小越严格（只有几乎相同才同组），越大越宽松。游戏截图建议 30-48")
                    with gr.Row():
                        ai_cls_dry = gr.Checkbox(label="试运行", value=s["ai_classify"].get("dry_run", True))
                        ai_cls_group = gr.Checkbox(label="⚡ 智能分组加速", value=s["ai_classify"].get("group_similar", False),
                                                   info="相似图片自动归组，每组分类多张后投票，大幅减少 API 调用")
                        ai_cls_samples = gr.Slider(1, 8, value=s["ai_classify"].get("max_samples_per_group", 4), step=1,
                                                    label="每组采样数", info="每组送几张给 AI 分类，多数投票决定结果")
                    with gr.Row():
                        ai_cls_btn = gr.Button("开始分类", variant="primary")
                        ai_cls_stop = gr.Button("⏹ 停止", variant="stop")
                        ai_cls_apply = gr.Button("✅ 应用试运行结果", variant="secondary")
                with gr.Column(scale=3):
                    ai_cls_output = gr.Textbox(label="分类结果", lines=15, elem_classes="output-text",
                                               placeholder="分类完成后这里会显示每张图片的归类结果...")

            ai_cls_evt = ai_cls_btn.click(
                _ai_classify_images,
                [ai_cls_input, ai_cls_checks, ai_cls_custom, ai_cls_custom_desc, ai_cls_model, ai_cls_maxsz, ai_cls_dry, provider_info, ai_cls_thinking, ai_cls_workers, ai_cls_group, ai_cls_threshold, ai_cls_samples],
                [ai_cls_output])

            def _stop_ai_cls():
                from services.image_services import ai_classify_stop
                ai_classify_stop()
                return "⏹️ 正在停止..."

            ai_cls_stop.click(_stop_ai_cls, outputs=[ai_cls_output], cancels=[ai_cls_evt])

            def _apply_cls_results(input_dir):
                return _svc_apply_classify(input_dir)

            ai_cls_apply.click(_apply_cls_results, inputs=[ai_cls_input], outputs=[ai_cls_output])
            _bind_model_fetch(ai_cls_fetch_btn, ai_cls_model_type, ai_cls_model, ai_cls_fetch_st,
                              provider_info, config.VISION_MODEL)

        # ---- Sub-tab: Image Rename ----
        with gr.Tab("🏷️ 图片重命名"):
            gr.Markdown(_make_title("图片 AI 重命名 — 用中文短句替代杂乱的文件名"))
            with gr.Accordion("📖 使用方法", open=False):
                gr.Markdown("**步骤：** ① 把图片放进文件夹 → ② 点「开始重命名」→ ③ 文件名变成中文描述短语\n\n"
                            "> 💡 建议先勾选「试运行」预览效果，满意后再取消勾选正式改名。")
            with gr.Row():
                with gr.Column(scale=2):
                    rn_input = gr.Textbox(label="图片文件夹",
                                          value=s["rename"].get("input_dir", str(config.DATA_DIR / "images")),
                                          placeholder="粘贴图片所在文件夹的完整路径",
                                          info="支持 .jpg / .jpeg / .png / .webp / .avif / .gif")
                    rn_mode = gr.Dropdown(
                        label="命名模式",
                        choices=[
                            ("🖼️ 通用描述", "general"),
                            ("👤 人像聚焦", "portrait"),
                            ("🌄 风景聚焦", "landscape"),
                            ("📱 截图识别", "screenshot"),
                            ("🍜 美食聚焦", "food"),
                            ("🎨 动漫二次元", "anime"),
                        ],
                        value=s["rename"].get("rename_mode", "general"),
                        info="选择图片类型以获得更准确的描述。自定义提示词会覆盖此设置",
                    )
                    with gr.Accordion("⚙️ 高级设置", open=False):
                        rn_model_type, rn_model, rn_fetch_btn, rn_fetch_st, rn_thinking = _make_model_selector(
                            "视觉模型", s["rename"].get("model", config.RENAME_MODEL),
                            "需要视觉模型，能理解图片内容",
                            thinking_default=s["rename"].get("thinking", True))
                        rn_workers = gr.Slider(1, 8, value=s["rename"].get("workers", config.DEFAULT_WORKERS), step=1,
                                               label="并行线程数", info="越大越快，但可能触发 API 限流")
                    rn_custom_prompt = gr.Textbox(
                        label="自定义提示词（留空使用上方模式的默认提示词）",
                        value=s["rename"].get("custom_prompt", ""),
                        lines=4,
                        placeholder="留空则使用上方所选模式的默认提示词。\n可自行定义描述风格，如：用英文描述、侧重构图、只写一句话等。",
                        info="自定义 AI 描述图片的方式，留空 = 使用模式默认提示词"
                    )
                    rn_ctx = gr.Slider(0, 10, value=s["rename"].get("context_count", 3), step=1,
                                       label="上下文数量",
                                       info="参考前几张图的描述作为上下文。0=不参考，适合高级模型独立判断")
                    rn_maxsz = gr.Slider(512, 4096, value=config.IMAGE_MAX_SIZE, step=256,
                                         label="图片最大边长（px）",
                                         info="超过此值的图片会等比缩小，越小速度越快但精度可能下降")
                    rn_dry = gr.Checkbox(label="试运行（只预览不实际改名）", value=s["rename"].get("dry_run", False),
                                         info="强烈建议第一次使用时先试运行，看看效果")
                    rn_keep = gr.Checkbox(label="保留原文件名", value=s["rename"].get("keep_original", False),
                                          info="在描述后面附加原始文件名，适合文件名含时间戳等有用信息的情况")
                    with gr.Row():
                        rn_btn = gr.Button("开始重命名", variant="primary")
                        rn_stop = gr.Button("停止", variant="stop")
                        rn_apply = gr.Button("✅ 应用试运行结果", variant="secondary")
                with gr.Column(scale=3):
                    rn_output = gr.Textbox(label="处理结果", lines=15, elem_classes="output-text",
                                           placeholder="处理完成后这里会显示每张图片的新名字...")
            rn_btn.click(_rename_images, [rn_input, rn_model, rn_workers, rn_dry, rn_keep, rn_mode, rn_custom_prompt, rn_ctx, rn_maxsz, provider_info, rn_thinking], [rn_output])

            def _stop_rename():
                from image_tools.rename_images import request_stop
                request_stop()
                return "⏹️ 已请求停止..."

            rn_stop.click(_stop_rename, outputs=[rn_output])

            def _apply_rn_results(input_dir, keep_original):
                return _svc_apply_rename(input_dir, keep_original)

            rn_apply.click(_apply_rn_results, inputs=[rn_input, rn_keep], outputs=[rn_output])
            _bind_model_fetch(rn_fetch_btn, rn_model_type, rn_model, rn_fetch_st,
                              provider_info, config.RENAME_MODEL)

            # ---- Classify by work ----
            gr.Markdown("---")
            gr.Markdown("### 📁 按《》作品名自动分类")
            gr.Markdown("将已重命名的图片按文件名中的《作品名》自动归入子文件夹。")
            with gr.Row():
                rn_cls_dry = gr.Checkbox(label="试运行", value=False,
                                         info="先预览分类结果，不实际移动")
                rn_cls_min = gr.Slider(1, 20, value=3, step=1,
                                       label="最少图片数",
                                       info="该作品图片少于此数量则不移动")
                rn_cls_btn = gr.Button("开始分类", variant="secondary")
            rn_cls_output = gr.Textbox(label="分类结果", lines=8, elem_classes="output-text",
                                       placeholder="点击后显示分类结果...")
            rn_cls_btn.click(_classify_by_work, [rn_input, rn_cls_dry, rn_cls_min], [rn_cls_output])

        # ---- Sub-tab: Score & Classify ----
        with gr.Tab("🔍 评分与分类"):
            gr.Markdown(_make_title("AI 评分 + 自动分拣 — 多维度评估，高分/低分图片自动归类"))
            with gr.Accordion("📖 使用方法", open=False):
                gr.Markdown("**评分：** ① 选择检测模式 → ② 填文件夹路径 → ③ 点「开始评分」→ 每张图生成 `.txt` 评分文件\n\n"
                            "**分类：** ④ 评分完成后点「开始分类」→ 优质图片移至 `HighQuality/`，劣质/错误图片移至 `LowQuality_Errors/`\n\n"
                            "**七种检测模式：**\n"
                            "- 🎨 **AI 图片错误检测** — 检测 AI 生成图的肢体错乱、面部畸形、结构崩坏等问题\n"
                            "- 📸 **漫展摄影筛选** — 检测跑焦模糊、过曝欠曝等拍摄问题，筛选可出片的 Cosplay 照片\n"
                            "- 🖼️ **通用照片质量** — 综合评估清晰度、曝光、构图、内容趣味性\n"
                            "- 👤 **人像摄影评估** — 侧重面部清晰度、肤色、表情、虚化氛围\n"
                            "- 🌄 **风景摄影评估** — 侧重光影层次、构图法则、色彩氛围\n"
                            "- 📄 **文档扫描清晰度** — 评估文字可读性、光照均匀度、畸变、完整度\n"
                            "- 🖌️ **绘画插图质量** — 评估造型比例、线条笔触、色彩光影、完成度\n\n"
                            "> 💡 评分和分类是独立的两步，评完分可以先看分数再决定如何分类")

            # ---- Score section ----
            gr.Markdown("### ① 评分")
            with gr.Row():
                with gr.Column(scale=2):
                    de_mode = gr.Dropdown(
                        label="检测模式",
                        choices=[
                            ("🎨 AI 图片错误检测", "ai"),
                            ("📸 漫展摄影筛选", "photo"),
                            ("🖼️ 通用照片质量", "general"),
                            ("👤 人像摄影评估", "portrait"),
                            ("🌄 风景摄影评估", "landscape"),
                            ("📄 文档扫描清晰度", "document"),
                            ("🖌️ 绘画插图质量", "art"),
                        ],
                        value=s["score"].get("mode", "ai"),
                        info="AI错误：找肢体畸形/崩坏 | 摄影：找跑焦/过曝 | 通用：综合评估 | 人像/风景/文档/绘画各有侧重"
                    )
                    de_input = gr.Textbox(label="图片文件夹",
                                          value=s["score"].get("input_dir", str(config.DATA_DIR / "images")),
                                          placeholder="粘贴图片所在文件夹的完整路径",
                                          info="评分结果保存为同名 .txt 文件")
                    de_model_type, de_model, de_fetch_btn, de_fetch_st, de_thinking = _make_model_selector(
                        "视觉模型", s["score"].get("model", config.VISION_MODEL),
                        "需要视觉模型。留空使用设置页默认值",
                        thinking_default=s["score"].get("thinking", True))
                    de_maxsz = gr.Slider(512, 4096, value=s["score"].get("max_size", config.IMAGE_MAX_SIZE), step=256,
                                         label="图片最大边长（px）",
                                         info="超过此值的图片会等比缩小，越小速度越快但精度可能下降")
                    de_prompt = gr.Textbox(
                        label="自定义评分提示词（留空使用默认）",
                        value=s["score"].get("custom_prompt", ""),
                        lines=6,
                        placeholder="留空则使用当前模式的默认提示词。\n切换检测模式后请清空此框或重新填写对应模式的提示词。",
                        info="留空 = 使用默认提示词"
                    )
                    with gr.Row():
                        de_score_btn = gr.Button("开始评分", variant="primary")
                        de_score_stop = gr.Button("停止", variant="stop")
                with gr.Column(scale=3):
                    de_score_output = gr.Textbox(label="评分日志", lines=15, elem_classes="output-text",
                                                 placeholder="评分完成后这里会显示每张图的分数和理由...")

            # ---- Classify section ----
            gr.Markdown("### ② 分类（根据已有评分结果）")
            with gr.Row():
                with gr.Column(scale=2):
                    de_cls_method = gr.Radio(
                        label="分类方式",
                        choices=[("按比例（前/后 N%）", "percent"), ("按分值（≥N 分 或 <M 分）", "threshold")],
                        value=s["classify"].get("classify_method", "percent"),
                        info="按比例：分数前N%入高分、后N%入低分 | 按分值：设定分数线"
                    )
                    with gr.Column():
                        with gr.Row(visible=True) as de_percent_row:
                            de_top = gr.Slider(1, 50, value=s["classify"].get("top_percent", int(config.TOP_PERCENT * 100)), step=1,
                                               label="高分比例（%）",
                                               info="评分前 N% 的图片移入 HighQuality")
                            de_bottom = gr.Slider(1, 50, value=s["classify"].get("bottom_percent", int(config.BOTTOM_PERCENT * 100)), step=1,
                                                  label="低分比例（%）",
                                                  info="评分后 N% + 所有 ERR 图片移入 LowQuality_Errors")
                        with gr.Row(visible=False) as de_threshold_row:
                            de_min = gr.Slider(0.0, 10.0, value=s["classify"].get("min_score", 7.0), step=0.1,
                                               label="高分线（≥）",
                                               info="分数 ≥ 此值的图片移入 HighQuality")
                            de_max = gr.Slider(0.0, 10.0, value=s["classify"].get("max_score", 4.0), step=0.1,
                                               label="低分线（<）",
                                               info="分数 < 此值的图片 + ERR 移入 LowQuality_Errors")
                    with gr.Row():
                        de_cls_btn = gr.Button("开始分类", variant="primary")
                with gr.Column(scale=3):
                    de_cls_output = gr.Textbox(label="分类日志", lines=12, elem_classes="output-text",
                                               placeholder="分类完成后这里会显示移动结果...")

            def _toggle_cls_method(method):
                if method == "percent":
                    return gr.update(visible=True), gr.update(visible=False)
                else:
                    return gr.update(visible=False), gr.update(visible=True)

            de_cls_method.change(_toggle_cls_method, [de_cls_method], [de_percent_row, de_threshold_row])

            def _stop_scoring():
                from image_tools.detect_ai_errors import request_stop
                request_stop()
                return "⏹️ 已请求停止..."

            # State for structured score results (shared by gallery & CSV)
            score_results_state = gr.State([])

            de_score_stop.click(_stop_scoring, outputs=[de_score_output])
            de_cls_btn.click(_classify_images, [de_input, de_cls_method, de_top, de_bottom, de_min, de_max], [de_cls_output])
            _bind_model_fetch(de_fetch_btn, de_model_type, de_model, de_fetch_st,
                              provider_info, config.VISION_MODEL)

            # ---- Score visualization section ----
            gr.Markdown("### ③ 评分结果画廊")
            with gr.Row():
                with gr.Column(scale=1):
                    de_gallery_load_btn = gr.Button("📂 从文件夹加载评分结果", variant="secondary")
                    de_csv_btn = gr.Button("📥 导出 CSV", variant="secondary")
                    de_csv_file = gr.File(label="下载 CSV", visible=False)
                    de_csv_status = gr.Textbox(label="导出状态", interactive=False, max_lines=2)
                with gr.Column(scale=4):
                    de_gallery = gr.Gallery(
                        label="评分结果（按分数从高到低排列）",
                        columns=4,
                        height=420,
                        object_fit="contain",
                        show_label=True,
                        value=[],
                    )
                    de_gallery_info = gr.Textbox(label="画廊信息", interactive=False,
                                                  value="评分完成后自动加载，也可点击左侧按钮从文件夹加载")

            # Wire up score button → log + state → gallery
            de_score_btn.click(
                _score_images,
                [de_input, de_mode, de_prompt, de_model, de_maxsz, provider_info, de_thinking],
                [de_score_output, score_results_state],
            ).then(_populate_gallery, [score_results_state], [de_gallery, de_gallery_info])

            de_gallery_load_btn.click(
                _load_scored_to_gallery, [de_input],
                [de_gallery, de_gallery_info, score_results_state],
            )
            de_csv_btn.click(_export_csv, [score_results_state], [de_csv_file, de_csv_status])

            # ---- Manual review section ----
            gr.Markdown("### ④ 手动审核（逐张查看评分并手动分类）")
            gr.Markdown("加载评分结果后，逐张查看缩略图和点评，手动决定每张图片的分类。")

            review_state = gr.State({"queue": [], "index": 0, "last_moved": None})

            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        review_input = gr.Textbox(
                            label="图片文件夹（已完成评分的）",
                            value=s["score"].get("input_dir", str(config.DATA_DIR / "images")),
                            placeholder="与上方评分使用同一目录",
                            scale=3
                        )
                        review_load_btn = gr.Button("📂 加载评分结果", variant="secondary", scale=1)

                    review_image = gr.Image(label="当前图片", height=420, show_label=True,
                                            elem_classes="output-text")
                    with gr.Row():
                        review_progress = gr.Textbox(label="进度", value="未加载", interactive=False, scale=1)

                with gr.Column(scale=1):
                    review_info = gr.Markdown("等待加载评分结果...", elem_classes="output-text")
                    with gr.Row():
                        review_high_btn = gr.Button("✅ 高质量", variant="primary", size="lg")
                        review_low_btn = gr.Button("❌ 低质量", variant="stop", size="lg")
                    with gr.Row():
                        review_skip_btn = gr.Button("⏭️ 跳过", variant="secondary")
                        review_undo_btn = gr.Button("↩️ 撤销上一步", variant="secondary")
                    review_log = gr.Textbox(label="操作日志", lines=4, interactive=False, elem_classes="output-text")

            def _load_image_safe(path, max_side=1024):
                from PIL import Image, ImageOps
                img = Image.open(path)
                try:
                    img = ImageOps.exif_transpose(img)
                except Exception:
                    pass
                w, h = img.size
                if max(w, h) > max_side:
                    ratio = max_side / max(w, h)
                    img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
                import numpy as np
                return np.array(img)

            def _on_review_load(input_dir):
                from image_tools.detect_ai_errors import load_review_queue
                queue = load_review_queue(input_dir)
                if not queue:
                    return {}, None, "❌ 未找到评分文件（.txt），请先执行评分。", "0/0", "未找到评分文件"
                item = queue[0]
                info = _format_review_item(item)
                progress = f"第 1/{len(queue)} 张"
                log = f"✅ 已加载 {len(queue)} 张已评分图片"
                return {"queue": queue, "index": 0, "last_moved": None}, _load_image_safe(item["path"]), info, progress, log

            def _format_review_item(item):
                if item["error"]:
                    return f"### ❌ 错误图片\n\n**错误**: {item['error']}\n\n---"
                score = item["score"]
                emoji = "🟢" if score >= 7 else ("🟡" if score >= 5 else ("🟠" if score >= 3 else "🔴"))
                return f"### {emoji} 评分: {score} 分\n\n**点评**: {item['reason']}\n\n---"

            def _review_action(action, state, input_dir):
                if not state or not state["queue"]:
                    return state, None, "无图片可处理", "0/0", "请先加载评分结果"
                queue = state["queue"]
                idx = state["index"]
                if idx >= len(queue):
                    return state, None, "✅ 审核完成！所有图片已处理。", f"{len(queue)}/{len(queue)}", "审核完成"
                item = queue[idx]
                img_path = str(item["path"])
                log = ""

                if action == "undo":
                    last = state.get("last_moved")
                    if last:
                        from image_tools.detect_ai_errors import move_single_to_category
                        result = move_single_to_category(last["path"], "undo", input_dir)
                        log = result["msg"]
                        from image_tools.detect_ai_errors import load_review_queue
                        queue = load_review_queue(input_dir)
                        idx = max(0, idx - 1)
                    else:
                        log = "没有可撤销的操作"
                elif action == "skip":
                    log = f"跳过: {Path(img_path).name}"
                    idx += 1
                else:
                    from image_tools.detect_ai_errors import move_single_to_category
                    result = move_single_to_category(img_path, action, input_dir)
                    log = result["msg"]
                    if result["ok"]:
                        state["last_moved"] = {"path": img_path, "category": action}
                        idx += 1

                if idx >= len(queue):
                    return state, None, "✅ 审核完成！", f"{len(queue)}/{len(queue)}", log

                next_item = queue[idx]
                info = _format_review_item(next_item)
                progress = f"第 {idx+1}/{len(queue)} 张"
                return {"queue": queue, "index": idx, "last_moved": state.get("last_moved")}, _load_image_safe(next_item["path"]), info, progress, log

            review_load_btn.click(_on_review_load, [review_input],
                                  [review_state, review_image, review_info, review_progress, review_log])

            review_high_btn.click(lambda s, d: _review_action("high", s, d),
                                  [review_state, review_input],
                                  [review_state, review_image, review_info, review_progress, review_log])
            review_low_btn.click(lambda s, d: _review_action("low", s, d),
                                 [review_state, review_input],
                                 [review_state, review_image, review_info, review_progress, review_log])
            review_skip_btn.click(lambda s, d: _review_action("skip", s, d),
                                  [review_state, review_input],
                                  [review_state, review_image, review_info, review_progress, review_log])
            review_undo_btn.click(lambda s, d: _review_action("undo", s, d),
                                  [review_state, review_input],
                                  [review_state, review_image, review_info, review_progress, review_log])

    return {"rn_model": rn_model, "rn_model_type": rn_model_type, "de_model": de_model, "de_model_type": de_model_type}
