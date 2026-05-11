"""Knowledge base management tab."""
import logging
from pathlib import Path
import config
import gradio as gr
from ui.common import _make_title
from utils.kb_chat_helpers import (
    _kb_create_kb, _kb_get_choices, _kb_list_docs, _kb_upload,
    _kb_delete, _kb_delete_all, _kb_build_index, _kb_stop_build, _kb_get_stats,
    _kb_import_url,
)

logger = logging.getLogger("LocalAITools")


def render_tab_kb(s):
    """Render the knowledge base management tab. Returns component dict."""
    gr.Markdown(_make_title("知识库管理 — 创建知识库、上传文档、构建索引"))

    _kb_bases, _kb_choices, _kb_labels = _kb_get_choices()
    _kb_default = _kb_choices[0] if _kb_choices else ""
    with gr.Row():
        kb_selector = gr.Dropdown(
            label="知识库名称",
            choices=_kb_choices,
            value=_kb_default,
            interactive=True,
            allow_custom_value=True,
            scale=4,
        )
        kb_selector_refresh = gr.Button("🔄", scale=0, min_width=50)
        kb_create_name = gr.Textbox(label="", placeholder="输入新知识库名称", scale=2,
                                    show_label=False, container=False)
        kb_create_btn = gr.Button("➕ 创建", variant="secondary", scale=0, min_width=70)
    kb_selector_info = gr.Markdown("")
    kb_selected_state = gr.State(_kb_default)

    def _on_kb_select(selected):
        if not selected:
            return "", selected
        from text_tools.kb_manager import list_documents
        docs = list_documents(Path(selected))
        label = _kb_labels.get(selected, Path(selected).name)
        info = f"**当前知识库:** {label}  |  路径: `{selected}`  |  文档数: {len(docs)}"
        return info, selected

    def _on_kb_refresh():
        bases, choices, labels = _kb_get_choices()
        default = choices[0] if choices else ""
        info, selected = _on_kb_select(default)
        return gr.update(choices=choices, value=default), info, selected

    kb_selector.change(_on_kb_select, [kb_selector], [kb_selector_info, kb_selected_state])
    kb_selector_refresh.click(_on_kb_refresh, outputs=[kb_selector, kb_selector_info, kb_selected_state])
    kb_create_btn.click(_kb_create_kb, [kb_create_name], [kb_selector_info, kb_selector])

    # Initial load
    _init_info = ""
    if _kb_default:
        from text_tools.kb_manager import list_documents as _list_docs_init
        _init_docs = _list_docs_init(Path(_kb_default))
        _init_label = _kb_labels.get(_kb_default, Path(_kb_default).name)
        _init_info = f"**当前知识库:** {_init_label}  |  路径: `{_kb_default}`  |  文档数: {len(_init_docs)}"
    kb_selector_info.value = _init_info

    with gr.Tabs():
        # ---- Sub-tab: Document management ----
        with gr.Tab("📄 文档管理"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 上传文档")
                    kb_upload_files = gr.File(label="选择文件（支持 .txt .md .csv .json .log .pdf .docx .html 等）",
                                              file_count="multiple",
                                              file_types=[".txt", ".md", ".csv", ".json", ".jsonl", ".log", ".py", ".rst",
                                                          ".pdf", ".docx", ".html", ".htm"])
                    kb_upload_btn = gr.Button("📤 上传到知识库", variant="primary")
                    kb_upload_result = gr.Textbox(label="上传结果", lines=4, interactive=False)

                    gr.Markdown("### 从 URL 导入")
                    kb_url_input = gr.Textbox(label="网页地址", placeholder="https://example.com/article")
                    kb_url_btn = gr.Button("🌐 抓取并导入", variant="primary")
                    kb_url_result = gr.Textbox(label="导入结果", lines=2, interactive=False)

                    gr.Markdown("### 删除文档")
                    kb_del_name = gr.Textbox(label="文件名", placeholder="输入要删除的文件名")
                    with gr.Row():
                        kb_del_btn = gr.Button("🗑️ 删除指定文件")
                        kb_del_all_btn = gr.Button("🗑️ 清空全部", variant="stop")
                    kb_del_result = gr.Textbox(label="删除结果", lines=2, interactive=False)

                with gr.Column(scale=3):
                    gr.Markdown("### 文档列表")
                    kb_doc_list = gr.Markdown("*点击「刷新」加载文档列表*")
                    kb_doc_info = gr.Textbox(label="统计", lines=1, interactive=False)
                    kb_doc_refresh_btn = gr.Button("🔄 刷新文档列表")

            kb_upload_btn.click(lambda f, kb: _kb_upload(f, kb), [kb_upload_files, kb_selected_state], [kb_upload_result])
            kb_url_btn.click(lambda url, kb: _kb_import_url(url, kb), [kb_url_input, kb_selected_state], [kb_url_result])
            kb_del_btn.click(lambda n, kb: _kb_delete(n, kb), [kb_del_name, kb_selected_state], [kb_del_result])
            kb_del_all_btn.click(lambda kb: _kb_delete_all(kb), [kb_selected_state], [kb_del_result])
            kb_doc_refresh_btn.click(lambda kb: _kb_list_docs(kb), [kb_selected_state], [kb_doc_list, kb_doc_info])

        # ---- Sub-tab: Index building ----
        with gr.Tab("🔧 索引构建"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 构建参数")
                    kb_chunk_size = gr.Slider(100, 2000, value=config.KB_CHUNK_SIZE, step=50, label="文本块大小（字符）",
                                              info="每个文本块的最大字符数")
                    kb_chunk_overlap = gr.Slider(0, 200, value=config.KB_CHUNK_OVERLAP, step=10, label="块间重叠（字符）",
                                                 info="相邻文本块重叠的字符数，有助于保持上下文连贯")
                    kb_embed_model = gr.Textbox(label="Embedding 模型（留空用默认）",
                                                placeholder="如 BAAI/bge-small-zh-v1.5 或本地路径",
                                                value="",
                                                info="HuggingFace 模型名或本地路径")
                    with gr.Row():
                        kb_build_btn = gr.Button("🔨 构建索引", variant="primary")
                        kb_build_stop = gr.Button("⏹️ 停止", variant="stop")
                    kb_build_result = gr.Textbox(label="构建日志", lines=10, interactive=False,
                                                 placeholder="点击「构建索引」开始...")

                with gr.Column(scale=2):
                    gr.Markdown("### 索引状态")
                    kb_stats_btn = gr.Button("🔄 刷新状态")
                    kb_stats_display = gr.Markdown("*点击「刷新状态」查看*")

            kb_build_btn.click(lambda cs, co, em, kb: _kb_build_index(cs, co, em, kb),
                               [kb_chunk_size, kb_chunk_overlap, kb_embed_model, kb_selected_state],
                               [kb_build_result])
            kb_build_stop.click(_kb_stop_build, outputs=[kb_build_result])
            kb_stats_btn.click(lambda kb: _kb_get_stats(kb), [kb_selected_state], [kb_stats_display])

    return {}
