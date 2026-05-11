"""Chat tab: multi-conversation chat with RAG and web search."""
import os
import time as _time
import logging
import re
import gradio as gr
import config
import history
from ui.common import _apply_provider, _bind_model_fetch
from utils.web_search import _do_web_search
from utils.kb_chat_helpers import (
    _kb_save_chat, _kb_load_chat, _kb_list_chats,
    _kb_delete_chat, _kb_auto_title,
)

logger = logging.getLogger("LocalAITools")


def render_tab_chat(s, provider_info, kb_chat_tab):
    """Render the chat tab. kb_chat_tab is the gr.Tab context. Returns component dict."""
    with gr.Row():
        # ---- Left sidebar: conversation list + settings ----
        with gr.Column(scale=1, min_width=220, elem_classes="kb-sidebar"):
            gr.Markdown("### 💬 对话")
            kb_new_chat_btn = gr.Button("＋ 新建对话", size="sm")
            with gr.Row():
                kb_search_input = gr.Textbox(
                    label="", placeholder="搜索对话内容...",
                    show_label=False, container=False, scale=3,
                    max_lines=1, elem_id="kb_search_input",
                )
                kb_search_btn = gr.Button("🔍 搜索", size="sm", scale=1, min_width=60)
            kb_chat_list = gr.Radio(label="", choices=[], interactive=True,
                                     container=False, elem_id="kb_chat_list")
            with gr.Row():
                kb_del_btn = gr.Button("🗑 删除", size="sm")
                kb_refresh_btn = gr.Button("🔄 刷新", size="sm")
            kb_chat_status = gr.Markdown("")

            # Feature toggles
            kb_use_rag = gr.Checkbox(label="📚 知识库检索", value=True, container=False)
            kb_rag_k = gr.Slider(1, 30, value=5, step=1, label="检索片段数",
                                  info="每次 RAG 检索返回的文本块数量，越大上下文越多但可能超限")
            kb_use_web = gr.Checkbox(label="🌐 联网搜索", value=False, container=False)
            kb_thinking = gr.Checkbox(label="🧠 深度思考", value=True, container=False)

            with gr.Accordion("⚙️ 模型设置", open=False):
                kb_model_type = gr.Dropdown(
                    choices=["全部模型", "chat 模型", "vlm 视觉模型", "embed 模型", "其他"],
                    value="全部模型", label="模型分类", container=False,
                )
                with gr.Row():
                    kb_model = gr.Dropdown(
                        choices=[config.TEXT_MODEL], value=config.TEXT_MODEL,
                        label="模型", filterable=True, allow_custom_value=True,
                        scale=3, container=False,
                    )
                    kb_fetch_btn = gr.Button("🔄", size="sm", scale=0, min_width=40)
                kb_fetch_st = gr.Textbox(visible=False, max_lines=1)

        # ---- Right: chat main area ----
        with gr.Column(scale=4, elem_classes="kb-chat-fill"):
            kb_chatbot = gr.Chatbot(label="", placeholder="输入问题开始对话...",
                                     elem_id="kb_chatbot", show_label=False, height=500)

            with gr.Column(elem_classes="kb-input-row"):
                kb_query = gr.Textbox(label="", placeholder="输入你的问题...",
                                      lines=2, show_label=False, container=False,
                                      elem_id="kb_query_box")
                with gr.Row():
                    kb_btn = gr.Button("发送", variant="primary", scale=0, min_width=60)
                    kb_stop_btn = gr.Button("⏹ 停止", variant="stop", size="sm", scale=0, min_width=60)
                    kb_clear_btn = gr.Button("清空", size="sm", scale=0, min_width=50)
                    kb_export_btn = gr.Button("导出对话", size="sm", scale=0, min_width=60)

            gr.Markdown("*AI 生成的内容可能不准确，请核实重要信息。*", elem_classes="kb-disclaimer")

    # ---- State ----
    kb_all_chats = gr.State({})
    kb_active_name = gr.State("")

    # ---- Switch conversation ----
    def _switch_chat(selection, all_chats):
        if not selection:
            return all_chats, "", [], ""
        all_chats = dict(all_chats) if all_chats else {}
        msgs = all_chats.get(selection, [])
        msgs, _ = _kb_load_chat(selection) if not msgs else (msgs, "")
        all_chats[selection] = msgs
        display = _format_display(msgs)
        return all_chats, selection, display, f"📌 {selection}"

    def _format_display(msgs):
        """Convert flat history msgs to Chatbot display format with thought blocks."""
        display = []
        for m in msgs:
            content = m.get("content", "")
            if content.startswith("[思考过程]\n"):
                parts = content.split("\n\n", 1)
                reasoning = parts[0].replace("[思考过程]\n", "", 1)
                answer = parts[1] if len(parts) > 1 else ""
                display.append({"role": "assistant", "content": reasoning,
                                "metadata": {"title": "🤔 思考过程"}})
                if answer:
                    display.append({"role": "assistant", "content": answer})
            else:
                display.append({"role": m["role"], "content": content})
        return display

    # ---- Send query (streaming) ----
    def _send_query_stream(query, model, all_chats, active_name, provider, thinking, use_rag, use_web, rag_k=5):
        if not query or not query.strip():
            yield all_chats, active_name, all_chats.get(active_name, []), "", gr.update(), "❌ 请输入问题"
            return
        _apply_provider(provider)
        os.environ["ENABLE_THINKING"] = "true" if thinking else "false"
        all_chats = dict(all_chats) if all_chats else {}
        msgs = list(all_chats.get(active_name, []))

        model_name = model or config.TEXT_MODEL
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

        new_msgs = list(msgs)
        new_msgs.append({"role": "user", "content": query.strip()})
        radio_update = gr.update()

        # ===== AI autonomous search loop =====
        search_parts = []
        if use_web:
            for _rnd in range(3):
                _hist = "\n".join(
                    f"{'用户' if m['role']=='user' else '助手'}: {m['content'][:300]}"
                    for m in msgs[-6:]
                )
                _ctx = "\n\n".join(search_parts) if search_parts else "（暂无）"
                d_msgs = [
                    SystemMessage(content=(
                        "判断是否需要搜索网络。\n"
                        "- 需要更多信息：只回复 [SEARCH:搜索词]\n"
                        "- 信息已足够：只回复 [ANSWER]\n"
                        "搜索词要精简准确，用中文。"
                    )),
                    HumanMessage(content=(
                        f"对话历史：\n{_hist}\n\n"
                        f"用户最新问题：{query.strip()}\n\n"
                        f"已有搜索结果：\n{_ctx}\n\n"
                        f"{'还需要搜索什么？' if search_parts else '需要搜索什么信息？'}"
                    )),
                ]
                try:
                    d_llm = ChatOpenAI(
                        base_url=config.OPENAI_BASE_URL, api_key=config.OPENAI_API_KEY,
                        model=model_name, temperature=0.2, max_tokens=100,
                        extra_body=config.get_llm_extra_body(False), request_timeout=30,
                    )
                    decision = d_llm.invoke(d_msgs).content.strip()
                except Exception:
                    break

                if decision.startswith("[SEARCH:") and "]" in decision:
                    sq = decision.split("[SEARCH:")[1].split("]")[0].strip()
                    if sq and len(sq) >= 2:
                        new_msgs.append({"role": "assistant", "content": f"🔍 搜索：**{sq}**"})
                        all_chats[active_name] = new_msgs
                        yield all_chats, active_name, _format_display(new_msgs), "", radio_update, ""

                        results = _do_web_search(sq, max_results=5)
                        if results:
                            search_parts.append(f"【搜索「{sq}」】\n{results}")
                            new_msgs[-1]["content"] = f"🔍 搜索：**{sq}**  ✅ 找到结果"
                        else:
                            new_msgs[-1]["content"] = f"🔍 搜索：**{sq}**  ❌ 无结果"
                        all_chats[active_name] = new_msgs
                        yield all_chats, active_name, _format_display(new_msgs), "", radio_update, ""
                        continue
                break  # [ANSWER] → proceed to answer

        web_results = "\n\n".join(search_parts) if search_parts else None

        # Build LLM input
        if use_rag:
            from text_tools.kb_chat import prepare_kb_context
            prompt, lc_msgs_fb = prepare_kb_context(
                query=query.strip(), keyword="", k=int(rag_k),
                batch_size=5, chat_history=msgs, web_context=web_results,
            )
            if prompt is not None:
                llm_input = prompt
            else:
                llm_input = lc_msgs_fb
        elif web_results:
            lc_msgs = [SystemMessage(content=f"以下是联网搜索的最新结果，请基于这些信息回答用户问题：\n\n{web_results}")]
            for m in msgs:
                if m["role"] == "user": lc_msgs.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant": lc_msgs.append(AIMessage(content=m["content"]))
            lc_msgs.append(HumanMessage(content=query.strip()))
            llm_input = lc_msgs
        else:
            lc_msgs = []
            for m in msgs:
                if m["role"] == "user": lc_msgs.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant": lc_msgs.append(AIMessage(content=m["content"]))
            lc_msgs.append(HumanMessage(content=query.strip()))
            llm_input = lc_msgs

        llm = ChatOpenAI(
            base_url=config.OPENAI_BASE_URL,
            api_key=config.OPENAI_API_KEY,
            model=model_name,
            temperature=0.7,
            streaming=True,
            extra_body=config.get_llm_extra_body(thinking),
        )

        def _get_reasoning(chunk) -> str:
            ak = getattr(chunk, 'additional_kwargs', {}) or {}
            for key in ('reasoning_content', 'thinking', 'reasoning'):
                if ak.get(key):
                    return ak[key]
            rm = getattr(chunk, 'response_metadata', {}) or {}
            for key in ('reasoning_content', 'thinking', 'reasoning'):
                if rm.get(key):
                    return rm[key]
            return ""

        new_msgs.append({"role": "assistant", "content": ""})
        all_chats[active_name] = new_msgs
        new_active = active_name

        reasoning = ""
        answer = ""

        # Pre-build the history part (doesn't change during streaming)
        _hist_display = [{"role": m["role"], "content": m["content"]} for m in new_msgs[:-1]]

        def _build_display(reasoning, answer):
            """Build display list with native thought blocks for Chatbot."""
            display = list(_hist_display)
            if reasoning:
                display.append({"role": "assistant", "content": reasoning,
                                "metadata": {"title": "🤔 思考过程"}})
            if answer:
                display.append({"role": "assistant", "content": answer})
            elif not reasoning:
                display.append({"role": "assistant", "content": ""})
            return display

        try:
            for chunk in llm.stream(llm_input):
                r = _get_reasoning(chunk)
                c = chunk.content if hasattr(chunk, 'content') and chunk.content else ""
                if r:
                    reasoning += r
                if c:
                    answer += c
                # Store flat content in history (no HTML tags)
                new_msgs[-1]["content"] = (f"[思考过程]\n{reasoning}\n\n{answer}" if reasoning else answer)
                display = _build_display(reasoning, answer)
                yield all_chats, new_active, display, "", radio_update, ""
        except Exception as e:
            new_msgs[-1]["content"] = f"❌ 调用失败: {e}"
            yield all_chats, new_active, new_msgs, "", radio_update, ""
            return

        _kb_save_chat(new_msgs, new_active)
        history.add_entry("知识库问答", query[:50], "查询完成")

        yield all_chats, new_active, _format_display(new_msgs), "", radio_update, ""

        # Auto-name in background thread
        if active_name.startswith("新对话_") and len(new_msgs) >= 2:
            import threading as _th

            def _auto_name_bg(_provider=provider, _active=active_name, _msgs=new_msgs, _chats=all_chats):
                try:
                    _apply_provider(_provider)
                    title = _kb_auto_title(_provider, _msgs)
                    if title and title != _active:
                        _kb_delete_chat(_active)
                        _chats.pop(_active, None)
                        _chats[title] = _msgs
                        _kb_save_chat(_msgs, title)
                except Exception:
                    pass

            _th.Thread(target=_auto_name_bg, daemon=True).start()

    # ---- Clear current conversation ----
    def _clear_chat_multi(all_chats, active_name):
        if not active_name:
            return all_chats, active_name, [], "", "❌ 没有激活的对话"
        all_chats = dict(all_chats) if all_chats else {}
        all_chats[active_name] = []
        return all_chats, active_name, [], "", "✅ 已清空"

    # ---- Export conversation to Markdown ----
    def _export_chat(all_chats, active_name):
        if not active_name:
            return "❌ 没有激活的对话"
        msgs = all_chats.get(active_name, [])
        if not msgs:
            return "❌ 当前对话为空，没有内容可导出"
        lines = [f"# {active_name}\n"]
        for m in msgs:
            role = "用户" if m.get("role") == "user" else "助手"
            content = m.get("content", "").strip()
            if content:
                lines.append(f"**{role}：**\n\n{content}\n")
        md_text = "\n---\n\n".join(lines)
        export_dir = config.OUTPUT_DIR / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r'[\\/*?:"<>|]', '', active_name).strip() or "chat"
        filename = f"{safe_name}_{_time.strftime('%Y%m%d_%H%M%S')}.md"
        path = export_dir / filename
        path.write_text(md_text, encoding="utf-8")
        return f"✅ 已导出到: {path}"

    # ---- New conversation ----
    def _new_chat_multi(all_chats, active_name):
        all_chats = dict(all_chats) if all_chats else {}
        new_name = f"新对话_{_time.strftime('%H%M%S')}"
        all_chats[new_name] = []
        choices = sorted(all_chats.keys())
        return all_chats, new_name, [], gr.update(choices=choices, value=new_name), f"✅ {new_name}"

    # ---- Refresh conversation list ----
    def _refresh_chat_list_multi(all_chats, active_name):
        all_chats = dict(all_chats) if all_chats else {}
        saved_choices, status = _kb_list_chats()
        merged = set(saved_choices) | set(all_chats.keys())
        choices = sorted(merged)
        return gr.update(choices=choices), status

    # ---- Delete conversation ----
    def _del_chat_multi(selection, all_chats, active_name):
        if not selection:
            return all_chats, active_name, [], gr.update(), "❌ 请选择要删除的对话"
        all_chats = dict(all_chats) if all_chats else {}
        all_chats.pop(selection, None)
        _kb_delete_chat(selection)
        choices = sorted(all_chats.keys())
        new_active = active_name if active_name != selection else ""
        new_msgs = all_chats.get(new_active, []) if new_active else []
        status = f"✅ 已删除 {selection}"
        return (all_chats, new_active, _format_display(new_msgs),
                gr.update(choices=choices, value=new_active if new_active else None), status)

    # ---- Search conversations ----
    def _search_chats(keyword, all_chats):
        all_chats = dict(all_chats) if all_chats else {}
        if not keyword or not keyword.strip():
            choices = sorted(all_chats.keys())
            return gr.update(choices=choices), ""
        keyword = keyword.strip().lower()
        matched = []
        for name, msgs in all_chats.items():
            for m in (msgs or []):
                content = (m.get("content") or "").lower()
                if keyword in content:
                    matched.append(name)
                    break
        matched.sort()
        count = len(matched)
        if not matched:
            status = f"未找到包含 '{keyword.strip()}' 的对话"
        else:
            status = f"找到 {count} 个对话包含 '{keyword.strip()}'"
        return gr.update(choices=matched), status

    def _clear_search(all_chats):
        all_chats = dict(all_chats) if all_chats else {}
        choices = sorted(all_chats.keys())
        return "", gr.update(choices=choices), ""

    # ---- Event binding ----
    kb_send_evt = kb_btn.click(
        _send_query_stream,
        [kb_query, kb_model, kb_all_chats, kb_active_name, provider_info, kb_thinking, kb_use_rag, kb_use_web, kb_rag_k],
        [kb_all_chats, kb_active_name, kb_chatbot, kb_query, kb_chat_list, kb_chat_status],
        show_progress="hidden")
    kb_submit_evt = kb_query.submit(
        _send_query_stream,
        [kb_query, kb_model, kb_all_chats, kb_active_name, provider_info, kb_thinking, kb_use_rag, kb_use_web, kb_rag_k],
        [kb_all_chats, kb_active_name, kb_chatbot, kb_query, kb_chat_list, kb_chat_status],
        show_progress="hidden")

    kb_chat_list.change(_switch_chat, [kb_chat_list, kb_all_chats],
                       [kb_all_chats, kb_active_name, kb_chatbot, kb_chat_status],
                       cancels=[kb_send_evt, kb_submit_evt])
    kb_stop_btn.click(fn=None, cancels=[kb_send_evt, kb_submit_evt])

    kb_clear_btn.click(_clear_chat_multi, [kb_all_chats, kb_active_name],
                      [kb_all_chats, kb_active_name, kb_chatbot, kb_query, kb_chat_status])
    kb_new_chat_btn.click(_new_chat_multi, [kb_all_chats, kb_active_name],
                         [kb_all_chats, kb_active_name, kb_chatbot, kb_chat_list, kb_chat_status])
    kb_refresh_btn.click(_refresh_chat_list_multi, [kb_all_chats, kb_active_name],
                        [kb_chat_list, kb_chat_status])
    kb_del_btn.click(_del_chat_multi, [kb_chat_list, kb_all_chats, kb_active_name],
                    [kb_all_chats, kb_active_name, kb_chatbot, kb_chat_list, kb_chat_status])
    kb_export_btn.click(_export_chat, [kb_all_chats, kb_active_name],
                       [kb_chat_status])

    kb_search_btn.click(_search_chats, [kb_search_input, kb_all_chats],
                        [kb_chat_list, kb_chat_status])
    kb_search_input.submit(_search_chats, [kb_search_input, kb_all_chats],
                           [kb_chat_list, kb_chat_status])

    _kb_do_fetch = _bind_model_fetch(kb_fetch_btn, kb_model_type, kb_model, kb_fetch_st,
                                     provider_info, config.TEXT_MODEL)

    return {
        "kb_model": kb_model,
        "kb_model_type": kb_model_type,
        "kb_fetch_st": kb_fetch_st,
        "_kb_do_fetch": _kb_do_fetch,
        "_refresh_chat_list_multi": _refresh_chat_list_multi,
        "kb_all_chats": kb_all_chats,
        "kb_active_name": kb_active_name,
        "kb_chat_list": kb_chat_list,
        "kb_chat_status": kb_chat_status,
        "kb_search_input": kb_search_input,
        "kb_search_btn": kb_search_btn,
    }
