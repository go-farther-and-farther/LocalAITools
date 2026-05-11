"""Welcome / Getting Started tab."""
import gradio as gr
import history
from ui.common import _make_title


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


def render_tab_welcome(app):
    """Render the welcome/getting-started tab."""
    gr.Markdown(_make_title("欢迎使用 LocalAITools！"))
    gr.Markdown("一套调用本地大模型的 AI 工具集，**所有功能免费、数据不上传云端**。")

    gr.Markdown("---")

    gr.Markdown("### 🔌 第一步：获取 AI 模型服务")

    gr.Markdown("""本工具需要连接一个 AI 模型服务才能工作。推荐以下方式（任选一种）：

**⭐ 方式一：云端 API（最简单，免下载模型，按量付费）**
1. [硅基流动 SiliconFlow](https://cloud.siliconflow.cn/) — 注册送额度，支持 Qwen 系列
2. [DeepSeek 开放平台](https://platform.deepseek.com/) — 便宜好用
3. [阿里云百炼](https://bailian.console.aliyun.com/) — Qwen 官方 API
4. 注册后在后台创建 API Key，填到下方设置页的 API 地址和密钥即可

**⭐⭐ 方式二：LM Studio（推荐本地运行，免费）**
1. 下载安装 [LM Studio](https://lmstudio.ai/)（支持 Windows / Mac）
2. 打开 LM Studio，在搜索框搜 `qwen3` 或 `qwen3.6`
3. 下载一个视觉模型（如 `qwen3.6-27b`，约 16 GB）
4. 切换到 **Local Server** 标签页，点击 **Start Server**
5. 默认地址就是 `http://localhost:1234/v1`，无需修改

**⭐⭐⭐ 方式三：Ollama（免费，需命令行基础）**
```bash
ollama serve          # 启动服务
ollama pull qwen3     # 下载模型
```
默认地址 `http://localhost:11434/v1`

**⭐⭐⭐ 方式四：vLLM / SGLang / Xinference（自建推理服务，适合多人共享）**
- [vLLM](https://github.com/vllm-project/vllm) — 生产级推理引擎，支持 PagedAttention，吞吐量高
- [SGLang](https://github.com/sgl-project/sglang) — 高效推理框架，结构化生成能力强
- [Xinference](https://github.com/xorbitsai/inference) — 一键部署，支持 Web UI 管理模型
- 部署后 OpenAI 兼容端点填到设置页即可使用

> 💡 **简单总结：** 不想折腾 → 云端 API；有显卡想省钱 → LM Studio；技术党想折腾 → Ollama/vLLM
""")

    gr.Markdown("---")

    gr.Markdown("### ⚙️ 第二步：填写配置")
    gr.Markdown("""切换到 **⚙️ 设置** 标签页，填写你的 API 信息：

- **API 地址**：本地服务默认 `http://localhost:1234/v1`（LM Studio），云端 API 填对应地址
- **API 密钥**：本地服务填任意值（如 `lm-studio`），云端 API 填真实的 Key
- 填好后点 **💾 保存设置**，然后点 **🔗 测试连接** 确认能连上

> 💡 云端 API 需填写平台提供的地址和 Key；本地服务一般无需修改地址。""")

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

    # ---- History ----
    gr.Markdown("---")
    with gr.Accordion("📋 处理历史", open=False):
        history_md = gr.Markdown("")

    app.load(_load_history, outputs=[history_md])

    return {}
