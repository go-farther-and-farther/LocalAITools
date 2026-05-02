# LocalAITools — 开发者技术文档

> 目标读者：后续接手开发的工程师。本文档涵盖项目架构、模块设计、关键决策、已知坑位和版本演进。

---

## 1. 项目概述

LocalAITools 是一套**本地优先的 AI 工具集**，通过 Gradio Web UI 调用 OpenAI 兼容 API（LM Studio / Ollama / 云端 API），提供图片分析、文本处理、知识库问答、LLM 压测等功能。

**核心原则：**
- 所有数据默认不上传云端，API 地址由用户自行配置
- 一键安装启动，面向非技术用户也面向开发者
- 纯 Python + Gradio，无前端构建工具链
- 配置文件驱动（.env），Web UI 内可直接编辑配置
- 每个工具模块可独立命令行运行，也可通过 Web UI 调用

**技术栈：**
- Python 3.10+（类型注解、pathlib、f-string）
- Gradio 4.x（Web UI 框架，gr.Blocks 模式）
- LangChain（ChatOpenAI + HuggingFaceEmbeddings + FAISS）
- Pillow（图片 EXIF 处理 + 编码）
- matplotlib + numpy（压测图表）
- jieba + rank-bm25（中文分词 + BM25 检索）

---

## 2. 项目结构

```
LocalAITools/
├── app.py                  # Gradio Web UI 主入口（~900 行）
├── config.py               # 全局配置中心（读取 .env）
├── history.py              # 处理历史记录模块
├── requirements.txt        # Python 依赖
│
├── image_tools/
│   ├── detect_ai_errors.py     # 图片质量评分 + 自动分拣（Tab 2）
│   ├── ocr_chat_screenshots.py # 聊天截图 OCR 识别（Tab 3）
│   └── rename_images.py        # AI 图片重命名（Tab 1）
│
├── text_tools/
│   ├── translate.py            # 长篇文本翻译（Tab 5）
│   ├── compress_chat.py        # 聊天记录压缩（Tab 4）
│   └── chapter_summary.py      # RAG 知识库问答（Tab 6）
│
├── benchmarks/
│   └── speedtest.py            # LLM 吞吐量压测（Tab 7）
│
├── .env.example            # 配置模板（含注释，可提交）
├── .env                    # 实际配置（gitignore，不上传）
├── .gitignore
├── setup.bat / setup.sh    # 一键安装脚本
├── run.bat / run.sh        # 快速启动脚本
└── README.md               # 用户文档
```

**目录约定：**
| 目录 | 用途 | Git |
|------|------|-----|
| `data/` | 用户输入文件（图片、文本） | 忽略 |
| `outputs/` | 处理结果（翻译、压缩、评分等） | 忽略 |
| `outputs/history.json` | 处理历史记录（最近 100 条） | 忽略 |
| `outputs/benchmarks/` | 压测结果 JSON + 图表 PNG | 忽略 |

---

## 3. 架构设计

### 3.1 配置系统（config.py）

```
┌──────────┐    dotenv     ┌──────────┐    import     ┌──────────────┐
│  .env    │──────────────→│ config.py│←──────────────│ 所有工具模块  │
│ (用户编辑)│   load_dotenv │ (单例)    │  懒加载/直接   │              │
└──────────┘               └──────────┘               └──────────────┘
```

- 所有配置项通过 `os.getenv(KEY, default)` 读取，带合理默认值
- 默认 API 地址为 `http://localhost:1234/v1`（LM Studio 默认端口）
- `get_llm_extra_body()` 辅助函数统一控制思考模式开关
- **不要**在工具模块中覆写 `config.xxx`（V2.2 修复了 compress_chat.py 的这一问题）

**配置项分组（共 30+ 项）：**

| 分组 | 关键配置项 |
|------|-----------|
| API 连接 | `OPENAI_BASE_URL`, `OPENAI_API_KEY` |
| 推理设置 | `ENABLE_THINKING` |
| 模型名称 | `VISION_MODEL`, `TEXT_MODEL`, `VISION_MODEL_THINKING`, `RENAME_MODEL`, `BENCHMARK_MODEL` |
| 并发超时 | `DEFAULT_WORKERS`, `RETRY_TIMES`, `REQUEST_TIMEOUT`, `REQUEST_TIMEOUT_SHORT` |
| 图片处理 | `SLICE_HEIGHT`, `OVERLAP`, `TOP_PERCENT`, `BOTTOM_PERCENT`, `IMAGE_EXTENSIONS` |
| 文本翻译 | `DEFAULT_CHUNK_SIZE`, `OVERLAP_MESSAGES`, `SOURCE_LANG`, `TARGET_LANG` |
| 更新 | `AUTO_UPDATE` |
| 目录索引 | `DATA_DIR`, `OUTPUT_DIR`, `FAISS_INDEX_PATH`, `EMBEDDING_MODEL_PATH` |

### 3.2 Web UI 架构（app.py）

```
┌─────────────────────────────────────────────────────────┐
│                    Gradio gr.Blocks                      │
│                                                         │
│  Tab 0: 🏠 开始使用     新手引导 + 处理历史表格           │
│  Tab 1: 🖼️ 图片重命名    rename_images.py                │
│  Tab 2: 🔍 质量评分分类  detect_ai_errors.py             │
│  Tab 3: 💬 截图识别      ocr_chat_screenshots.py         │
│  Tab 4: 📝 聊天压缩      compress_chat.py                │
│  Tab 5: 🌐 文本翻译      translate.py                    │
│  Tab 6: 📚 知识库问答    chapter_summary.py              │
│  Tab 7: ⚡ LLM 压测      speedtest.py                    │
│  Tab 8: ⚙️ 设置         读写 .env + API 连接测试 + 更新   │
└─────────────────────────────────────────────────────────┘
```

**关键设计模式：**

1. **懒加载导入** — 每个 Tab 的处理函数内部 `from xxx import yyy`，避免启动时加载所有依赖。这意味着 `import` 错误只在用户实际使用该功能时才暴露。

2. **`_capture_log` 捕获日志** — 通用日志捕获函数（`app.py:29`），向 root logger 临时添加 StringIO handler，捕获工具内部 logging 输出返回给 UI：
   ```python
   def _capture_log(fn, *args, **kwargs):
       stream = io.StringIO()
       handler = logging.StreamHandler(stream)
       root = logging.getLogger()
       root.addHandler(handler)
       try:
           fn(*args, **kwargs)
       except Exception as e:
           stream.write(f"❌ 处理出错: {e}\n")
           stream.write(traceback.format_exc())
       finally:
           root.removeHandler(handler)
       return stream.getvalue() or "✅ 处理完成"
   ```
   **注意：** 函数内部异常会打印完整 traceback 到输出（含本地文件路径）。本地使用无问题，部署到公网需注意。

3. **进度回调模式** — 所有耗时工具统一签名：`progress_callback(completed: int, total: int)`。工具模块在 ThreadPoolExecutor 完成每个任务时回调，app.py 层封装为 `progress(completed/total, desc=...)` 驱动 Gradio 进度条：
   ```
   工具模块:  progress_callback(completed, total)
        ↓
   app.py:    progress(completed/total, desc=f"处理中 {completed}/{total}")
        ↓
   Gradio:    gr.Progress() 组件自动更新进度条
   ```

4. **历史记录** — `history.py` 提供 `add_entry()` 和 `get_recent()`。每个 Tab 的处理函数完成后调用 `history.add_entry()`。数据存为 `outputs/history.json`（JSON 数组，最多 100 条）。

### 3.3 设置页数据流

```
用户修改 UI 控件 → 点击保存
  → _on_save(*values)         # 按 SETTINGS_SCHEMA 顺序收集值
    → _save_env(updates)       # 正则逐行匹配 .env，更新或追加
      → 写入 .env 文件
      → 部分配置需手动重启生效
```

`SETTINGS_SCHEMA` 是一个列表，定义每个设置项的 key、显示名、默认值、类型（text/password/bool/int/float）。bool 类型使用 `gr.Dropdown(["True", "False"])`。

---

## 4. 模块详解

### 4.1 图片重命名（rename_images.py）

**功能：** 调用视觉模型看图，生成 3-25 个汉字的中文短句作为文件名。

**核心流程：**
1. 扫描目录中支持的图片格式（含大小写扩展名）
2. 按时间戳排序（优先从文件名提取，回退到文件修改时间）
3. 全局复用单个 ChatOpenAI 实例（`get_shared_llm()` 双重检查锁）
4. 多线程并行处理（ThreadPoolExecutor），共享最近 5 次描述作为上下文
5. 支持试运行模式（`--dry_run`）

**上下文机制：**
- 维护 `deque(maxlen=5)` 存储最近生成的描述
- 每次生成时把历史描述注入 Prompt，提示模型保持风格一致
- 线程安全：读用 `get_recent_descriptions()` 加锁复制，写用 `add_recent_description()` 加锁追加

**重试与回退：**
- 无中文输出 → 带提示重试一次
- 汉字数不在 3-25 范围 → 带要求重试一次
- 两次重试后仍不符 → 放弃，保留原名

**文件名时间戳提取：**
- 格式 1：`屏幕截图 2026-04-11 005848`
- 格式 2：`Client-Win64-Shipping-2025-11-12-19-37-57-600.jpg`
- 都转成统一格式用于排序

**V2.2 修复：** `encode_image()` 改用 PIL + `ImageOps.exif_transpose()` 处理 EXIF 方向。

### 4.2 图片质量评分（detect_ai_errors.py）

**功能：** 视觉模型评分 + 按百分比自动分拣到 HighQuality / LowQuality_Errors 子目录。

**两种模式：**
| 模式 | Prompt | 评分维度 |
|------|--------|---------|
| `ai` | AI_ERROR_PROMPT | 真实感、艺术性、细节协调、清晰度 |
| `photo` | PHOTOGRAPHY_PROMPT | 对焦清晰度、曝光、构图取景、色彩白平衡、姿态表情 |

**核心流程：**
1. 扫描目录中所有图片
2. `restore_original_name()` — 如果之前做过评分分类（文件名带评分前缀），先还原原名
3. 多线程调用 `detect_image_quality()` 获取 (score, error, reason)
4. 每张图生成同名 `.txt` 评分文件
5. 按评分排序，取 top N% 移至 HighQuality，bottom N% + 所有 ERR 移至 LowQuality_Errors
6. 阈值比例在 Web UI 可调（`TOP_PERCENT` / `BOTTOM_PERCENT`）
7. 提示词在 Web UI 可自定义

**输出解析逻辑：**
- `OK:<分数> <点评>` → 解析分数（0.0-10.0）+ 理由
- `ERR:<错误简述>` → 直接标记为错误（≤10 中文字）
- 兼容纯数字格式

**注意：** 初始 LLM 实例在模块级创建（`llm = ChatOpenAI(...)`），是单例。所有线程共用同一个实例。

### 4.3 截图 OCR（ocr_chat_screenshots.py）

**功能：** 将微信/QQ 聊天长截图切片后发给视觉模型识别，输出完整对话文本。

**切片策略：**
- 自顶向下按 `SLICE_HEIGHT`（默认 2000px）切分
- 相邻切片间有 `OVERLAP`（默认 400px）重叠防止漏字
- 最后一片从底部往上取 `SLICE_HEIGHT`

**并发策略：**
- 单张图 > 2 片且 `internal_workers > 1` 时，将切片对半分两组并发识别
- `process_folder()` 支持多张图片并行（`max_workers`）
- **注意：** 多图 × 内部并发的组合可能产生大量并发请求

**输出：** 每张图生成同名 `.txt`，内容为时间戳 + 说话人 + 消息的逐行文本。

### 4.4 聊天压缩（compress_chat.py）

**功能：** 将 OCR 输出的原始聊天记录 TXT 压缩精简，合并冗余时间戳，保留所有对话内容。

**核心流程：**
1. `parse_messages()` — 正则匹配时间戳 + 说话人行，收集后续消息体
2. `split_messages_into_chunks()` — 按消息完整边界分块，块间重叠 `OVERLAP_MESSAGES` 条
3. 多线程调用 LLM 压缩各块（`COMPRESS_PROMPT` 定义压缩规则）
4. 合并：去除相邻块重复的时间段头部，多空行归并

**容错设计（三层防护）：**
- 第 1-2 次尝试：正常 LLM 调用
- 第 3 次失败（最终）：返回 `[压缩失败，保留原文]\n{原文}`
- 执行异常：返回 `[执行异常，保留原文]\n{原文}`
- 结果为空：返回 `[结果为空，保留原文]\n{原文}`

**V2.2 修复：** 删除了模块级 `config.xxx = ...` 硬编码覆写，`request_timeout` 改用 `config.REQUEST_TIMEOUT`。

### 4.5 文本翻译（translate.py）

**功能：** 长篇文本按章节翻译，支持断点续传、分批并发。

**章节切分（`split_into_chapters_fast()`）：**
- 正则匹配 "第X章/节/卷" 等标题行
- 支持中文数字和阿拉伯数字
- 匹配到标题后按位置切片
- 无标题时整本书作为单章处理

**子块切分：** 单章超过 4000 字符时按段落边界切分为子块（保证段落不被截断）。

**翻译上下文：**
- 第一章第一块：使用 `PROMPT_FIRST`（无上文）
- 后续块：使用 `PROMPT_CONTINUE`，附前一块译文末尾 1000 字符作为上下文

**断点续传：**
- 进度保存到 `outputs/translation/progress.json`
- 每批次完成后原子写入
- `resume=True` 时从 `last_chapter_index` 继续

**并发模型：**
- 每批翻译 `batch_size` 章（默认 10），批内 `workers` 线程并发
- 一批完成后再开始下一批（保证进度可恢复）

### 4.6 知识库问答（chapter_summary.py）

**功能：** FAISS 向量检索 + BM25 关键词检索混合，多轮迭代回答。

**检索流程：**
1. 加载 HuggingFace Embedding 模型（默认 `BAAI/bge-small-zh-v1.5`）
2. 加载本地 FAISS 索引
3. 向量检索 + BM25 检索各取 top-k×2
4. 分数归一化后加权合并（`vector_weight` + `bm25_weight`）
5. 可选关键词硬过滤（只保留包含关键词的文档）

**多轮迭代：**
- 将检索结果分 `batch_size` 批
- 第一轮：`PROMPT_R1` 根据已知信息回答
- 后续轮：`PROMPT_RN` —— 给 LLM 上一轮自己的回答 + 新增信息，让它修正/深化
- streaming=True（流式输出）

**安全注意：** FAISS 使用 `allow_dangerous_deserialization=True`（pickle 反序列化）。仅加载用户自己的索引文件，本地使用可接受。

### 4.7 LLM 压测（speedtest.py）

**功能：** 测试 API 在不同输入长度、固定并发下的吞吐量。

**测量指标：**
- TTFT（Time To First Token）
- ITTL（Inter-Token Latency）
- Prefill Throughput（预填充吞吐量）
- Decode Throughput（输出吞吐量）

**实现细节：**
- 使用 `requests` 直接调 API（非 LangChain），支持 SSE 流式解析
- `tiktoken` 精确计数 token，fallback 到 `cl100k_base` 编码
- 输出双 Panel 图表（matplotlib）：预填充 + 输出吞吐 vs 输入长度
- 标注 P50/P90/P95 百分位线

---

## 5. 关键设计决策

### 5.1 思考模式控制

```python
def get_llm_extra_body() -> dict:
    if not ENABLE_THINKING:
        return {
            "thinking": False,
            "enable_thinking": False,
            "reasoning": False,
        }
    return {}
```

同时发送三个参数覆盖不同 API 后端的键名：LM Studio 用 `thinking`，某些 Ollama 适配用 `enable_thinking`，某些部署用 `reasoning`。对不支持思考模式的模型传这些参数不会报错（API 会忽略）。

### 5.2 为什么 ChatOpenAI 实例创建方式不统一

- `detect_ai_errors.py`：模块级单例（评分是高频短请求，复用连接）
- `rename_images.py`：`get_shared_llm()` 双重检查锁（全局复用 + 线程安全）
- `translate.py`：函数内创建单个，传参给各线程（共享同一实例）
- `compress_chat.py` / `ocr_chat_screenshots.py`：每次请求创建新实例（带重试）

**原因：** 各模块由不同时期开发，未统一重构。ChatOpenAI 底层使用 `requests.Session`，模块级复用可以减少 TCP 握手开销，但要注意线程安全性。建议后续统一为单例模式。

### 5.3 为什么使用 `sys.path.insert` 而非相对导入

```python
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
```

工具模块需要同时支持：
- 作为脚本直接运行（`python translate.py`）
- 被 app.py 导入调用

如果使用 `from .. import config`（相对导入），直接运行会报 `ImportError: attempted relative import with no known parent package`。`sys.path.insert` 是折中方案。

### 5.4 批量启动脚本的设计

```
setup.bat/sh: 检测 Python → 创建 .venv → pip install → cp .env.example .env → 启动 app.py
run.bat/sh:   直接启动 app.py（仮定 .venv 已存在）
```

- `setup.bat` 使用 `chcp 65001` 设置 UTF-8 代码页（Windows 中文不乱码）
- `run.bat` 保持纯 ASCII 输出，避免编码问题（V2.1 修复）
- `run.sh` 检测是否从 GUI 双击运行，是则自动打开终端窗口

---

## 6. 已知问题与注意事项

### 6.1 线程安全

- `_capture_log()` 向 root logger 添加/移除 handler，多 Tab 同时操作时可能互相干扰输出。实际上 Gradio 默认串行处理队列，影响不大。
- `rename_images.py` 的 `safe_rename()` 使用文件级锁（`_rename_lock`），防止多线程重命名冲突。

### 6.2 内存与性能

- `chapter_summary.py` 加载 Embedding 模型到内存（bge-small 约 100MB），首次查询延迟明显。
- `detect_ai_errors.py` 的 `encode_image()` 会将整张 PIL 图转为 base64 字符串在内存中，大批量处理时注意内存占用。
- `translate.py` 整个文件读入内存后处理，超大文件（>100MB）可能 OOM。

### 6.3 编码与平台兼容

- 所有 `.py` 文件声明 `# -*- coding: utf-8 -*-`
- Windows 批处理用 `chcp 65001` 切换 UTF-8
- `run.bat` V2.1 从中文改为纯 ASCII 输出，因为某些 Windows 版本 cmd.exe 对 UTF-8 中文支持不稳定
- `Path()` 对象在不同 OS 下自动适配路径分隔符

### 6.4 配置覆写陷阱（V2.2 修复）

**绝对不要**在工具模块的模块级别（import 时）修改 `config.xxx`。如果必须覆盖默认值，应在函数内部使用局部变量或参数传递。V2.1 时期 `compress_chat.py` 的 7 行硬编码覆写导致了"用 Tab 4 后其他 Tab 配置被篡改"的 bug。

### 6.5 EXIF 处理

所有读取图片发给视觉模型的代码，必须先用 `PIL.Image.open()` + `ImageOps.exif_transpose()` 应用 EXIF 旋转信息后再编码。直接读 `open(path, "rb")` 会导致 AI 看到旋转后的图片。

### 6.6 设置页类型映射

添加新配置项时注意 `SETTINGS_SCHEMA` 中的类型：
- `"text"` → `gr.Textbox`
- `"password"` → `gr.Textbox(type="password")`（掩码显示）
- `"bool"` → `gr.Dropdown(["True", "False"])`
- `"int"` / `"float"` → `gr.Textbox`（无输入校验，保存时直接写入 .env）

---

## 7. 开发指南

### 7.1 添加新工具 Tab

1. 在对应子目录创建工具模块（或复用现有）
2. 在 `app.py` 中编写处理函数（参考现有模式）：
   - 懒加载 import
   - progress_callback 适配
   - `_capture_log` 包装（如工具使用 logging）
   - `history.add_entry()` 记录
3. 在 `build_ui()` 中添加 Tab，编写使用说明 Accordion
4. 如需可配置参数，在 `config.py`、`.env.example`、`SETTINGS_SCHEMA` 三处同步添加

### 7.2 工具模块规范

最小可运行模板：
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from langchain_openai import ChatOpenAI

def process(input_path, progress_callback=None):
    llm = ChatOpenAI(
        model=config.TEXT_MODEL,
        base_url=config.OPENAI_BASE_URL,
        api_key=config.OPENAI_API_KEY,
        extra_body=config.get_llm_extra_body()
    )
    # ... 处理逻辑 ...
    if progress_callback:
        progress_callback(completed, total)

if __name__ == "__main__":
    # CLI 入口
    process(sys.argv[1] if len(sys.argv) > 1 else ".")
```

### 7.3 测试 API 连接

启动 LM Studio 或其他 OpenAI 兼容服务后：
```bash
# 测试单个工具
.venv\Scripts\python image_tools\detect_ai_errors.py test_image.jpg

# 启动 Web UI
.venv\Scripts\python app.py
# 浏览器打开 http://localhost:7860
# 切换到 ⚙️ 设置 → 点 "测试连接"
```

### 7.4 提交规范

- 版本号格式：`V<major>.<minor>`
- Commit message：`V<version> <中文简述>`
- 不提交 `.env`、`data/`、`outputs/`、`.venv/`
- `.env.example` 中的 API Key 默认值必须是 `lm-studio`（本地占位符），不可包含真实密钥

---

## 8. 版本演进

| 版本 | 主要变更 |
|------|---------|
| V1.0-V1.2 | 项目初始化，基础 Gradio 界面，核心工具功能 |
| V1.3 | 修复启动脚本 bug |
| V1.4 | 重写代码，增加网页版使用提示 |
| V1.5 | 增加 GUI 配置 API，不再需要手动编辑文件 |
| V1.6 | 设置页新增更新分组 |
| **V2.0** | 自动更新（git pull）、一键安装脚本（setup.bat/sh）、新模块、README 重写 |
| **V2.1** | 思考模式开关、EXIF 旋转修复、网页可调评分规则/提示词、所有 Tab 进度条、Linux 脚本、异常处理改进 |
| **V2.2** | 历史记录模块、修复 compress_chat.py 配置覆写 bug、修复硬编码超时、rename_images.py EXIF 处理 |

---

## 9. 依赖清单

```
langchain-openai          # ChatOpenAI LLM 调用
langchain-core            # LangChain 核心（消息类型）
langchain-community       # FAISS 向量存储集成
langchain-huggingface     # HuggingFace Embedding 模型
gradio                    # Web UI 框架
pillow                    # 图片 EXIF 处理、编码
numpy                     # 数值计算（百分位、均值）
matplotlib                # 压测图表
requests                  # HTTP（压测直接调 API）
tiktoken                  # Token 计数（压测用）
faiss-cpu                 # 向量相似度检索
jieba                     # 中文分词
rank-bm25                 # BM25 关键词检索
python-dotenv             # .env 文件加载
tqdm                      # 命令行进度条
```

---

*最后更新：2026-05-02 · V2.2*
