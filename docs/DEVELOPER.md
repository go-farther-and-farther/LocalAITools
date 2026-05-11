# LocalAITools — 开发者技术文档

> 目标读者：后续接手开发的工程师。本文档涵盖项目架构、模块设计、关键决策和已知坑位。

---

## 1. 项目概述

LocalAITools 是一套**本地优先的 AI 工具集**，通过 Gradio Web UI 调用 OpenAI 兼容 API（LM Studio / Ollama / 云端 API / 内置本地模型），提供图片分析、文本处理、知识库问答、LLM 压测等功能。

**核心原则：**
- 所有数据默认不上传云端，API 地址由用户自行配置
- 一键安装启动，面向非技术用户也面向开发者
- 纯 Python + Gradio，无前端构建工具链
- 配置文件驱动（.env + state.json），Web UI 内可直接编辑
- 每个工具模块可独立命令行运行，也可通过 Web UI 调用

**技术栈：**
- Python 3.10+
- Gradio 6.x（gr.Blocks 模式）
- LangChain（ChatOpenAI + HuggingFaceEmbeddings + FAISS）
- Pillow / matplotlib / numpy / jieba / rank-bm25
- llama-cpp-python（可选，内置本地模型）

---

## 2. 项目结构

```
LocalAITools/
├── app.py                  # Gradio Web UI 入口（~185 行）
├── config.py               # 全局配置中心（读取 .env + state.json）
├── history.py              # 处理历史记录模块
├── requirements.txt        # Python 依赖
│
├── ui/                     # UI 模块
│   ├── common.py           # 共享 UI 组件（模型选择器、日志捕获等）
│   ├── providers.py        # 供应商管理（增删改查）
│   ├── tab_welcome.py      # 开始使用页
│   ├── tab_image_tools.py  # 图片工具（AI分类+重命名+评分+画廊）
│   ├── tab_ocr.py          # 截图识别
│   ├── tab_chat.py         # 聊天（含搜索+导出+联网+RAG）
│   ├── tab_kb.py           # 知识库管理
│   ├── tab_text_tools.py   # 文本工具（压缩+翻译）
│   ├── tab_benchmark.py    # LLM 压测（含历史记录）
│   └── tab_settings.py     # 设置（含本地模型管理+备份恢复+主题）
│
├── services/               # 服务层（纯 Python，无 Gradio 依赖）
│   ├── image_services.py   # 图片工具服务
│   ├── text_services.py    # 文本工具服务
│   ├── ocr_services.py     # OCR 服务
│   ├── benchmark_services.py # 压测服务
│   └── local_model.py      # 内置本地模型服务
│
├── utils/                  # 工具函数
│   ├── kb_chat_helpers.py  # 对话/知识库辅助函数
│   └── web_search.py       # 联网搜索（Bing CN + DDG，含缓存）
│
├── image_tools/            # 图片处理核心
│   ├── rename_images.py    # AI 图片重命名
│   ├── detect_ai_errors.py # 图片质量评分 + 评分结果加载
│   └── ocr_chat_screenshots.py # 聊天截图 OCR
│
├── text_tools/             # 文本 & 知识库
│   ├── compress_chat.py    # 聊天记录压缩
│   ├── translate.py        # 长篇文本翻译
│   ├── kb_chat.py          # RAG 知识库问答引擎
│   └── kb_manager.py       # 知识库文档管理
│
├── benchmarks/
│   └── speedtest.py        # LLM 吞吐量压测
│
├── static/
│   ├── chat.css            # 聊天页样式（含暗色主题+主题色彩）
│   └── chat.js             # 聊天页 JS（Enter发送+主题+DOM稳定性）
│
├── models/                 # 本地模型文件（.gguf）
├── setup_local_model.bat   # 本地模型一键安装脚本
└── docs/
    ├── README.md           # 用户文档
    └── DEVELOPER.md        # 本文档
```

**目录约定：**

| 目录 | 用途 | Git |
|------|------|-----|
| `data/` | 用户输入文件 | 忽略 |
| `data/state.json` | 工具参数 + 供应商设置持久化 | 忽略 |
| `data/dry_run_rename.json` | 试运行重命名结果 | 忽略 |
| `data/dry_run_classify.json` | 试运行分类结果 | 忽略 |
| `outputs/` | 处理结果 | 忽略 |
| `models/` | .gguf 模型文件 | 仅 .gitkeep |

---

## 3. 架构设计

### 3.1 分层架构

```
┌─────────────────────────────────────────────────┐
│  UI 层 (ui/tab_*.py)                            │
│  Gradio 组件 + 事件绑定 + thin wrapper 函数      │
├─────────────────────────────────────────────────┤
│  服务层 (services/*.py)                          │
│  纯 Python 业务逻辑，无 Gradio 依赖              │
├─────────────────────────────────────────────────┤
│  工具层 (image_tools/ text_tools/ benchmarks/)   │
│  核心算法 + LLM 调用 + 文件操作                  │
├─────────────────────────────────────────────────┤
│  基础层 (config.py + utils/)                     │
│  配置 + 搜索 + 知识库辅助                        │
└─────────────────────────────────────────────────┘
```

**UI 层 → 服务层调用模式：**
```python
# ui/tab_image_tools.py
def _rename_images(input_dir, model, workers, ...):
    _apply_provider(provider)                    # 1. 应用供应商配置
    result = _svc_rename(input_dir, model, ...)  # 2. 调用服务层
    config.save_state("rename", ...)             # 3. 保存状态
    history.add_entry("图片重命名", ...)          # 4. 记录历史
    return result
```

### 3.2 配置系统（config.py）

**双配置源：**
- `.env` — 环境变量（API 地址、模型名称、并发参数等）
- `data/state.json` — UI 状态持久化（每个工具的上次参数、供应商列表）

```python
# .env 读取
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")

# state.json 读写
config.load_state("rename")  # → {"input_dir": "...", "model": "...", ...}
config.save_state("rename", input_dir="...", model="...", workers=4)
```

**state.json 结构：**
```json
{
  "providers": {
    "list": [
      {"name": "默认", "base_url": "http://localhost:1234/v1", "api_key": "lm-studio"},
      {"name": "本地模型", "base_url": "http://127.0.0.1:8081/v1", "api_key": "local"}
    ],
    "active": "默认"
  },
  "rename": {"input_dir": "...", "model": "...", "workers": 4, "dry_run": false, ...},
  "score": {"input_dir": "...", "mode": "ai", ...},
  "classify": {"input_dir": "...", "classify_method": "percent", ...},
  "ai_classify": {"input_dir": "...", "categories": "...", ...},
  "ocr": {"input_dir": "...", "model": "...", ...},
  "compress": {"input_path": "...", "model": "...", ...},
  "translate": {"input_file": "...", "model": "...", ...},
  "benchmark": {"url": "...", "model": "...", ...}
}
```

每个工具的 `load_state` 在 `app.py` 的 `build_ui()` 中统一加载，传给各 Tab 的 `render_tab_*()` 函数。

### 3.3 供应商管理

```
供应商下拉框切换
  → _on_provider_change(name, providers)
    → 更新 provider_info State
    → _apply_provider(provider_info) 写入 os.environ + config 模块变量
    → _refresh_all_models(provider_info) 刷新所有 Tab 的模型列表
```

**`_apply_provider()` 为什么同时写 `os.environ` 和 `config` 模块：**
- `config.py` 在 import 时读取 `os.getenv()` 赋值给模块变量
- LangChain ChatOpenAI 在创建时读取 `config.OPENAI_BASE_URL`
- 只修改 `os.environ` 不够，config 模块变量已缓存旧值
- 只修改 `config` 不够，有些代码直接读环境变量

### 3.4 内置本地模型（services/local_model.py）

```
app.py 启动
  → auto_start() 检查 LOCAL_MODEL_ENABLED
    → start_server_simple(model_path, n_ctx, n_threads)
      → Llama() 加载 .gguf 模型
      → HTTPServer 提供 OpenAI 兼容 API
      → 自动注册为「本地模型」供应商
```

**两种服务模式：**
1. `start_server()` — 使用 llama-cpp-python 内置的 uvicorn 服务器（需安装 uvicorn）
2. `start_server_simple()` — 使用 Python 标准库 http.server（零额外依赖，推荐）

**API 端点：**
- `GET /v1/models` — 返回模型列表
- `POST /v1/chat/completions` — 聊天补全（支持流式）

**自动端口选择：** 从 8081 开始扫描找到空闲端口。

### 3.5 状态持久化（config.save_state / load_state）

所有工具的 UI 参数在执行时自动保存到 `data/state.json`，下次打开自动恢复。

**保存时机：** 每个工具 wrapper 函数执行完成后调用 `config.save_state(key, ...)`。

**加载时机：** `app.py` 的 `build_ui()` 开头统一加载，传给各 Tab 的渲染函数。

**试运行结果持久化：**
- `data/dry_run_rename.json` — 重命名试运行映射
- `data/dry_run_classify.json` — 分类试运行映射
- 用户可一键应用，无需重新调用 AI

---

## 4. 模块详解

### 4.1 图片重命名（rename_images.py + image_services.py）

**六种命名模式：** general / portrait / landscape / screenshot / food / anime，各有专用 Prompt。

**核心流程：**
1. 扫描目录中支持的图片格式（含大小写扩展名）
2. 按时间戳排序（8 种文件名格式识别 + 文件修改时间回退）
3. 全局复用单个 ChatOpenAI 实例（`get_shared_llm()` 双重检查锁 + 模型变更检测）
4. 多线程并行处理，共享最近 N 次描述作为上下文
5. 支持试运行 + 试运行结果保存/应用

**试运行结果保存：**
```python
# 试运行时收集结构化结果
structured_pairs.append({"old_name": old_name, "new_stem": new_phrase})
# 保存到 JSON
_DRY_RUN_RENAME_FILE.write_text(json.dumps({...}))

# 应用时读取 JSON 逐条执行 safe_rename()
def apply_rename_results(input_dir, keep_original=False) -> str
```

### 4.2 AI 智能分类（image_services.py）

**功能：** 视觉模型识别图片内容，自动归入用户定义的分类文件夹。

**分类描述支持：** 输入格式 `名称：描述`，如 `游戏：游戏相关的图片和同人插画,照片：真实拍摄的照片,其他`。描述会注入 Prompt 让 AI 分类更准确。

**并行 + 停止：** ThreadPoolExecutor + threading.Event。

**试运行结果保存：** 同重命名，保存到 `data/dry_run_classify.json`。

### 4.3 图片质量评分（detect_ai_errors.py）

**七种检测模式：** ai / photo / general / portrait / landscape / document / illustration。

**核心流程：** 多线程调用视觉模型 → 每张图生成 `.txt` 评分文件 → 按百分比或阈值分拣到 HighQuality / LowQuality_Errors。

**评分结果可视化：** `load_scored_results()` 从 `.txt` 文件加载 → Gallery 画廊展示 → CSV 导出。

### 4.4 截图 OCR（ocr_chat_screenshots.py）

**切片策略：** 自顶向下按 SLICE_HEIGHT（2000px）切分，相邻 OVERLAP（400px）防漏字。

**并发：** 多图并行 + 单图内部切片对半并发。

### 4.5 聊天压缩（compress_chat.py）

正则解析时间戳 → 按消息边界分块 → LLM 压缩各块 → 合并去重。三层容错（重试 → 保留原文 → 异常捕获）。

### 4.6 文本翻译（translate.py）

章节切分 → 子块切分 → 分批并发翻译 → 断点续传。支持术语表注入。

### 4.7 知识库问答（kb_chat.py + kb_manager.py）

**文档格式：** txt / md / csv / json / pdf / docx / html / url。

**检索：** FAISS 向量 + BM25 关键词混合，分数归一化加权合并。

**多轮对话：** chat_history 注入 Prompt，支持指代消解。

### 4.8 聊天功能（tab_chat.py）

**AI 自主搜索：** AI 分析问题 → 输出 `[SEARCH:关键词]` 搜索 → 最多 3 轮 → 信息够了输出 `[ANSWER]` 回答。

**流式输出：** 后台线程 + 生成器，支持停止按钮（cancels 取消）。

**思考过程：** Gradio 6.x 原生 thought blocks，`metadata={"title": "🤔 思考过程"}`。

**DOM 稳定性：** MutationObserver + `document.contains()` 检测 stale DOM，防止前端卡死。

### 4.9 联网搜索（web_search.py）

Bing CN（lxml 解析）+ DuckDuckGo（ddgs 库）双引擎。内存缓存 TTL 10 分钟，最多 100 条，线程安全。

### 4.10 LLM 压测（speedtest.py + benchmark_services.py）

requests 直接调 API + SSE 流式解析。tiktoken 计数。matplotlib 双 Panel 图表。自动保存历史记录。

---

## 5. 关键设计决策

### 5.1 思考模式控制

```python
def get_llm_extra_body(enabled: bool = None) -> dict:
    if enabled is None:
        enabled = os.getenv("ENABLE_THINKING", "true").lower() == "true"
    return {"enable_thinking": enabled}
```

每次调用时动态读取环境变量。对不支持思考模式的模型传参不会报错。

### 5.2 LLM 单例与模型变更检测

```python
def get_shared_llm(model: str) -> ChatOpenAI:
    if _llm_instance is None or getattr(_llm_instance, 'model', None) != model:
        with _llm_lock:
            if _llm_instance is None or getattr(_llm_instance, 'model', None) != model:
                _llm_instance = ChatOpenAI(model=model, ...)
    return _llm_instance
```

双重检查锁 + 模型变更检测。切换模型时旧单例自动重建。

### 5.3 为什么使用 `sys.path.insert` 而非相对导入

工具模块需同时支持直接运行和被 app.py 导入。相对导入在直接运行时会报 `ImportError`。

### 5.4 Windows 重启机制

`subprocess.Popen` + `os._exit(0)`，因为 `os.execv()` 在 Windows 上对带空格路径有 bug。

### 5.5 RGBA 图片处理

JPEG/WebP 不支持透明通道，保存前 `img.convert('RGB')`。修复了 RGBA PNG 保存报错的问题。

### 5.6 搜索缓存

缓存 `None` 结果避免反复请求失败后端。key 格式 `query||max_results`。

### 5.7 本地模型 HTTP 服务

使用标准库 `http.server` 而非 uvicorn，零额外依赖。支持 `/v1/models` 和 `/v1/chat/completions`（含流式）。

### 5.8 试运行结果保存

试运行时收集结构化映射保存到 JSON，应用时读取 JSON 逐条执行。避免用户满意后重新调用 AI 浪费时间和 API。

---

## 6. 已知问题与注意事项

### 6.1 线程安全

- `safe_rename()` 使用文件级锁防止多线程重命名冲突
- 历史描述使用 `deque + _history_lock`
- `_capture_log()` 向 root logger 添加 handler，多 Tab 同时操作可能干扰（Gradio 默认串行队列影响不大）

### 6.2 内存与性能

- Embedding 模型加载约 100MB，首次查询延迟明显
- `encode_image()` 将 PIL 图转 base64，大批量注意内存
- 翻译整文件读入内存，超大文件（>100MB）可能 OOM
- 本地模型加载 .gguf 需要足够内存（0.6B 约 400MB，1.7B 约 1GB）

### 6.3 编码与平台兼容

- Windows 批处理用 `chcp 65001` 切换 UTF-8
- `Path()` 自动适配路径分隔符
- 重启用 `subprocess.Popen` 处理 Windows 路径空格

### 6.4 配置覆写陷阱

**绝对不要**在工具模块的模块级别修改 `config.xxx`。必须覆盖时用局部变量或参数传递。

### 6.5 EXIF 处理

所有读取图片发给视觉模型的代码，必须先用 `ImageOps.exif_transpose()` 应用 EXIF 旋转。

### 6.6 Gradio 6 兼容性

- `gr.Chatbot` 移除了 `type` 参数（默认 messages 格式）
- `gr.Chatbot` 支持 `metadata={"title": "..."}` 实现原生 thought blocks
- `gr.update()` 仍可用但推荐直接返回组件

### 6.7 FAISS 安全

使用 `allow_dangerous_deserialization=True`（pickle）。仅加载用户自己的索引文件，本地使用可接受。

---

## 7. 开发指南

### 7.1 添加新工具 Tab

1. 在对应子目录创建工具模块
2. 在 `services/` 创建服务函数（纯 Python，无 Gradio）
3. 在 `ui/tab_*.py` 创建渲染函数 + thin wrapper
4. wrapper 模式：`_apply_provider` → 调用服务 → `save_state` → `history.add_entry`
5. 在 `app.py` 的 `build_ui()` 中添加 Tab，返回组件字典供 `_refresh_all_models` 使用

### 7.2 工具模块规范

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

_stop_flag = threading.Event()

def request_stop():
    _stop_flag.set()

def process(input_path, progress_callback=None):
    _stop_flag.clear()
    # ... 处理逻辑 ...
    if progress_callback:
        progress_callback(completed, total)
```

### 7.3 测试

```bash
# 启动 Web UI
python app.py

# 浏览器打开 http://localhost:7860
# 切换到 ⚙️ 设置 → 点 "测试连接"

# 单独测试工具
python image_tools/detect_ai_errors.py data/images --mode photo
```

### 7.4 提交规范

- 版本号格式：`V<major>.<minor>`
- Commit message：`V<version> <中文简述>`
- 不提交 `.env`、`data/`、`outputs/`、`.venv/`、`models/*.gguf`

---

## 8. 依赖清单

```
# 核心
langchain-openai          # ChatOpenAI
langchain-core            # 消息类型、Document
langchain-community       # FAISS 集成
langchain-huggingface     # HuggingFace Embedding
langchain-text-splitters  # 文本分块
gradio                    # Web UI（6.x）
openai                    # API 客户端

# 图片
pillow                    # EXIF 处理、编码

# 知识库
faiss-cpu                 # 向量检索
sentence-transformers     # Embedding 模型
jieba                     # 中文分词
rank-bm25                 # BM25 检索

# 文本
tiktoken                  # Token 计数
PyPDF2                    # PDF 解析
python-docx               # DOCX 解析
beautifulsoup4            # HTML 解析

# 搜索 & 网络
requests                  # HTTP
lxml                      # HTML 解析（Bing 搜索）
ddgs                      # DuckDuckGo 搜索

# 工具
python-dotenv             # .env 加载
tqdm                      # 进度条
numpy                     # 数值计算
matplotlib                # 图表

# 可选
llama-cpp-python          # 内置本地模型
huggingface-hub           # 模型下载
```

---

## 9. 版本演进

| 版本 | 主要变更 |
|------|---------|
| V1.x | 项目初始化，基础 Gradio 界面 |
| V2.0 | 自动更新、一键安装、新模块 |
| V2.1 | 思考模式开关、EXIF 修复、评分规则可调 |
| V2.2 | 历史记录、配置覆写 bug 修复 |
| V2.3-V2.9 | 供应商管理、模型选择、RGBA 修复 |
| V3.0 | Tab 合并、重命名 6 种模式、按作品分类 |
| V3.1 | 知识库管理器、对话式问答、聊天保存 |
| V3.2 | app.py 模块拆分、对话导出/搜索、联网搜索缓存、PDF/DOCX/HTML 支持、评分画廊、压测历史、备份恢复 |
| V3.3 | 内置本地模型（llama-cpp-python）、AI 智能分类（并行+停止）、试运行结果保存/应用、分类描述支持、状态持久化全覆盖、主题色彩、思考过程显示、前端稳定性修复 |

---

*最后更新：2026-05-11 · V3.3*
