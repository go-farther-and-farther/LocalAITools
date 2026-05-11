# LocalAITools - 本地 AI 工具箱

> **零门槛上手**：下载 → 双击 `setup.bat` → 浏览器里填 API → 开用。

一套调用 AI 大模型的实用工具集合：图片智能重命名、AI 图片分类、质量评分筛选、聊天截图 OCR 提取与压缩、长篇翻译、知识库问答、API 性能压测。**所有功能免费使用，数据不上传云端。**

---

## 三步开始

### 第一步：下载 & 一键安装

1. 下载本项目 ZIP 包并解压
2. 双击 `setup.bat`
3. 脚本自动检测 Python 环境 → 安装依赖 → 启动

> 如果没有 Python，脚本会自动打开下载页面。安装 Python 时请勾选 **"Add Python to PATH"**。

### 第二步：获取 AI 模型服务

打开浏览器 `http://localhost:7860`，看「🏠 开始使用」页面，任选一种：

| 方式 | 难度 | 费用 | 说明 |
|------|------|------|------|
| **内置本地模型** | ⭐ 最简单 | 免费 | 双击 `setup_local_model.bat` 自动下载 Qwen3.5 0.6B，无需额外软件 |
| **云端 API** | ⭐ 简单 | 按量付费 | 硅基流动 / DeepSeek 注册 → 创建 Key → 填到设置里 |
| **LM Studio** | ⭐⭐ | 免费 | 下载软件 → 搜 qwen3 → 下载模型 → 点 Start Server |
| **Ollama** | ⭐⭐⭐ | 免费 | 命令行工具，需一点技术基础 |

> 不想折腾 → 内置本地模型或云端 API；有显卡想用大模型 → LM Studio；技术党 → Ollama / vLLM

### 第三步：填写配置 & 开用

1. 切换到 **⚙️ 设置** 标签页
2. 填写 API 地址和密钥（本地服务默认 `http://localhost:1234/v1`，云端填平台提供地址）
3. 点 **🔗 测试连接** 确认连上
4. 点 **💾 保存设置**
5. 切到对应功能 Tab，点「📖 使用方法」看步骤，开始使用！

---

## 内置本地模型

不想装 LM Studio？项目内置了轻量级本地模型支持：

1. 双击 `setup_local_model.bat` → 选择下载 Qwen3.5-0.6B（~400MB）或 1.7B（~1GB）
2. 启动 app，进入 **⚙️ 设置 → 🏠 内置本地模型**
3. 点「启动本地模型」
4. 顶部供应商栏选择「本地模型」即可使用

**技术细节：**
- 基于 `llama-cpp-python`，CPU 推理，无需 GPU
- 自动启动 OpenAI 兼容 API 服务（端口 8081）
- 支持流式输出、思考模式
- 模型文件放在 `models/` 目录（`.gguf` 格式）

---

## 功能一览

| 功能 | 适用场景 | 输入 → 输出 |
|------|---------|-------------|
| 🤖 AI 智能分类 | 按内容自动归类图片 | 图片文件夹 → 分类子文件夹 |
| 🖼️ 图片重命名 | 整理图片素材 | 图片文件夹 → 中文短句文件名 |
| 🔍 质量评分分类 | 7种模式覆盖各类图片评选 | 图片 → 自动分拣高/低质量文件夹 |
| 💬 截图识别 | 聊天记录提取文字 | 长截图 → TXT 文本 |
| 📝 聊天压缩 | 精简冗余聊天 | TXT → 紧凑格式 TXT |
| 🌐 文本翻译 | 长文/小说翻译 | TXT → 翻译 TXT |
| 📚 知识库问答 | 文档管理 + 对话式检索 | 上传文档 → 构建索引 → 对话问答 |
| ⚡ LLM 压测 | 测试 API 性能 | 参数 → 吞吐量图表 |

---

## 图片工具

### AI 智能分类

用 AI 识别每张图片的内容，自动归入对应分类文件夹。适合整理大量混杂图片。

- **勾选分类** — 9 个预设类别（照片、动漫、游戏、绘画、截图、风景、美食、文档、其他），勾选需要的类别即可
- **自定义补充** — 可在输入框添加额外分类，支持 `名称：描述` 格式让 AI 分类更准确
- **⚡ 智能分组加速** — 基于感知哈希（pHash）自动将相似图片归组，每组只调用一次 AI（一次发送多张图）。1000 张相似截图可能只需几十次 API 调用，速度提升 10-50 倍
  - 相似度阈值可调（0-64，默认 20）。游戏截图建议 30-48
  - 每组采样数可调（1-8，默认 4），送多张图给 AI 看更准
- **并行处理** — 可配置 1-8 个并行线程
- **试运行** — 先预览效果再正式移动
- **试运行结果保存** — 试运行结果自动保存，满意后一键应用，无需重新调用 AI
- **停止按钮** — 随时中断

### 图片重命名

- **六种命名模式** — 通用描述 / 人像聚焦 / 风景聚焦 / 截图识别 / 美食聚焦 / 动漫二次元
- **自定义提示词** — 用户可自由定义 AI 描述图片的方式，留空则使用模式默认提示词
- **试运行结果保存** — 同上，试运行后可一键应用
- **保留原文件名** — 带时间戳的文件名很有用，勾选后描述追加在原名前面
- **上下文参考** — 可配置最近 N 条描述作为风格参考（0-10）
- **图片压缩** — 设置最大边长（512-4096px），降低 API 传输量

### 图片按作品分类

重命名后如果文件名包含 `《作品名》`，可一键按作品归类到子文件夹。支持设置最少数量阈值。

### 图片质量评分 — 七种检测模式

| 模式 | 适用场景 | 检测重点 |
|------|---------|---------|
| 🎨 AI 图片错误检测 | AI 生成图筛选 | 肢体/面部畸形、结构崩坏 |
| 📸 漫展摄影筛选 | Cosplay 返图选片 | 对焦模糊、过曝欠曝、构图表情 |
| 🖼️ 通用照片质量 | 日常照片综合评估 | 清晰度、曝光、构图、趣味性 |
| 👤 人像摄影评估 | 人物写真/证件照 | 面部清晰度、肤色、表情、虚化 |
| 🌄 风景摄影评估 | 风光/旅行照 | 光影层次、构图法则、色彩氛围 |
| 📄 文档扫描清晰度 | 扫描件/课件/合同 | 文字可读性、光照均匀、畸变 |
| 🖌️ 绘画插图质量 | 绘画/插画/原画 | 造型比例、线条笔触、完成度 |

评分完成后自动展示 Gallery 画廊，支持导出 CSV 报告。

---

## 聊天

- 多对话管理 — 创建、切换、删除、搜索对话
- 流式输出 — 实时显示 AI 回复
- 思考过程可见 — 支持 Thinking 模型的思考过程展示
- 联网搜索 — AI 自主判断是否需要搜索，自动拆搜索词
- RAG 知识库检索 — 侧边栏可调检索片段数（1-30）
- 对话导出 — 导出为 Markdown 文件
- Enter 发送 / Shift+Enter 换行
- 自动保存 — 每次收发消息自动存盘

---

## 知识库问答

三步使用：上传文档 → 构建索引 → 对话问答。

### 文档管理

- 支持格式：`.txt` `.md` `.csv` `.json` `.jsonl` `.log` `.py` `.rst` `.pdf` `.docx` `.html` `.htm`
- URL 网页抓取 — 输入网页地址，自动提取文本内容入库
- 上传、删除、清空操作

### 索引构建

- 自动分块（可配置块大小和重叠）
- HuggingFace Embedding 模型向量化（默认 `bge-small-zh-v1.5`）
- FAISS 向量索引，支持本地离线使用
- 构建后自动生效，无需重启

### 对话问答

- 对话式界面，支持多轮上下文
- 混合检索：向量相似度 + BM25 关键词匹配
- 可选关键词过滤
- 聊天记录可保存、加载、删除

---

## 文本工具

### 聊天记录压缩

OCR 生成的 TXT → 精简格式化。合并连续相同说话人的时间戳，移除系统消息，保留所有对话实质。

### 长篇翻译

- 按章节切分翻译，支持断点续传
- 术语表 — 输入 `原文=译文` 确保专有名词一致
- 章节进度表格 — 查看每章翻译状态
- 并行翻译 — 可配置每批章节数和并行线程数

---

## 供应商管理

支持多个 API 供应商快速切换：

- 顶部供应商下拉框一键切换
- 每个供应商独立存储 API 地址和密钥
- 切换后所有工具自动使用新供应商的配置
- ➕ 添加 / ✏️ 编辑 / 🗑️ 删除供应商
- 内置本地模型自动注册为「本地模型」供应商

---

## 个性化

### 主题色彩

设置页「🎨 主题色彩」选择颜色，全局渐变色立即生效，刷新后依然保持。

### 状态记忆

所有工具的参数（输入路径、模型选择、并行数、试运行勾选等）自动保存，下次打开自动恢复。

---

## 配置说明

所有配置在 Web 界面的「⚙️ 设置」中修改，或直接编辑 `.env` 文件：

```bash
# API 连接
OPENAI_BASE_URL=http://localhost:1234/v1    # 本地服务默认地址
OPENAI_API_KEY=lm-studio                     # 本地填任意值，云端填真实的 Key

# 模型名称
VISION_MODEL=qwen/qwen3.6-27b              # 视觉模型，用于图片识别
VISION_MODEL_THINKING=qwen3.6-35b-a3b-Thinking  # 思考版视觉模型
RENAME_MODEL=qwen/Qwen3.6-27b              # 图片重命名模型
TEXT_MODEL=qwen/qwen3.5-9b                 # 文本模型

# 内置本地模型（可选）
LOCAL_MODEL_ENABLED=false                    # 设为 true 开机自动启动本地模型
LOCAL_MODEL_PATH=                            # 模型文件路径，留空自动检测 models/ 目录
LOCAL_MODEL_CTX=4096                         # 上下文窗口大小

# 图片处理
IMAGE_MAX_SIZE=2048                          # 输入图片最大边长（像素）

# 知识库
FAISS_INDEX_PATH=faiss_index                 # FAISS 索引目录
EMBEDDING_MODEL_PATH=                        # Embedding 模型路径，留空在线下载
KB_CHUNK_SIZE=500                            # 文本块大小（字符）
KB_CHUNK_OVERLAP=50                          # 块间重叠（字符）
```

---

## 命令行用法（可选，进阶用户）

```bash
# 图片 AI 重命名
python image_tools/rename_images.py -i data/images -w 4 --mode portrait

# 图片质量评分
python image_tools/detect_ai_errors.py data/images --mode photo

# 聊天截图 → 文字
python image_tools/ocr_chat_screenshots.py -i data/screenshots

# 聊天记录压缩
python text_tools/compress_chat.py -i data/screenshots/texts

# 长篇翻译
python text_tools/translate.py -i data/texts/novel.txt -w 4

# 知识库问答
python text_tools/kb_chat.py "你的问题" "可选关键词"

# API 性能压测
python benchmarks/speedtest.py --url http://localhost:1234/v1 --model qwen3.6-35b
```

---

## 推荐模型

| 用途 | 推荐模型 | 大小 |
|------|---------|------|
| 图片识别、质量评分 | `qwen3.6-27b` / `Qwen3-VL-30B` | ~16 GB |
| 聊天截图 OCR | `qwen3.6-35b-a3b-Thinking` | ~20 GB |
| 文本翻译、压缩 | `qwen3.5-9b` / `Qwen3-8B` | ~5 GB |
| 内置本地模型 | `Qwen3.5-0.6B` (快速) / `Qwen3.5-1.7B` (更准) | ~400MB / ~1GB |
| Embedding（知识库） | `bge-small-zh-v1.5` / `bge-m3` | ~1-2 GB |

---

## 目录结构

```
LocalAITools/
├── setup.bat / run.bat         # 一键安装 / 快速启动
├── setup_local_model.bat       # 内置本地模型安装脚本
├── app.py                      # Web 界面入口
├── config.py                   # 配置模块
├── history.py                  # 操作历史记录
├── requirements.txt            # Python 依赖
│
├── ui/                         # UI 模块
│   ├── common.py               #   共享组件（模型选择器、日志捕获）
│   ├── providers.py            #   供应商管理
│   ├── tab_welcome.py          #   开始使用页
│   ├── tab_image_tools.py      #   图片工具（分类+重命名+评分）
│   ├── tab_ocr.py              #   截图识别
│   ├── tab_chat.py             #   聊天
│   ├── tab_kb.py               #   知识库管理
│   ├── tab_text_tools.py       #   文本工具（压缩+翻译）
│   ├── tab_benchmark.py        #   LLM 压测
│   └── tab_settings.py         #   设置页（含本地模型管理）
│
├── services/                   # 服务层（纯 Python，无 Gradio 依赖）
│   ├── image_services.py       #   图片工具服务（重命名/评分/分类/试运行应用）
│   ├── text_services.py        #   文本工具服务（压缩/翻译）
│   ├── ocr_services.py         #   OCR 服务
│   ├── benchmark_services.py   #   压测服务
│   └── local_model.py          #   内置本地模型服务
│
├── utils/                      # 工具函数
│   ├── kb_chat_helpers.py      #   对话/知识库工具
│   └── web_search.py           #   联网搜索（含缓存）
│
├── image_tools/                # 图片处理核心
│   ├── rename_images.py        #   图片重命名
│   ├── detect_ai_errors.py     #   图片质量评分
│   └── ocr_chat_screenshots.py #   截图 OCR
│
├── text_tools/                 # 文本 & 知识库
│   ├── compress_chat.py        #   聊天记录压缩
│   ├── translate.py            #   长篇翻译
│   ├── kb_chat.py              #   知识库问答引擎
│   └── kb_manager.py           #   知识库文档管理
│
├── benchmarks/                 # 性能测试
│   └── speedtest.py            #   LLM 压测
│
├── models/                     # 本地模型文件（.gguf）
├── static/                     # 前端资源（CSS/JS）
├── data/                       # 输入数据目录
└── outputs/                    # 输出结果目录
```
