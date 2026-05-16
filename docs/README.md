# LocalAITools - 本地 AI 工具箱

> **零门槛上手**：下载 → 双击 `安装.bat` → 浏览器里填 API → 开用。

一套调用 AI 大模型的实用工具集合：图片智能重命名、AI 图片分类、质量评分筛选、聊天截图 OCR 提取与压缩、长篇翻译、知识库问答、API 性能压测。**所有功能免费使用，数据不上传云端。**

---

## 三步开始

### 第一步：下载 & 一键安装

1. 下载本项目 ZIP 包并解压
2. 双击 `安装.bat`
3. 脚本自动检测 Python 环境 → 安装依赖 → 启动

> 如果没有 Python，脚本会自动打开下载页面。安装 Python 时请勾选 **"Add Python to PATH"**。

### 第二步：获取 AI 模型服务

打开浏览器 `http://localhost:7860`，看「🏠 开始使用」页面，任选一种：

| 方式 | 难度 | 费用 | 说明 |
|------|------|------|------|
| **⭐ llama.cpp（推荐）** | ⭐ 最简单 | 免费 | 下载解压 → 添加供应商 → 选模型 → 一键启动，无需额外软件 |
| **云端 API** | ⭐ 简单 | 按量付费 | 硅基流动 / DeepSeek 注册 → 创建 Key → 填到设置里 |
| **LM Studio** | ⭐⭐ | 免费 | 下载软件 → 搜 qwen3 → 下载模型 → 点 Start Server |
| **Ollama** | ⭐⭐⭐ | 免费 | 命令行工具，需一点技术基础 |

> 💡 **推荐：** 本地跑模型首选 **llama.cpp**，性能最好、显存最省；不想折腾 → 云端 API；技术党 → Ollama / vLLM

### 第三步：填写配置 & 开用

1. 切换到 **⚙️ 设置** 标签页
2. 填写 API 地址和密钥（本地服务默认 `http://localhost:1234/v1`，云端填平台提供地址）
3. 点 **🔗 测试连接** 确认连上
4. 点 **💾 保存设置**
5. 切到对应功能 Tab，点「📖 使用方法」看步骤，开始使用！

---

## ⭐ llama.cpp 本地推理（推荐）

直接用 llama.cpp 跑本地模型，性能最好、显存最省，无需安装 LM Studio。

### 下载 llama.cpp

1. 打开 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases)
2. 找到最新版本，下载 **`llama-bxxx-bin-win-vulkan-x64.zip`**（Windows + Vulkan GPU 加速）
   - 有 NVIDIA 显卡也可用 `cuda` 版本
   - 无显卡用 `cpu` 版本
3. 解压到任意目录，如 `D:\llama\llama-b9116-bin-win-vulkan-x64\`

### 下载模型

推荐从 [HuggingFace](https://huggingface.co) 或 [ModelScope](https://modelscope.cn) 下载 GGUF 格式模型：

| 模型 | 大小 | 说明 |
|------|------|------|
| `Qwen3.6-27B-Q4_K_S` | ~16 GB | 通用主力，视觉+文本 |
| `Qwen3.6-35B-A3B-UD-Q3_K_M` | ~16 GB | MoE 架构，速度快 |
| `Qwen3.5-9B-Q4_K_M` | ~5 GB | 文本任务够用 |
| `Qwen3.5-4B-Q4_K_M` | ~2.5 GB | 轻量快速 |

> 模型文件放在 `.lmstudio/models/` 或任意目录均可，添加供应商时自动扫描。

### 在应用中使用

1. 启动应用，点击顶栏 **添加** 供应商
2. 类型选 **`llama_server`**
3. 填写 **llama 安装目录**（如 `D:\llama\llama-b9116-bin-win-vulkan-x64`）
4. 点 **🔍 扫描模型**，自动列出所有可用 `.gguf` 模型
5. 选择模型 → 保存
6. 在顶栏选择该供应商 → 点 **▶️ 启动服务**
7. 等待 10-60 秒加载完成，即可使用

**特性：**
- 自动检索 `llama-server.exe`，无需手动指定完整路径
- 自动扫描 `.lmstudio/models/` 下的模型文件
- 视觉模型自动检测同目录下的 `mmproj` 文件
- 支持 MoE 模型 Expert Count 配置
- 支持推理模式（Reasoning）开关
- 启动/停止一键管理，无需命令行

---

## 功能一览

| 标签页 | 功能 | 适用场景 |
|--------|------|---------|
| 🖼️ 图片工具 | AI 智能分类 + 重命名 + 评分筛选 | 图片整理一条龙 |
| 💬 聊天 | 多对话 AI 助手 + 联网搜索 + 知识库 | 日常问答、文档检索 |
| 📚 知识库 | 文档上传 + 索引构建 | 打造个人知识库 |
| 📝 文本工具 | 聊天压缩 + 长篇翻译 | 文本后处理 |
| ⚡ LLM 压测 | API 吞吐量测试 | 评估模型服务性能 |

---

## 图片工具

图片工具页面包含三个子标签：

### 🤖 AI 智能分类

用 AI 识别每张图片的内容，自动归入对应分类文件夹。

- **9 个预设类别** — 照片、动漫、游戏、绘画、聊天截图、应用截图、风景、美食、文档、其他，勾选需要的即可
- **自定义补充** — 支持 `名称：描述` 格式添加额外分类
- **⚡ 智能分组加速** — 相似图片自动归组，一组只调一次 AI（速度提升 10-50 倍）
- **并行处理** — 1-8 线程可调
- **试运行保存** — 满意后一键应用，不重复调用 AI

### 🏷️ 图片重命名

AI 生成中文短句替换杂乱文件名，右侧内置 **📁 按《》作品名自动分类**。

- **六种命名模式** — 通用描述 / 人像 / 风景 / 截图 / 美食 / 动漫
- **自定义提示词** — 自由定义 AI 描述风格
- **包含子文件夹** — 支持处理 1-2 层子目录中的图片
- **按作品分类** — 文件名含《作品名》自动归入子文件夹，支持提取子文件夹图片
- **试运行保存** — 预览满意后一键正式改名

### 🔍 评分与分类

三个步骤，通过内嵌子标签切换：

| 子标签 | 功能 |
|--------|------|
| 📊 评分 | 7 种检测模式 → AI 评分 → 画廊预览 → 导出 CSV |
| 📂 分拣 | 按比例/按分值自动分拣到 HighQuality / LowQuality_Errors |
| 🔍 审核 | 逐张查看评分结果，手动决定每张图去向 |

**七种检测模式：**

| 模式 | 适用场景 | 检测重点 |
|------|---------|---------|
| 🎨 AI 图片错误检测 | AI 生成图筛选 | 肢体/面部畸形、结构崩坏 |
| 📸 漫展摄影筛选 | Cosplay 返图选片 | 对焦模糊、过曝欠曝、构图表情 |
| 🖼️ 通用照片质量 | 日常照片评估 | 清晰度、曝光、构图、趣味性 |
| 👤 人像摄影评估 | 人物写真/证件照 | 面部清晰度、肤色、表情、虚化 |
| 🌄 风景摄影评估 | 风光/旅行照 | 光影层次、构图法则、色彩氛围 |
| 📄 文档扫描清晰度 | 扫描件/课件/合同 | 文字可读性、光照均匀、畸变 |
| 🖌️ 绘画插图质量 | 绘画/插画/原画 | 造型比例、线条笔触、完成度 |

---

## 聊天

- **多对话管理** — 创建、切换、删除、搜索对话
- **流式输出** — 实时显示 AI 回复，自动保持阅读位置不跳底
- **思考过程** — Thinking 模型的推理过程可见
- **联网搜索** — AI 自主判断是否需要搜索，自动拆搜索词，最多 3 轮
- **知识库检索** — 侧边栏可调检索片段数（1-30），混合向量+BM25 检索
- **对话导出** — 导出为 Markdown 文件
- **快捷键** — Enter 发送 / Shift+Enter 换行
- **自动保存** — 每次收发消息自动存盘，AI 自动生成对话标题

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
- 每个供应商独立存储配置
- 切换后所有工具自动使用新供应商的配置
- ➕ 添加 / ✏️ 编辑 / 🗑️ 删除供应商
- **`llama_server` 类型**：内置进程管理，一键启动/停止本地模型服务
- **`openai_compatible` 类型**：连接 LM Studio、Ollama、云端 API 等外部服务

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

# 图片处理
IMAGE_MAX_SIZE=2048                          # 输入图片最大边长（像素）

# 知识库
FAISS_INDEX_PATH=faiss_index                 # FAISS 索引目录
EMBEDDING_MODEL_PATH=                        # Embedding 模型路径，留空在线下载
KB_CHUNK_SIZE=500                            # 文本块大小（字符）
KB_CHUNK_OVERLAP=50                          # 块间重叠（字符）
```

> 💡 本地模型通过供应商系统管理，配置保存在 `data/state.json` 中，无需手动编辑。

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
| 轻量本地推理 | `Qwen3.5-4B` / `Qwen3.5-1.7B` | ~2.5 GB / ~1 GB |
| Embedding（知识库） | `bge-small-zh-v1.5` / `bge-m3` | ~1-2 GB |

---

## 目录结构

```
LocalAITools/
├── 安装.bat / 启动.bat          # 一键安装 / 快速启动
├── app.py                      # Web 界面入口
├── config.py                   # 配置模块
├── history.py                  # 操作历史记录
├── requirements.txt            # Python 依赖
│
├── ui/                         # UI 模块
│   ├── common.py               #   共享组件（模型选择器、日志捕获）
│   ├── providers.py            #   供应商管理（含 llama.cpp 集成）
│   ├── tab_welcome.py          #   开始使用页
│   ├── tab_image_tools.py      #   图片工具（分类+重命名+评分）
│   ├── tab_ocr.py              #   截图识别
│   ├── tab_chat.py             #   聊天
│   ├── tab_kb.py               #   知识库管理
│   ├── tab_text_tools.py       #   文本工具（压缩+翻译）
│   ├── tab_benchmark.py        #   LLM 压测
│   └── tab_settings.py         #   设置页
│
├── services/                   # 服务层（纯 Python，无 Gradio 依赖）
│   ├── image_services.py       #   图片工具服务（重命名/评分/分类/试运行应用）
│   ├── text_services.py        #   文本工具服务（压缩/翻译）
│   ├── ocr_services.py         #   OCR 服务
│   ├── benchmark_services.py   #   压测服务
│   └── llama_server.py         #   llama.cpp 服务管理（启动/停止/自动检索）
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
├── static/                     # 前端资源（CSS/JS）
├── data/                       # 输入数据目录（含 state.json 供应商配置）
└── outputs/                    # 输出结果目录
```
