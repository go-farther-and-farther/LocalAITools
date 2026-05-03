# LocalAITools - 本地 AI 工具箱

> **零门槛上手**：下载 → 双击 `setup.bat` → 浏览器里填 API → 开用。

一套调用 AI 大模型的实用工具集合：图片智能重命名、AI 图片质量筛选、漫展摄影选片、聊天记录 OCR 提取与压缩、长篇翻译、知识库问答、API 性能压测。**所有功能免费使用，数据不上传云端。**

---

## 三步开始（用电脑就会）

### 第一步：下载 & 一键安装

1. 下载本项目 ZIP 包并解压
2. 双击 `setup.bat`
3. 脚本自动检测 Python 环境 → 安装依赖 → 启动

> 如果没有 Python，脚本会自动打开下载页面。安装 Python 时请勾选 **"Add Python to PATH"**。

### 第二步：获取 AI 模型服务

打开浏览器 `http://localhost:7860`，看「🏠 开始使用」页面，任选一种：

| 方式 | 难度 | 费用 | 说明 |
|------|------|------|------|
| **云端 API** | ⭐ 最简单 | 按量付费 | 硅基流动 / DeepSeek 注册 → 创建 Key → 填到设置里 |
| **LM Studio** | ⭐⭐ | 免费 | 下载软件 → 搜 qwen3 → 下载模型 → 点 Start Server |
| **Ollama** | ⭐⭐⭐ | 免费 | 命令行工具，需一点技术基础 |
| **vLLM / SGLang** | ⭐⭐⭐ | 免费 | 自建推理服务，适合多人共享 |

> 不想折腾 → 云端 API；有显卡想省钱 → LM Studio；技术党想自建 → Ollama / vLLM

### 第三步：填写配置 & 开用

1. 切换到 **⚙️ 设置** 标签页
2. 填写 API 地址和密钥（本地服务默认 `http://localhost:1234/v1`，云端填平台提供地址）
3. 点 **🔗 测试连接** 确认连上
4. 点 **💾 保存设置**
5. 切到对应功能 Tab，点「📖 使用方法」看步骤，开始使用！

---

## 功能一览

| 功能 | 适用场景 | 输入 → 输出 |
|------|---------|-------------|
| 🖼️ 图片重命名 | 整理图片素材 | 图片文件夹 → 中文短句文件名 |
| 🔍 质量评分分类 | 7种模式覆盖各类图片评选 | 图片 → 自动分拣高/低质量文件夹 |
| 💬 截图识别 | 聊天记录提取文字 | 长截图 → TXT 文本 |
| 📝 聊天压缩 | 精简冗余聊天 | TXT → 紧凑格式 TXT |
| 🌐 文本翻译 | 长文/小说翻译 | TXT → 翻译 TXT |
| 📚 知识库问答 | 文档智能检索 | 问题 → AI 检索回答 |
| ⚡ LLM 压测 | 测试 API 性能 | 参数 → 吞吐量图表 |

---

## 图片质量评分 — 七种检测模式

| 模式 | 适用场景 | 检测重点 |
|------|---------|---------|
| 🎨 AI 图片错误检测 | AI 生成图筛选 | 肢体/面部畸形、结构崩坏 |
| 📸 漫展摄影筛选 | Cosplay 返图选片 | 对焦模糊、过曝欠曝、构图表情 |
| 🖼️ 通用照片质量 | 日常照片综合评估 | 清晰度、曝光、构图、趣味性 |
| 👤 人像摄影评估 | 人物写真/证件照 | 面部清晰度、肤色、表情、虚化 |
| 🌄 风景摄影评估 | 风光/旅行照 | 光影层次、构图法则、色彩氛围 |
| 📄 文档扫描清晰度 | 扫描件/课件/合同 | 文字可读性、光照均匀、畸变 |
| 🖌️ 绘画插图质量 | 绘画/插画/原画 | 造型比例、线条笔触、完成度 |

切换方式：Tab 2 顶部下拉框选择模式。分类阈值支持 0.0~10.0 精确到 0.1 分。

---

## 命令行用法（可选，进阶用户）

```bash
# 图片 AI 重命名
python image_tools/rename_images.py -i data/images -w 4

# 图片质量评分（默认 AI 错误检测模式，--mode photo 为摄影模式）
python image_tools/detect_ai_errors.py data/images

# 聊天截图 → 文字
python image_tools/ocr_chat_screenshots.py -i data/screenshots

# 聊天记录压缩
python text_tools/compress_chat.py -i data/screenshots/texts

# 长篇翻译
python text_tools/translate.py -i data/texts/novel.txt -w 4

# 知识库问答
python text_tools/chapter_summary.py "你的问题" "可选关键词"

# API 性能压测
python benchmarks/speedtest.py --url http://localhost:1234/v1 --model qwen3.6-35b
```

---

## 配置说明

所有配置在 Web 界面的「⚙️ 设置」中修改，或直接编辑 `.env` 文件：

```bash
# API 连接
OPENAI_BASE_URL=http://localhost:1234/v1    # 本地服务默认地址，云端填平台提供地址
OPENAI_API_KEY=lm-studio                     # 本地填任意值，云端填真实的 Key

# 模型名称（按你下载的模型修改）
VISION_MODEL=qwen/qwen3.6-27b              # 视觉模型，用于图片识别
TEXT_MODEL=qwen/qwen3.5-9b                 # 文本模型，用于翻译压缩
```

---

## 推荐模型

| 用途 | 推荐模型 | 大小 |
|------|---------|------|
| 图片识别、质量评分 | `qwen3.6-27b` / `Qwen3-VL-30B` | ~16 GB |
| 聊天截图 OCR | `qwen3.6-35b-a3b-Thinking` | ~20 GB |
| 文本翻译、压缩 | `qwen3.5-9b` / `Qwen3-8B` | ~5 GB |
| Embedding（知识库） | `bge-m3` / `bge-large-zh-v1.5` | ~2 GB |

> 都在 LM Studio 搜索框里直接搜名字就能找到。

---

## 目录结构

```
LocalAITools/
├── setup.bat               # 👈 双击这个一键安装
├── run.bat                 # 已安装环境时快速启动
├── app.py                  # Web 界面入口
├── config.py               # 配置模块
├── .env.example            # 配置模板
│
├── data/                   # 输入目录
│   ├── images/             #   放图片
│   ├── screenshots/        #   放聊天截图
│   └── texts/              #   放文本文件
│
├── outputs/                # 输出目录
├── image_tools/            # 图片处理工具
├── text_tools/             # 文本 & 知识库工具
└── benchmarks/             # 性能测试
```
