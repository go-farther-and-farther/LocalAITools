import re
import sys
import io as io_module
import base64
import time
import logging
import shutil
import threading
from pathlib import Path
from typing import Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from tqdm import tqdm

# ================= 配置 =================
IMAGE_EXTENSIONS = config.IMAGE_EXTENSIONS
HIGH_QUALITY_FOLDER = config.HIGH_QUALITY_FOLDER
LOW_QUALITY_ERRORS_FOLDER = config.LOW_QUALITY_ERRORS_FOLDER

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

# ================= 停止标记 =================
_stop_flag = threading.Event()

def request_stop():
    """请求停止当前正在执行的评分任务"""
    _stop_flag.set()

# ================= 模型（每次创建新实例，确保读取最新的 ENABLE_THINKING 配置） =================
def _get_llm(model: str = None):
    return ChatOpenAI(
        model=model or config.VISION_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        temperature=0.2,
        max_tokens=2048,  # 需要足够大以容纳思考 tokens + 输出 tokens
        timeout=config.REQUEST_TIMEOUT_SHORT,
        extra_body=config.get_llm_extra_body()
    )

AI_ERROR_PROMPT = """你是一个专业的图像质量评审。请综合以下四个维度，对图片进行 0.0~10.0 的评分（精确一位小数），并给出简要点评。

维度：
1. 真实感：是否接近真实照片或优秀绘画，有无明显 AI 痕迹。
2. 艺术性：构图、色彩、光影、氛围是否有美感。
3. 细节协调：肢体、五官、物体结构、文字等是否合理自然。
4. 清晰度：图像是否锐利，有无模糊或压缩伪影。

评分参考（连续区间，无断档）：
- 9.0-10.0：非常出色，几乎无瑕疵。
- 7.0-8.9：整体良好，仅细微不足。
- 5.0-6.9：中等，有较明显的不足，但可接受。
- 3.0-4.9：较差，多个维度有明显缺陷。
- 0.0-2.9：质量很低，存在严重错误或崩坏。

输出格式：
- 如果图片存在明显的严重缺陷（如肢体错乱、面部畸形、物体结构崩溃），输出 "ERR:<错误简述>"，错误简述不超过10个中文字。
- 否则输出 "OK:<分数> <点评>"，点评用中文概括主要优缺点，50-100字。

不要输出其他任何内容。"""

PHOTOGRAPHY_PROMPT = """你是一个专业的漫展/Cosplay 现场选片助手。你的任务不是艺术评论，而是快速、准确地从大量照片中挑出“能交片”的图，并标记废片。

请严格基于以下五个维度对照片进行 0.0~10.0 的连续评分（精确到一位小数）：

1. 对焦与清晰度（权重最高）：
   - 核心焦点必须在人物面部，尤其是较近的那只眼睛。眼珠无高光细节、睫毛模糊即视为失焦。
   - 允许大光圈下的前景/背景虚化，但主体（面部、服装关键纹理）必须清晰。
   - 因手抖或快门过慢导致的整体模糊，直接视为废片。

2. 曝光（注重可用性）：
   - 主体面部和服装高光区不能出现大面积死白（RGB 全 255 无纹理），暗部不能出现大片死黑。
   - 轻微过曝/欠曝但保留细节的，可在点评中指出“后期可救”，不影响基础评分。
   - 逆光未补光导致面部全黑，视为严重欠曝。

3. 构图与取景（漫展特化）：
   - 竖幅人像是标准构图，不要将其判断为“方向错误”。当前显示方向即为正确方向。
   - 加分项：人物居中或三分线构图，背景简洁、虚化漂亮，利用光线勾勒轮廓。
   - 扣分项：关节处（手腕、脚踝、膝盖）生硬裁切；头顶空间极窄或切头；背景有突兀的路人、指示牌、垃圾桶、反光板穿帮。
   - 注意：Cosplay 照片允许留出空间展示全身服装与道具，半身特写与全身照同样合理。

4. 色彩与白平衡：
   - 肤色还原为第一标准，面色蜡黄、发绿、发紫、过饱和都需扣分。
   - 如果现场有定向彩色灯光（如红蓝舞台灯）导致肤色改变，请在点评中说明“环境光染色，非白平衡故障”，并据此适当放宽评分，但仍需指出。

5. 姿态与表情（选片关键）：
   - Coser 表情必须到位：眼睛睁开有神，嘴型自然或符合角色设定，无“半眨眼”“翻白眼”“崩坏脸”。
   - 动作不能处在尴尬过渡帧，手指应舒展而非蜷缩，肢体不遮挡面部关键点。
   - 若为刻意还原角色的“面瘫”“闭眼”“歪嘴”等表情，需根据角色常识判断，无法确定时按一般标准处理。
   - 多人合照要同时检查每个人的表情，一人闭眼即为废片。

评分参考（连续区间，无断档）：
- 9.0 - 10.0：影楼级直出佳作，对焦精准、曝光完美、构图讲究、表情富有感染力，完全符合 Coser 交片标准。
- 7.0 - 8.9：良好成片，仅有轻微缺陷（如背景略杂、肤色需微调），无需二次裁剪即可保留。
- 5.0 - 6.9：中等，存在明显问题（如轻微跑焦、高光微溢、表情平淡），仅在没有更好同角度的照片时保留。
- 3.0 - 4.9：较差，多维度缺陷（失焦、欠曝、构图残缺），通常建议淘汰，除非有重要纪念意义。
- 0.0 - 2.9：废片，已无挽救价值，直接删除。

输出格式（严格遵守）：
- 如果照片存在以下严重缺陷之一，直接输出：ERR:<错误简述>（简述不超过10个中文字，例如“全黑欠曝”“严重跑焦模糊”“大面积死白”）
- 否则，输出完整评估结果，格式为：
  OK:<分数> <操作标记> <点评>
  - 操作标记固定为三个选项之一：【推荐保留】【可留】 【删除】
  - 点评控制在30-70字，用中文概括主要优缺点，并对可修正缺陷给出简短后期建议（如“可二次构图避开左侧路人”）。

不要输出任何多余内容，不要评价照片的显示方向，不要提出旋转建议。"""

# 默认使用 AI 错误检测模式
SYSTEM_PROMPT = AI_ERROR_PROMPT

GENERAL_PHOTO_PROMPT = """你是一个专业的照片质量评审。请综合以下四个维度，对照片进行 0.0~10.0 的评分（精确一位小数），并给出简要点评。

维度：
1. 清晰度与对焦：主体是否清晰锐利，有无模糊、跑焦、运动拖影。
2. 曝光与色彩：曝光是否正常，有无过曝死白或欠曝死黑，白平衡是否自然。
3. 构图与取景：构图是否合理，主体位置、背景处理是否恰当，画面是否平衡。
4. 内容趣味性：画面的内容是否有吸引力、故事性或观赏价值。

评分参考（连续区间，无断档）：
- 9.0-10.0：非常出色，几乎无瑕疵。
- 7.0-8.9：整体良好，仅细微不足。
- 5.0-6.9：中等，有较明显的问题但可接受。
- 3.0-4.9：较差，多个维度有明显缺陷。
- 0.0-2.9：质量很低，存在严重问题。

输出格式：
- 如果照片存在严重缺陷（如完全模糊、大面积死白死黑、画面全黑），输出 "ERR:<错误简述>"，错误简述不超过10个中文字。
- 否则输出 "OK:<分数> <点评>"，点评用中文概括主要优缺点，50-100字。

不要输出其他任何内容。"""

PORTRAIT_PROMPT = """你是一个专业的人像摄影评审。请综合以下五个维度，对人像照片进行 0.0~10.0 的评分（精确一位小数），并给出简要点评。

维度：
1. 人物清晰度：面部/眼睛是否合焦锐利，皮肤细节是否可见。
2. 光线与肤色：光线是否柔和自然，肤色是否健康好看，有无油光、死黑、阴阳脸。
3. 构图与背景：人物在画面中的位置是否合理，背景是否简洁不抢戏，有无头部"长树"等构图错误。
4. 表情与神态：人物表情是否自然到位，眼神是否有神采，有无闭眼、歪嘴、表情僵硬。
5. 氛围与美感：整体画面是否有氛围感，虚化是否舒服，色调是否和谐。

评分参考（连续区间，无断档）：
- 9.0-10.0：影楼级佳作，各方面均优秀，可直接出片。
- 7.0-8.9：整体良好，仅细微不足，值得保留。
- 5.0-6.9：中等，有较明显问题，可选择性保留。
- 3.0-4.9：较差，多个维度有明显缺陷。
- 0.0-2.9：废片，存在严重问题。

输出格式：
- 如果照片存在严重缺陷（如完全失焦、面部模糊不清、严重闭眼、大面积死白），输出 "ERR:<错误简述>"，错误简述不超过10个中文字。
- 否则输出 "OK:<分数> <点评>"，点评用中文概括主要优缺点，50-100字。

不要输出其他任何内容。"""

LANDSCAPE_PROMPT = """你是一个专业的风景摄影评审。请综合以下四个维度，对风景照片进行 0.0~10.0 的评分（精确一位小数），并给出简要点评。

维度：
1. 清晰度：远景和近景是否锐利，有无大气雾霾导致的朦胧、相机抖动模糊。
2. 曝光与光影：高光是否溢出（天空死白），暗部是否有细节，光影层次是否丰富。
3. 构图：是否符合三分法/引导线/前景框架等构图原则，水平线是否水平。
4. 色彩与氛围：色彩是否自然或具有艺术感，天气和光线氛围是否动人。

评分参考（连续区间，无断档）：
- 9.0-10.0：壁纸级佳作，各方面均出色。
- 7.0-8.9：整体良好，仅细微不足。
- 5.0-6.9：中等，有较明显问题但可接受。
- 3.0-4.9：较差，多个维度有明显缺陷。
- 0.0-2.9：质量很低，存在严重问题。

输出格式：
- 如果照片存在严重缺陷（如完全模糊、严重偏色、画面几乎全黑/全白），输出 "ERR:<错误简述>"，错误简述不超过10个中文字。
- 否则输出 "OK:<分数> <点评>"，点评用中文概括主要优缺点，50-100字。

不要输出其他任何内容。"""

DOCUMENT_PROMPT = """你是一个专业的文档/扫描件质量评审。请综合以下四个维度，对文档图片进行 0.0~10.0 的评分（精确一位小数），并给出简要点评。

维度：
1. 文字可读性：文字是否清晰可辨认，有无模糊、断裂、粘连。
2. 光照均匀度：是否有反光、阴影、半边暗的问题，光线是否均匀。
3. 畸变与角度：是否有透视畸变、倾斜、弯曲变形，页面是否平整。
4. 完整性与整洁度：内容是否完整无缺失，有无污渍、折痕、手指遮挡。

评分参考（连续区间，无断档）：
- 9.0-10.0：扫描级质量，文字锐利、页面平整、光线均匀。
- 7.0-8.9：整体良好，轻微阴影或畸变，完全可读。
- 5.0-6.9：中等，有较明显问题但主体内容可辨认。
- 3.0-4.9：较差，部分文字难以辨认或页面变形严重。
- 0.0-2.9：无法使用，文字基本不可读或内容严重缺失。

输出格式：
- 如果文档存在严重缺陷（如完全不可读、大面积遮挡、严重反光导致内容缺失），输出 "ERR:<错误简述>"，错误简述不超过10个中文字。
- 否则输出 "OK:<分数> <点评>"，点评用中文概括主要优缺点，50-100字。

不要输出其他任何内容。"""

ART_PROMPT = """你是一个专业的绘画/插图质量评审。请综合以下四个维度，对作品进行 0.0~10.0 的评分（精确一位小数），并给出简要点评。

维度：
1. 造型与比例：人物/物体的造型是否准确，比例是否协调，有无明显结构错误。
2. 线条与笔触：线条是否流畅有控制力，笔触是否有表现力，有无潦草杂乱。
3. 色彩与光影：色彩搭配是否和谐，光影关系是否正确，色调是否有氛围。
4. 构图与完成度：构图是否饱满有层次，细节是否丰富，完成度是否高。

评分参考（连续区间，无断档）：
- 9.0-10.0：专业级作品，各方面均优秀。
- 7.0-8.9：整体良好，仅细微不足。
- 5.0-6.9：中等，有较明显问题但整体可看。
- 3.0-4.9：较差，多个维度有明显缺陷。
- 0.0-2.9：质量很低，存在严重问题。

输出格式：
- 如果作品存在严重缺陷（如结构严重崩坏、大面积涂抹、画面完全混乱），输出 "ERR:<错误简述>"，错误简述不超过10个中文字。
- 否则输出 "OK:<分数> <点评>"，点评用中文概括主要优缺点，50-100字。

不要输出其他任何内容。"""

MODE_PROMPTS = {
    "ai": AI_ERROR_PROMPT,
    "photo": PHOTOGRAPHY_PROMPT,
    "general": GENERAL_PHOTO_PROMPT,
    "portrait": PORTRAIT_PROMPT,
    "landscape": LANDSCAPE_PROMPT,
    "document": DOCUMENT_PROMPT,
    "art": ART_PROMPT,
}

def _get_prompt(mode: str) -> str:
    return MODE_PROMPTS.get(mode, AI_ERROR_PROMPT)


def get_mime_type(suffix: str) -> str:
    mime_map = {
        '.jpg': 'jpeg', '.jpeg': 'jpeg', '.png': 'png',
        '.webp': 'webp', '.bmp': 'bmp', '.tiff': 'tiff'
    }
    return mime_map.get(suffix.lower(), 'jpeg')


def encode_image(image_path: Path, max_size: int = None) -> str:
    if max_size is None:
        max_size = config.IMAGE_MAX_SIZE
    from PIL import Image, ImageOps
    img = Image.open(image_path)
    # 应用 EXIF 方向信息，防止 AI 看到旋转后图片而误判
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    # 压缩到 max_size，减少内存和传输量
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io_module.BytesIO()
    fmt = img.format or 'JPEG'
    if fmt.upper() == 'WEBP' and img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def restore_original_name(image_path: Path) -> Path:
    stem = image_path.stem
    match = re.match(r'^(\d+)(分)?_(.+)$', stem)
    if match:
        original_stem = match.group(3)
        new_name = original_stem + image_path.suffix
        new_path = image_path.with_name(new_name)
        try:
            image_path.rename(new_path)
            logging.info(f"已还原: {image_path.name} -> {new_name}")
            return new_path
        except Exception as e:
            logging.warning(f"还原失败 {image_path.name}: {e}")
            return image_path
    return image_path


def detect_image_quality(image_path: Path, mode: str = "ai", custom_prompt: str = "", model: str = None, max_size: int = None) -> Tuple[Optional[float], Optional[str], str]:
    prompt = custom_prompt.strip() if custom_prompt.strip() else _get_prompt(mode)
    for attempt in range(1, config.RETRY_TIMES + 1):
        try:
            base64_img = encode_image(image_path, max_size=max_size)
            message = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{get_mime_type(image_path.suffix)};base64,{base64_img}"
                    }
                }
            ])
            response = _get_llm(model).invoke([message])
            result = response.content.strip()

            if result.startswith("OK:"):
                parts = result[3:].strip().split(maxsplit=1)
                if len(parts) >= 1:
                    try:
                        score = float(parts[0])
                        reason = parts[1] if len(parts) > 1 else ""
                        if 0.0 <= score <= 10.0:
                            logging.info(f"📸 {image_path.name} → 评分: {score} 分  {reason}")
                            return round(score, 1), None, reason
                    except ValueError:
                        pass
            elif result.startswith("ERR:"):
                error = result[4:].strip()
                if error:
                    logging.info(f"📸 {image_path.name} → ❌ 错误: {error}")
                    return None, error, ""

            # 兼容纯数字
            try:
                score = float(result)
                if 0.0 <= score <= 10.0:
                    logging.info(f"📸 {image_path.name} → 评分: {score} 分")
                    return round(score, 1), None, ""
            except ValueError:
                pass

            logging.warning(f"无法解析 ({image_path.name}): {result[:50]}")
            return None, None, ""

        except Exception as e:
            logging.warning(f"第{attempt}次失败 ({image_path.name}): {e}")
            if attempt < config.RETRY_TIMES:
                time.sleep(2)
            else:
                logging.error(f"失败 ({image_path.name}): {e}")
                return None, None, ""
    return None, None, ""


def save_txt(image_path: Path, score: Optional[float], error: Optional[str], reason: str):
    txt_path = image_path.with_suffix(".txt")
    if error:
        content = f"错误: {error}"
    elif score is not None:
        content = f"评分: {score} 分\n理由: {reason}" if reason else f"评分: {score} 分"
    else:
        return
    try:
        txt_path.write_text(content, encoding="utf-8")
    except Exception as e:
        logging.error(f"保存 txt 失败 ({txt_path.name}): {e}")


def safe_move(src: Path, dst_dir: Path) -> bool:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    counter = 1
    while dst.exists():
        stem = src.stem
        dst = dst_dir / f"{stem}_{counter}{src.suffix}"
        counter += 1
    try:
        shutil.move(str(src), str(dst))
        logging.info(f"移动: {src.name} -> {dst_dir.name}/{dst.name}")
        return True
    except Exception as e:
        logging.error(f"移动失败 {src.name}: {e}")
        return False


def move_with_txt(src: Path, dst_dir: Path) -> bool:
    success = safe_move(src, dst_dir)
    if success:
        txt_src = src.with_suffix(".txt")
        if txt_src.exists():
            safe_move(txt_src, dst_dir)
    return success


def load_scores_from_txt(target_dir: str) -> List[Tuple[Path, Optional[float], Optional[str]]]:
    """从目录中已有的 .txt 文件读取评分结果，返回 [(path, score, error), ...]"""
    dir_path = Path(target_dir)
    results = []
    for txt_path in sorted(dir_path.glob("*.txt")):
        img_path = txt_path.with_suffix("")
        for ext in IMAGE_EXTENSIONS:
            candidate = txt_path.with_suffix(ext)
            if candidate.exists():
                img_path = candidate
                break
        try:
            content = txt_path.read_text(encoding="utf-8").strip()
            if content.startswith("错误:"):
                error = content[3:].strip()
                results.append((img_path, None, error))
            elif content.startswith("评分:"):
                m = re.match(r'评分:\s*([\d.]+)\s*分', content)
                if m:
                    results.append((img_path, float(m.group(1)), None))
        except Exception:
            pass
    return results


def load_review_queue(target_dir: str) -> list:
    """加载目录中所有已评分图片的完整信息，按分数从低到高排序，返回 [{path, score, error, reason}, ...]"""
    dir_path = Path(target_dir)
    items = []
    for txt_path in sorted(dir_path.glob("*.txt")):
        img_path = None
        for ext in IMAGE_EXTENSIONS:
            candidate = txt_path.with_suffix(ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue
        try:
            content = txt_path.read_text(encoding="utf-8").strip()
            if content.startswith("错误:"):
                items.append({"path": img_path, "score": None, "error": content[3:].strip(), "reason": ""})
            elif content.startswith("评分:"):
                m = re.match(r'评分:\s*([\d.]+)\s*分(?:\s*\n?\s*理由:\s*(.*))?', content, re.DOTALL)
                if m:
                    items.append({"path": img_path, "score": float(m.group(1)), "error": None, "reason": (m.group(2) or "").strip()})
        except Exception:
            pass
    # 按分数从低到高排序（None=错误放最前面）
    items.sort(key=lambda x: x["score"] if x["score"] is not None else -1)
    return items


def load_scored_results(target_dir: str) -> List[dict]:
    """从目录中已有的 .txt 文件读取评分结果，返回按分数从高到低排序的列表。
    每个元素: {path, score, error, reason, classification}
    classification: 'high' (>=7), 'low' (<4) 或 'mid'
    """
    dir_path = Path(target_dir)
    items = []
    for txt_path in sorted(dir_path.glob("*.txt")):
        img_path = None
        for ext in IMAGE_EXTENSIONS:
            candidate = txt_path.with_suffix(ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue
        try:
            content = txt_path.read_text(encoding="utf-8").strip()
            if content.startswith("错误:"):
                items.append({"path": str(img_path), "score": None, "error": content[3:].strip(), "reason": "", "classification": "low"})
            elif content.startswith("评分:"):
                m = re.match(r'评分:\s*([\d.]+)\s*分(?:\s*\n?\s*理由:\s*(.*))?', content, re.DOTALL)
                if m:
                    score = float(m.group(1))
                    reason = (m.group(2) or "").strip()
                    cls = "high" if score >= 7 else ("low" if score < 4 else "mid")
                    items.append({"path": str(img_path), "score": score, "error": None, "reason": reason, "classification": cls})
        except Exception:
            pass
    items.sort(key=lambda x: x["score"] if x["score"] is not None else -1, reverse=True)
    return items


def move_single_to_category(img_path: str, category: str, target_dir: str) -> dict:
    """手动审核：将单张图片移到指定分类目录。category: 'high', 'low', 'undo'"""
    src = Path(img_path)
    dir_path = Path(target_dir)
    if not src.exists():
        return {"ok": False, "msg": f"文件不存在: {src.name}"}

    if category == "undo":
        # 从分类目录移回原目录
        high_dir = dir_path / HIGH_QUALITY_FOLDER
        low_dir = dir_path / LOW_QUALITY_ERRORS_FOLDER
        for subdir in [high_dir, low_dir]:
            moved_src = subdir / src.name
            if moved_src.exists():
                safe_move(moved_src, dir_path)
                txt = moved_src.with_suffix(".txt")
                if txt.exists():
                    safe_move(txt, dir_path)
                return {"ok": True, "msg": f"已撤销: {src.name} → {dir_path.name}"}
        return {"ok": False, "msg": "未在分类目录中找到该文件"}

    dst_dir = dir_path / (HIGH_QUALITY_FOLDER if category == "high" else LOW_QUALITY_ERRORS_FOLDER)
    ok = move_with_txt(src, dst_dir)
    label = "高质量" if category == "high" else "低质量"
    return {"ok": ok, "msg": f"已移入{label}: {src.name}" if ok else f"移动失败: {src.name}"}


def score_images(target_dir: str, mode: str = "ai", progress_callback=None,
                 custom_prompt: str = "", model: str = None, max_size: int = None):
    """仅评分，不分类。为每张图片生成 .txt 评分文件"""
    dir_path = Path(target_dir)
    if not dir_path.is_dir():
        logging.error(f"目录不存在: {target_dir}")
        return []

    image_files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_files:
        logging.info("未找到支持的图片文件。")
        return []

    total = len(image_files)
    logging.info(f"共 {total} 张图片，开始还原旧文件名...")

    restored_files = [restore_original_name(img) for img in image_files]
    image_files = restored_files

    mode_names = {
        "ai": "AI图片错误检测",
        "photo": "漫展摄影筛选",
        "general": "通用照片质量评估",
        "portrait": "人像摄影评估",
        "landscape": "风景摄影评估",
        "document": "文档扫描清晰度",
        "art": "绘画插图质量评估",
    }
    mode_name = mode_names.get(mode, "AI图片质量评估")
    logging.info(f"检测模式: {mode_name}，开始评估...")
    if custom_prompt.strip():
        logging.info("使用自定义提示词")
    if progress_callback:
        progress_callback(0, total)

    results = []
    completed = 0
    _stop_flag.clear()
    with ThreadPoolExecutor(max_workers=config.DEFAULT_WORKERS) as executor:
        future_to_path = {executor.submit(detect_image_quality, img, mode, custom_prompt, model, max_size): img for img in image_files}
        with tqdm(total=total, desc="处理进度") as pbar:
            for future in as_completed(future_to_path):
                if _stop_flag.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    logging.info("⏹️ 用户手动停止")
                    break
                img_path = future_to_path[future]
                try:
                    score, error, reason = future.result()
                    save_txt(img_path, score, error, reason)
                    results.append((img_path, score, error, reason))
                except Exception as e:
                    logging.error(f"异常 ({img_path.name}): {e}")
                    results.append((img_path, None, None, ""))
                finally:
                    completed += 1
                    pbar.update(1)
                    if progress_callback:
                        progress_callback(completed, total)

    scored = [(p, s) for p, s, e, r in results if s is not None and e is None]
    errors = [(p, e) for p, s, e, r in results if e is not None]
    scored.sort(key=lambda x: x[1], reverse=True)

    logging.info(f"✅ 评分完成。有效评分: {len(scored)} 张，错误: {len(errors)} 张")
    for p, s in scored:
        logging.info(f"  {p.name}: {s} 分")
    for p, e in errors:
        logging.info(f"  {p.name}: ❌ {e}")

    return results


def classify_images(target_dir: str, progress_callback=None,
                    top_percent: float = None, bottom_percent: float = None,
                    min_score: float = None, max_score: float = None,
                    use_threshold: bool = False):
    """根据已有评分分类。支持两种模式：
    - use_threshold=False: 按百分比，top/bottom N%
    - use_threshold=True: 按分值，≥min_score 入 HighQuality，<max_score 入 LowQuality_Errors
    同时将 ERR 图片移入 LowQuality_Errors。
    """
    dir_path = Path(target_dir)
    if not dir_path.is_dir():
        logging.error(f"目录不存在: {target_dir}")
        return

    # 从 .txt 文件加载评分
    results = load_scores_from_txt(target_dir)
    if not results:
        logging.info("未找到评分文件（.txt），请先执行评分。")
        return

    scored = [(p, s) for p, s, e in results if s is not None and e is None]
    errors = [(p, e) for p, s, e in results if e is not None]

    if not scored:
        logging.info("无有效评分可供分类。")
        return

    scored.sort(key=lambda x: x[1], reverse=True)

    if use_threshold:
        # 按分值分类
        hi = min_score if min_score is not None else 7.0
        lo = max_score if max_score is not None else 4.0
        top_paths = set(p for p, s in scored if s >= hi)
        bottom_paths = set(p for p, s in scored if s < lo)
        logging.info(f"分类规则: 分值 ≥ {hi} → HighQuality, < {lo} → LowQuality_Errors")
    else:
        # 按百分比分类
        _top = top_percent if top_percent is not None else config.TOP_PERCENT
        _bottom = bottom_percent if bottom_percent is not None else config.BOTTOM_PERCENT
        N = len(scored)
        top_count = max(1, int(N * _top)) if N > 0 else 0
        bottom_count = max(1, int(N * _bottom)) if N > 0 else 0
        top_paths = set(p for p, _ in scored[:top_count])
        bottom_paths = set(p for p, _ in scored[-bottom_count:] if bottom_count > 0)
        logging.info(f"分类规则: 前 {_top*100:.0f}% → HighQuality, 后 {_bottom*100:.0f}% → LowQuality_Errors")

    error_paths = set(p for p, _ in errors)
    bottom_paths -= top_paths
    # ERR 图片也移入低分目录
    bottom_paths |= error_paths

    high_dir = dir_path / HIGH_QUALITY_FOLDER
    low_dir = dir_path / LOW_QUALITY_ERRORS_FOLDER

    if progress_callback:
        progress_callback(0, 1)

    moved_high = sum(move_with_txt(p, high_dir) for p in top_paths)
    moved_low = sum(move_with_txt(p, low_dir) for p in bottom_paths)

    if progress_callback:
        progress_callback(1, 1)

    logging.info(f"✅ 分类完成。HighQuality: {moved_high} 张，LowQuality_Errors: {moved_low} 张")
    if len(scored) - moved_high - moved_low > 0:
        logging.info(f"其余 {len(scored) - moved_high - moved_low} 张保留在原目录。")


# 保持旧函数兼容
def process_and_classify(target_dir: str, mode: str = "ai", progress_callback=None,
                         top_percent: float = None, bottom_percent: float = None,
                         custom_prompt: str = "", model: str = None):
    """兼容旧接口：评分 + 按百分比分类"""
    score_images(target_dir, mode, progress_callback, custom_prompt, model)
    classify_images(target_dir, progress_callback, top_percent, bottom_percent)