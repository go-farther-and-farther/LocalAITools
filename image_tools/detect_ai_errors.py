import re
import sys
import io as io_module
import base64
import time
import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple
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

# ================= 模型（每次创建新实例，确保读取最新的 ENABLE_THINKING 配置） =================
def _get_llm():
    return ChatOpenAI(
        model=config.VISION_MODEL,
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

PHOTOGRAPHY_PROMPT = """你是一个专业的漫展/Cosplay 摄影选片助手。请综合以下五个维度，对照片进行 0.0~10.0 的评分（精确一位小数），并给出简要点评。

维度：
1. 对焦与清晰度：主体（人物面部/眼睛）是否合焦锐利，有无跑焦、手抖模糊、运动模糊。这是最重要的维度。
2. 曝光：是否曝光正常，有无大面积过曝（死白、高光溢出）或欠曝（死黑、暗部无细节）。
3. 构图与取景：构图是否合理，主体位置、裁剪、背景处理是否恰当，有无杂物干扰。注意：竖幅人像是漫展摄影的标准构图，不是方向错误。
4. 色彩与白平衡：肤色是否自然，有无偏色、诡异色调。
5. 姿态与表情：Coser 的姿态是否自然好看，表情是否到位，有无闭眼、歪嘴等废片表情。

重要提醒：照片的当前显示方向即为正确方向，不需要对方向提出质疑。竖幅竖向照片是完全正常的。

评分参考（连续区间，无断档）：
- 9.0-10.0：影楼级佳作，对焦精准、曝光完美、构图优秀、表情到位，可直接出片。
- 7.0-8.9：整体良好，仅细微不足（如背景略乱、肤色轻微偏色），值得保留。
- 5.0-6.9：中等，有较明显的问题（如稍软、轻微过曝、构图一般），可选择性保留。
- 3.0-4.9：较差，多个维度有明显缺陷（如明显失焦、严重过曝/欠曝），一般不建议保留。
- 0.0-2.9：废片，存在严重问题（如完全模糊、大面积死白死黑），应直接淘汰。

输出格式：
- 如果照片存在严重拍摄缺陷（如完全跑焦模糊、严重过曝大片死白、严重欠曝几乎全黑、剧烈晃动），输出 "ERR:<错误简述>"，错误简述不超过10个中文字。
- 否则输出 "OK:<分数> <点评>"，点评用中文概括主要优缺点，50-100字。

不要输出其他任何内容。"""

# 默认使用 AI 错误检测模式
SYSTEM_PROMPT = AI_ERROR_PROMPT

MODE_PROMPTS = {
    "ai": AI_ERROR_PROMPT,
    "photo": PHOTOGRAPHY_PROMPT,
}

def _get_prompt(mode: str) -> str:
    return MODE_PROMPTS.get(mode, AI_ERROR_PROMPT)


def get_mime_type(suffix: str) -> str:
    mime_map = {
        '.jpg': 'jpeg', '.jpeg': 'jpeg', '.png': 'png',
        '.webp': 'webp', '.bmp': 'bmp', '.tiff': 'tiff'
    }
    return mime_map.get(suffix.lower(), 'jpeg')


def encode_image(image_path: Path) -> str:
    from PIL import Image, ImageOps
    img = Image.open(image_path)
    # 应用 EXIF 方向信息，防止 AI 看到旋转后图片而误判
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
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


def detect_image_quality(image_path: Path, mode: str = "ai", custom_prompt: str = "") -> Tuple[Optional[float], Optional[str], str]:
    prompt = custom_prompt.strip() if custom_prompt.strip() else _get_prompt(mode)
    for attempt in range(1, config.RETRY_TIMES + 1):
        try:
            base64_img = encode_image(image_path)
            message = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{get_mime_type(image_path.suffix)};base64,{base64_img}"
                    }
                }
            ])
            response = _get_llm().invoke([message])
            result = response.content.strip()

            if result.startswith("OK:"):
                parts = result[3:].strip().split(maxsplit=1)
                if len(parts) >= 1:
                    try:
                        score = float(parts[0])
                        reason = parts[1] if len(parts) > 1 else ""
                        if 0.0 <= score <= 10.0:
                            return round(score, 1), None, reason
                    except ValueError:
                        pass
            elif result.startswith("ERR:"):
                error = result[4:].strip()
                if error:
                    return None, error, ""

            # 兼容纯数字
            try:
                score = float(result)
                if 0.0 <= score <= 10.0:
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


def process_and_classify(target_dir: str, mode: str = "ai", progress_callback=None,
                         top_percent: float = None, bottom_percent: float = None,
                         custom_prompt: str = ""):
    dir_path = Path(target_dir)
    if not dir_path.is_dir():
        logging.error(f"目录不存在: {target_dir}")
        return

    image_files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_files:
        logging.info(f"未找到支持的图片文件。")
        return

    _top = top_percent if top_percent is not None else config.TOP_PERCENT
    _bottom = bottom_percent if bottom_percent is not None else config.BOTTOM_PERCENT

    total = len(image_files)
    logging.info(f"共 {total} 张图片，开始还原旧文件名...")

    restored_files = [restore_original_name(img) for img in image_files]
    image_files = restored_files

    mode_name = "漫展摄影筛选" if mode == "photo" else "AI图片质量评估"
    logging.info(f"检测模式: {mode_name}，开始评估...")
    if custom_prompt.strip():
        logging.info("使用自定义提示词")
    if progress_callback:
        progress_callback(0, total)
    results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=config.DEFAULT_WORKERS) as executor:
        future_to_path = {executor.submit(detect_image_quality, img, mode, custom_prompt): img for img in image_files}
        with tqdm(total=total, desc="处理进度") as pbar:
            for future in as_completed(future_to_path):
                img_path = future_to_path[future]
                try:
                    score, error, reason = future.result()
                    save_txt(img_path, score, error, reason)
                    results.append((img_path, score, error))
                except Exception as e:
                    logging.error(f"异常 ({img_path.name}): {e}")
                    results.append((img_path, None, None))
                finally:
                    completed += 1
                    pbar.update(1)
                    if progress_callback:
                        progress_callback(completed, total)

    scored = [(p, s) for p, s, e in results if s is not None and e is None]
    errors = [(p, e) for p, s, e in results if e is not None]

    scored.sort(key=lambda x: x[1], reverse=True)
    N = len(scored)
    top_count = max(1, int(N * _top)) if N > 0 else 0
    bottom_count = max(1, int(N * _bottom)) if N > 0 else 0

    top_paths = set(p for p, _ in scored[:top_count])
    bottom_paths = set(p for p, _ in scored[-bottom_count:] if bottom_count > 0)
    error_paths = set(p for p, _ in errors)
    bottom_paths -= top_paths

    high_dir = dir_path / HIGH_QUALITY_FOLDER
    low_dir = dir_path / LOW_QUALITY_ERRORS_FOLDER

    moved_high = sum(move_with_txt(p, high_dir) for p in top_paths)
    moved_low = sum(move_with_txt(p, low_dir) for p in bottom_paths.union(error_paths))

    logging.info(f"✅ 完成。HighQuality: {moved_high} 张，LowQuality_Errors: {moved_low} 张")
    if N - moved_high - moved_low > 0:
        logging.info(f"其余 {N - moved_high - moved_low} 张保留在原目录。")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    process_and_classify(target)