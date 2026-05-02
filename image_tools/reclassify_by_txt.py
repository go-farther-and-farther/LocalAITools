import re
import sys
import shutil
import logging
from pathlib import Path
from typing import Optional, Tuple, List

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

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


def parse_txt(txt_path: Path) -> Tuple[Optional[float], Optional[str]]:
    """
    解析 txt 文件，返回 (分数, 错误描述)。
    如果正常评分：返回 (score, None)
    如果错误：返回 (None, error_str)
    如果解析失败或不含有效信息：返回 (None, None)
    """
    try:
        content = txt_path.read_text(encoding='utf-8')
    except Exception as e:
        logging.warning(f"无法读取 txt: {txt_path} ({e})")
        return None, None

    # 检查是否为错误描述
    if content.startswith("错误:"):
        error = content[3:].strip()
        return None, error if error else "未知错误"

    # 尝试匹配 “评分: X.X 分” 或 “评分: X.X”
    match = re.search(r'评分[：:]\s*(\d+\.?\d*)\s*分?', content)
    if match:
        try:
            score = float(match.group(1))
            if 0.0 <= score <= 10.0:
                return round(score, 1), None
        except ValueError:
            pass

    # 备用：直接查找浮点数
    match = re.search(r'(\d+\.\d+)', content)
    if match:
        try:
            score = float(match.group(1))
            if 0.0 <= score <= 10.0:
                return round(score, 1), None
        except ValueError:
            pass

    return None, None


def safe_move(src: Path, dst_dir: Path) -> bool:
    """移动文件到目标目录，重名自动加序号。"""
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
    """移动图片及其同名 txt（如果存在）。"""
    success = safe_move(src, dst_dir)
    if success:
        txt_src = src.with_suffix('.txt')
        if txt_src.exists():
            safe_move(txt_src, dst_dir)
    return success


def process_directory(target_dir: str):
    root = Path(target_dir)
    if not root.is_dir():
        logging.error(f"目录不存在: {target_dir}")
        return

    # 收集所有图片文件
    image_files = [p for p in root.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_files:
        logging.info("未找到支持的图片文件。")
        return

    logging.info(f"找到 {len(image_files)} 张图片，开始解析评分...")

    scored = []      # (path, score)
    errors = []      # (path, error_desc)
    unparsed = []    # 没有有效txt或无法解析

    for img in tqdm(image_files, desc="解析 txt"):
        txt = img.with_suffix('.txt')
        if not txt.exists():
            unparsed.append(img)
            continue

        score, error = parse_txt(txt)
        if score is not None:
            scored.append((img, score))
        elif error is not None:
            errors.append((img, error))
        else:
            unparsed.append(img)

    if unparsed:
        logging.warning(f"有 {len(unparsed)} 张图片未能解析评分/错误，保留在原地。")

    # 按分数降序排序
    scored.sort(key=lambda x: x[1], reverse=True)
    N = len(scored)
    top_count = max(1, int(N * config.TOP_PERCENT)) if N > 0 else 0
    bottom_count = max(1, int(N * config.BOTTOM_PERCENT)) if N > 0 else 0

    top_paths = set(p for p, _ in scored[:top_count])
    bottom_paths = set(p for p, _ in scored[-bottom_count:] if bottom_count > 0)
    error_paths = set(p for p, _ in errors)

    # 避免同一张图同时出现在 top 和 bottom（理论上不可能，但安全处理）
    bottom_paths -= top_paths

    high_dir = root / HIGH_QUALITY_FOLDER
    low_dir = root / LOW_QUALITY_ERRORS_FOLDER

    logging.info(f"开始分类移动... (前{config.TOP_PERCENT*100:.0f}%: {len(top_paths)} 张, 后{config.BOTTOM_PERCENT*100:.0f}%+错误: {len(bottom_paths | error_paths)} 张)")

    moved_high = 0
    moved_low = 0

    for p in tqdm(top_paths, desc="移动高质量"):
        if move_with_txt(p, high_dir):
            moved_high += 1

    for p in tqdm(bottom_paths.union(error_paths), desc="移动低质量/错误"):
        if move_with_txt(p, low_dir):
            moved_low += 1

    remaining = N - moved_high - moved_low
    logging.info(f"✅ 完成！高质量: {moved_high} 张, 低质量/错误: {moved_low} 张, 保留原目录: {remaining} 张")
    if unparsed:
        logging.info(f"未处理(无有效txt): {len(unparsed)} 张，已保留在原位置。")


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    process_directory(target)