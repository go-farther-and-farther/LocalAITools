"""Image tool service functions -- pure Python, no Gradio dependency."""
import io
import json
import logging
import os
import shutil
import sys
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger("LocalAITools")

_DRY_RUN_RENAME_FILE = config.ROOT_DIR / "data" / "dry_run_rename.json"
_DRY_RUN_CLASSIFY_FILE = config.ROOT_DIR / "data" / "dry_run_classify.json"

_IMG_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif", ".bmp"]


def _find_images(input_dir: Path, include_subfolders: bool = False, max_depth: int = 2) -> list[Path]:
    """Find image files in directory. If include_subfolders, traverse up to max_depth sub-levels."""
    images = []
    for ext in _IMG_EXTS:
        images.extend(input_dir.glob(f"*{ext}"))
        images.extend(input_dir.glob(f"*{ext.upper()}"))

    if include_subfolders:
        _traverse = [input_dir]
        for depth in range(max_depth):
            next_level = []
            for d in _traverse:
                for sub in sorted(d.iterdir()):
                    if sub.is_dir():
                        next_level.append(sub)
                        for ext in _IMG_EXTS:
                            images.extend(sub.glob(f"*{ext}"))
                            images.extend(sub.glob(f"*{ext.upper()}"))
            _traverse = next_level
            if not _traverse:
                break

    return sorted(set(images))


def rename_images(
    input_dir: str,
    model: str = "",
    workers: int = 4,
    dry_run: bool = False,
    keep_original: bool = False,
    custom_prompt: str = "",
    context_count: int = 5,
    max_size: int = None,
    thinking: bool = True,
    progress_callback: Callable[[int, int], None] = None,
    rename_mode: str = "general",
    include_subfolders: bool = False,
) -> str:
    """Rename images with AI-generated descriptions. Returns log text.

    The caller (UI layer) is responsible for calling ``_apply_provider()``
    and setting ``os.environ["ENABLE_THINKING"]`` before invoking this.
    """
    os.environ["ENABLE_THINKING"] = "true" if thinking else "false"
    logger.info(f"[图片重命名] 目录={input_dir} 模型={model} 线程={workers} 保留原名={keep_original} max_size={max_size} 子文件夹={include_subfolders}")
    from image_tools.rename_images import process_one_image, get_shared_llm, _stop_flag

    input_path = Path(input_dir)
    if not input_path.is_dir():
        return "❌ 请输入有效的文件夹路径"

    images = _find_images(input_path, include_subfolders)

    if not images:
        return "📂 未找到图片文件"

    get_shared_llm(model or config.RENAME_MODEL)
    recent_history = deque(maxlen=max(1, context_count))
    effective_max_size = max_size if max_size and max_size > 0 else None
    results = []
    total = len(images)
    completed = 0

    structured_pairs = []  # [{old_name, new_stem, rel_path}] for dry-run save

    _stop_flag.clear()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_one_image, img, model or config.RENAME_MODEL,
                dry_run, recent_history, keep_original, rename_mode,
                effective_max_size, custom_prompt,
            ): img
            for img in images
        }
        for future in as_completed(futures):
            if _stop_flag.is_set():
                executor.shutdown(wait=False, cancel_futures=True)
                results.append("⏹️ 已请求停止")
                break
            img = futures[future]
            rel_path = img.relative_to(input_path) if img.is_relative_to(input_path) else img.name
            try:
                old_name, new_phrase = future.result()
                display_name = str(rel_path) if include_subfolders else old_name
                results.append(f"{display_name} → {new_phrase}")
                if dry_run and "生成失败" not in new_phrase and "错误" not in new_phrase:
                    structured_pairs.append({
                        "old_name": old_name,
                        "new_stem": new_phrase,
                        "rel_path": str(rel_path),
                    })
            except Exception as e:
                results.append(f"❌ {display_name}: {e}")
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    if dry_run and structured_pairs:
        try:
            _DRY_RUN_RENAME_FILE.parent.mkdir(parents=True, exist_ok=True)
            _DRY_RUN_RENAME_FILE.write_text(json.dumps({
                "input_dir": str(input_dir),
                "keep_original": keep_original,
                "include_subfolders": include_subfolders,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "count": len(structured_pairs),
                "results": structured_pairs,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(f"\n💾 试运行结果已保存，可点击「应用试运行结果」直接执行")
        except Exception as e:
            results.append(f"\n⚠️ 保存试运行结果失败: {e}")

    if _stop_flag.is_set():
        return f"已停止：已完成 {completed}/{total} 张图片\n\n" + "\n".join(results)
    return f"处理完成：{total} 张图片\n\n" + "\n".join(results)


def apply_rename_results(input_dir: str = None, keep_original: bool = False) -> str:
    """Apply saved dry-run rename results. Returns log text."""
    if not _DRY_RUN_RENAME_FILE.exists():
        return "❌ 未找到试运行结果，请先执行一次试运行"

    try:
        data = json.loads(_DRY_RUN_RENAME_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        return f"❌ 读取试运行结果失败: {e}"

    saved_dir = data.get("input_dir", "")
    target_dir = Path(input_dir) if input_dir else Path(saved_dir)
    if not target_dir.is_dir():
        return f"❌ 目录不存在: {target_dir}"

    from image_tools.rename_images import safe_rename

    results_list = data.get("results", [])
    if not results_list:
        return "❌ 试运行结果为空"

    ok, fail = 0, 0
    log = []
    for item in results_list:
        old_name = item["old_name"]
        new_stem = item["new_stem"]
        # rel_path for subfolder support, fall back to old_name
        rel_path = item.get("rel_path", old_name)
        old_path = target_dir / rel_path
        if not old_path.exists():
            log.append(f"⚠️ 文件不存在: {rel_path}")
            fail += 1
            continue

        if keep_original:
            new_stem = f"{new_stem} {Path(old_name).stem}"

        if safe_rename(old_path, new_stem):
            log.append(f"✅ {rel_path} → {new_stem}{Path(old_name).suffix}")
            ok += 1
        else:
            log.append(f"❌ 重命名失败: {rel_path}")
            fail += 1

    summary = f"应用完成：成功 {ok} 张，失败 {fail} 张（共 {len(results_list)} 张）"
    if data.get("timestamp"):
        summary += f"\n试运行时间: {data['timestamp']}"
    return summary + "\n\n" + "\n".join(log)


def classify_by_work(
    input_dir: str,
    dry_run: bool = False,
    min_count: int = 3,
    extract_subfolders: bool = False,
) -> str:
    """Classify images by work title found in filenames. Returns log text."""
    logger.info(f"[作品分类] 目录={input_dir} 试运行={dry_run} 最少={min_count} 提取子文件夹={extract_subfolders}")
    from image_tools.rename_images import classify_by_work as _classify_by_work_impl

    results = _classify_by_work_impl(input_dir, dry_run, int(min_count), extract_subfolders)
    if not results:
        return "📂 未找到可分类的图片"
    prefix = "🧪 [试运行]\n" if dry_run else ""
    return prefix + "\n".join(results)


def score_images(
    input_dir: str,
    mode: str = "ai",
    custom_prompt: str = "",
    model: str = "",
    max_size: int = None,
    thinking: bool = True,
    progress_callback: Callable[[int, int], None] = None,
) -> tuple[str, list]:
    """Score images. Returns (log_text, structured_results).

    ``structured_results`` is a list of dicts with keys:
    ``path``, ``score``, ``error``, ``reason``, ``classification``.
    """
    os.environ["ENABLE_THINKING"] = "true" if thinking else "false"
    logger.info(f"[质量评分] 目录={input_dir} 模式={mode} 模型={model} max_size={max_size}")
    from image_tools.detect_ai_errors import score_images as _score_images_impl

    effective_max_size = max_size if max_size and max_size > 0 else None

    def on_progress(completed, total):
        if progress_callback:
            progress_callback(completed, total)

    # Capture logging output while also getting structured results
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        raw_results = _score_images_impl(
            input_dir, mode, on_progress, custom_prompt, model or None, effective_max_size,
        )
    except Exception as e:
        import traceback
        stream.write(f"❌ 处理出错: {e}\n")
        stream.write(traceback.format_exc())
        raw_results = []
    finally:
        root.removeHandler(handler)
    log_text = stream.getvalue() or "✅ 处理完成"

    # Convert to structured format for gallery / CSV
    structured = []
    for img_path, score, error, reason in raw_results:
        if error:
            cls = "low"
        elif score is not None:
            cls = "high" if score >= 7 else ("low" if score < 4 else "mid")
        else:
            cls = "mid"
        structured.append({
            "path": str(img_path),
            "score": score,
            "error": error,
            "reason": reason or "",
            "classification": cls,
        })
    structured.sort(key=lambda x: x["score"] if x["score"] is not None else -1, reverse=True)
    return log_text, structured


_ai_cls_stop = threading.Event()


def ai_classify_stop():
    """Request stop for AI classification."""
    _ai_cls_stop.set()


def _compute_phash(img_path: str, hash_size: int = 16) -> int:
    """Compute perceptual hash (pHash) for an image. Returns integer hash."""
    from PIL import Image as PILImage
    try:
        img = PILImage.open(img_path).convert("L").resize((hash_size, hash_size), PILImage.LANCZOS)
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = 0
        for p in pixels:
            bits = (bits << 1) | (1 if p > avg else 0)
        return bits
    except Exception:
        return -1


def _hamming_distance(h1: int, h2: int) -> int:
    """Count differing bits between two hashes."""
    if h1 < 0 or h2 < 0:
        return 999
    xor = h1 ^ h2
    count = 0
    while xor:
        count += xor & 1
        xor >>= 1
    return count


def _group_similar(images: list, threshold: int = 8) -> list[list]:
    """Group images by perceptual hash similarity.

    Returns list of groups, each group is a list of Path objects.
    Images with Hamming distance <= threshold are in the same group.
    """
    total_bits = 16 * 16  # hash_size^2
    # Compute all hashes
    hashes = []
    for img in images:
        h = _compute_phash(str(img))
        hashes.append((img, h))

    # Group by merging similar hashes
    groups = []  # list of (representative_hash, [images])
    for img, h in hashes:
        if h < 0:
            # Failed to hash, put in its own group
            groups.append((h, [img]))
            continue
        merged = False
        for i, (gh, gimgs) in enumerate(groups):
            if _hamming_distance(h, gh) <= threshold:
                gimgs.append(img)
                merged = True
                break
        if not merged:
            groups.append((h, [img]))

    return [g[1] for g in groups]


def ai_classify_images(
    input_dir: str,
    categories: str = "照片,动漫,游戏,截图,其他",
    model: str = "",
    max_size: int = None,
    dry_run: bool = False,
    thinking: bool = True,
    workers: int = 4,
    group_similar: bool = False,
    similarity_threshold: int = 20,
    max_samples_per_group: int = 1,
    progress_callback: Callable[[int, int], None] = None,
) -> str:
    """AI classify images into user-defined categories and move to subdirectories."""
    os.environ["ENABLE_THINKING"] = "true" if thinking else "false"
    logger.info(f"[AI分类] 目录={input_dir} 分类={categories} 模型={model} 并发={workers} 试运行={dry_run}")

    input_path = Path(input_dir)
    if not input_path.is_dir():
        return "❌ 请输入有效的文件夹路径"

    # Parse categories: support "名称：描述" or plain "名称"
    cat_list = []  # [(name, description)]
    for c in categories.split(","):
        c = c.strip()
        if not c:
            continue
        if "：" in c:
            name, desc = c.split("：", 1)
            cat_list.append((name.strip(), desc.strip()))
        elif ":" in c:
            name, desc = c.split(":", 1)
            cat_list.append((name.strip(), desc.strip()))
        else:
            cat_list.append((c, ""))
    if len(cat_list) < 2:
        return "❌ 请至少输入两个分类名称，用逗号分隔"

    cat_names = [c[0] for c in cat_list]

    exts = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif", ".bmp"}
    images = sorted([f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in exts])
    if not images:
        return "📂 未找到图片文件"

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from image_tools.rename_images import encode_image, _clean_model_tokens

    # Build prompt with descriptions
    cat_lines = []
    for name, desc in cat_list:
        if desc:
            cat_lines.append(f"  - {name}：{desc}")
        else:
            cat_lines.append(f"  - {name}")
    cat_desc_str = "\n".join(cat_lines)
    cat_name_str = "、".join(cat_names)

    def _classify_one(img):
        """Classify a single image. Returns (img, matched_category, error)."""
        img_base64 = encode_image(str(img), max_size=max_size)
        if img_base64 is None:
            return img, None, "读取失败"
        try:
            llm = ChatOpenAI(
                model=model or config.VISION_MODEL,
                base_url=config.OPENAI_BASE_URL,
                api_key=config.OPENAI_API_KEY,
                temperature=0.1,
                max_tokens=50,
                extra_body=config.get_llm_extra_body(thinking),
            )
            prompt = f"""请根据图片内容和文件名，将图片归入最合适的类别。

## 可选类别
{cat_desc_str}

## 文件名
{img.stem}

## 要求
1. 仔细观察图片的视觉内容（场景、主体、风格、构图）
2. 结合文件名中的信息辅助判断（文件名可能包含作品名、角色名等关键线索）
3. 选择最匹配的一个类别，仅输出类别名称（{cat_name_str}）
4. 不要输出任何解释或其他内容"""

            msg = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ])
            resp = llm.invoke([msg])
            raw = _clean_model_tokens(resp.content.strip())

            matched = None
            for name in cat_names:
                if name in raw:
                    matched = name
                    break
            if not matched:
                for name in cat_names:
                    if any(ch in raw for ch in name):
                        matched = name
                        break
            if not matched:
                matched = cat_names[-1]
            return img, matched, None
        except Exception as e:
            return img, None, str(e)
        finally:
            del img_base64

    def _classify_group(imgs):
        """Classify a group of similar images in ONE API call. Returns (matched_category, error)."""
        content_parts = [
            {"type": "text", "text": f"""请根据以下 {len(imgs)} 张图片的内容，判断它们共同属于哪个类别。
这些图片内容相似，请给出一个统一的分类结果。

## 可选类别
{cat_desc_str}

## 文件名参考
{', '.join(img.stem for img in imgs[:5])}

## 要求
1. 观察所有图片的共同特征
2. 选择最匹配的一个类别，仅输出类别名称（{cat_name_str}）
3. 不要输出任何解释或其他内容"""}
        ]
        for img in imgs:
            img_base64 = encode_image(str(img), max_size=max_size)
            if img_base64 is None:
                continue
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}})
        if len(content_parts) < 2:
            return None, "所有图片读取失败"
        try:
            llm = ChatOpenAI(
                model=model or config.VISION_MODEL,
                base_url=config.OPENAI_BASE_URL,
                api_key=config.OPENAI_API_KEY,
                temperature=0.1,
                max_tokens=50,
                extra_body=config.get_llm_extra_body(thinking),
            )
            msg = HumanMessage(content=content_parts)
            resp = llm.invoke([msg])
            raw = _clean_model_tokens(resp.content.strip())

            matched = None
            for name in cat_names:
                if name in raw:
                    matched = name
                    break
            if not matched:
                for name in cat_names:
                    if any(ch in raw for ch in name):
                        matched = name
                        break
            if not matched:
                matched = cat_names[-1]
            return matched, None
        except Exception as e:
            return None, str(e)

    total = len(images)
    completed = 0
    results = []
    structured_pairs = []  # [{name, category}] for dry-run save
    cat_counts = {name: 0 for name in cat_names}
    skipped = 0

    _ai_cls_stop.clear()

    if group_similar and len(images) > 1:
        # ---- Grouped mode: one API call per group with multiple images ----
        logger.info(f"[AI分组] 正在计算相似度，共 {len(images)} 张图片...")
        groups = _group_similar(images, threshold=similarity_threshold)
        n_sample = max(1, int(max_samples_per_group))
        logger.info(f"[AI分组] 分成 {len(groups)} 组，每组最多送 {n_sample} 张给 AI（一次调用）")
        if progress_callback:
            progress_callback(0, total)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for g in groups:
                samples = g[:n_sample]
                fut = executor.submit(_classify_group, samples)
                futures[fut] = g  # map future -> group members

            for future in as_completed(futures):
                if _ai_cls_stop.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    results.append("⏹️ 已停止")
                    break

                members = futures[future]
                matched, err = future.result()
                if err:
                    for m in members:
                        results.append(f"❌ {m.name}: {err}")
                        skipped += 1
                else:
                    for m in members:
                        if dry_run:
                            tag = f"🧪 {m.name} → {matched}"
                            if len(members) > 1 and m == members[0]:
                                tag += f" (同组 {len(members)} 张)"
                            results.append(tag)
                            cat_counts[matched] = cat_counts.get(matched, 0) + 1
                            structured_pairs.append({"name": m.name, "category": matched})
                        else:
                            dst_dir = input_path / matched
                            dst_dir.mkdir(exist_ok=True)
                            dst = dst_dir / m.name
                            counter = 1
                            while dst.exists():
                                dst = dst_dir / f"{m.stem}_{counter}{m.suffix}"
                                counter += 1
                            shutil.move(str(m), str(dst))
                            results.append(f"✅ {m.name} → {matched}/")
                            cat_counts[matched] = cat_counts.get(matched, 0) + 1
                completed += len(members)
                if progress_callback:
                    progress_callback(completed, total)
    else:
        # ---- Normal mode: one API call per image ----
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_classify_one, img): img for img in images}
            for future in as_completed(futures):
                if _ai_cls_stop.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    results.append("⏹️ 已停止")
                    break

                img, matched, err = future.result()
                if err:
                    results.append(f"❌ {img.name}: {err}")
                    skipped += 1
                elif dry_run:
                    results.append(f"🧪 {img.name} → {matched}")
                    cat_counts[matched] = cat_counts.get(matched, 0) + 1
                    structured_pairs.append({"name": img.name, "category": matched})
                else:
                    dst_dir = input_path / matched
                    dst_dir.mkdir(exist_ok=True)
                    dst = dst_dir / img.name
                    counter = 1
                    while dst.exists():
                        dst = dst_dir / f"{img.stem}_{counter}{img.suffix}"
                        counter += 1
                    shutil.move(str(img), str(dst))
                    results.append(f"✅ {img.name} → {matched}/")
                    cat_counts[matched] = cat_counts.get(matched, 0) + 1
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

    if dry_run and structured_pairs:
        try:
            _DRY_RUN_CLASSIFY_FILE.parent.mkdir(parents=True, exist_ok=True)
            _DRY_RUN_CLASSIFY_FILE.write_text(json.dumps({
                "input_dir": str(input_dir),
                "categories": categories,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "count": len(structured_pairs),
                "results": structured_pairs,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(f"\n💾 试运行结果已保存，可点击「应用试运行结果」直接执行")
        except Exception as e:
            results.append(f"\n⚠️ 保存试运行结果失败: {e}")

    prefix = "🧪 [试运行] " if dry_run else ""
    summary = [f"{prefix}分类完成：共 {total} 张图片"]
    for c, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        if cnt > 0:
            summary.append(f"  {c}: {cnt} 张")
    if skipped:
        summary.append(f"  失败: {skipped} 张")

    return "\n".join(summary) + "\n\n" + "\n".join(results)


def apply_classify_results(input_dir: str = None) -> str:
    """Apply saved dry-run classify results. Returns log text."""
    if not _DRY_RUN_CLASSIFY_FILE.exists():
        return "❌ 未找到试运行结果，请先执行一次试运行"

    try:
        data = json.loads(_DRY_RUN_CLASSIFY_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        return f"❌ 读取试运行结果失败: {e}"

    saved_dir = data.get("input_dir", "")
    target_dir = Path(input_dir) if input_dir else Path(saved_dir)
    if not target_dir.is_dir():
        return f"❌ 目录不存在: {target_dir}"

    results_list = data.get("results", [])
    if not results_list:
        return "❌ 试运行结果为空"

    ok, fail = 0, 0
    log = []
    for item in results_list:
        name = item["name"]
        category = item["category"]
        src = target_dir / name
        if not src.exists():
            log.append(f"⚠️ 文件不存在: {name}")
            fail += 1
            continue

        dst_dir = target_dir / category
        dst_dir.mkdir(exist_ok=True)
        dst = dst_dir / name
        counter = 1
        while dst.exists():
            dst = dst_dir / f"{Path(name).stem}_{counter}{Path(name).suffix}"
            counter += 1
        try:
            shutil.move(str(src), str(dst))
            log.append(f"✅ {name} → {category}/")
            ok += 1
        except Exception as e:
            log.append(f"❌ 移动失败 {name}: {e}")
            fail += 1

    summary = f"应用完成：成功 {ok} 张，失败 {fail} 张（共 {len(results_list)} 张）"
    if data.get("timestamp"):
        summary += f"\n试运行时间: {data['timestamp']}"
    return summary + "\n\n" + "\n".join(log)


def classify_images(
    input_dir: str,
    classify_method: str = "auto",
    top_percent: float = None,
    bottom_percent: float = None,
    min_score: float = None,
    max_score: float = None,
    progress_callback: Callable[[int, int], None] = None,
) -> str:
    """Classify images into quality folders based on prior scores. Returns log text."""
    logger.info(f"[质量分类] 目录={input_dir} 方法={classify_method} top={top_percent} bottom={bottom_percent}")
    from image_tools.detect_ai_errors import classify_images as _classify_images_impl

    use_threshold = (classify_method == "threshold")

    # Capture logging output
    import io
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        _classify_images_impl(
            input_dir, None,
            top_percent / 100 if top_percent else None,
            bottom_percent / 100 if bottom_percent else None,
            min_score, max_score, use_threshold,
        )
    except Exception as e:
        import traceback
        stream.write(f"❌ 处理出错: {e}\n")
        stream.write(traceback.format_exc())
    finally:
        root.removeHandler(handler)
    return stream.getvalue() or "✅ 处理完成"
