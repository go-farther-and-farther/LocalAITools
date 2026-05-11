"""Benchmark service functions -- pure Python, no Gradio dependency."""
import io
import json
import csv
import logging
import sys
import time as _time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger("LocalAITools")

# Benchmark history file
_BENCH_HISTORY_FILE = config.OUTPUT_DIR / "benchmark_history.json"
_BENCH_HISTORY_MAX = 100


# ---- History helpers (pure I/O, no Gradio) ----

def _load_bench_history() -> list[dict]:
    """Load benchmark history from JSON file."""
    if _BENCH_HISTORY_FILE.exists():
        try:
            return json.loads(_BENCH_HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _save_bench_history(entries: list[dict]) -> None:
    """Save benchmark history list to JSON file."""
    _BENCH_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _BENCH_HISTORY_FILE.write_text(
        json.dumps(entries[:_BENCH_HISTORY_MAX], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _append_bench_history(record: dict) -> None:
    """Append a single benchmark record to history."""
    entries = _load_bench_history()
    entries.insert(0, record)
    _save_bench_history(entries)


def get_history_rows(limit: int = 20) -> list[list]:
    """Load recent benchmark history for display. Returns list of rows."""
    entries = _load_bench_history()[:limit]
    rows = []
    for e in entries:
        rows.append([
            e.get("timestamp", ""),
            e.get("model", ""),
            e.get("tokens_per_sec", ""),
            e.get("total_tokens", ""),
            e.get("duration", ""),
            e.get("prompt_length", ""),
        ])
    return rows


def export_bench_csv() -> str:
    """Export benchmark history to CSV and return the file path.

    Returns an empty string if there is no history.
    """
    entries = _load_bench_history()
    if not entries:
        return ""
    csv_path = config.OUTPUT_DIR / "benchmark_history.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["时间", "模型", "tokens/s", "总token数", "耗时(秒)", "Prompt长度"])
        for e in entries:
            writer.writerow([
                e.get("timestamp", ""),
                e.get("model", ""),
                e.get("tokens_per_sec", ""),
                e.get("total_tokens", ""),
                e.get("duration", ""),
                e.get("prompt_length", ""),
            ])
    return str(csv_path)


# ---- Core benchmark ----

def run_benchmark(
    url: str = "",
    model: str = "",
    api_key: str = "",
    concurrency: int = 1,
    timeout: int = 300,
    output_tokens: int = 1024,
    lengths_str: str = "128,512,1024,2048",
    progress_callback: Callable[[int, int], None] = None,
) -> str:
    """Run LLM benchmark. Returns result text.

    This is a **blocking** call. The caller should run it in a background
    thread if non-blocking behaviour is needed.
    """
    logger.info(f"[LLM压测] URL={url} 模型={model} 并发={concurrency} 长度={lengths_str}")
    from benchmarks.speedtest import run_benchmark as _run_benchmark_impl

    if lengths_str.strip():
        lengths = [int(x.strip()) for x in lengths_str.split(",")]
    else:
        lengths = [512, 1024, 2048, 4096]

    save_json = str(config.OUTPUT_DIR / "benchmarks" / "benchmark_results.json")
    save_plot = str(config.OUTPUT_DIR / "benchmarks" / "throughput_chart.png")

    def _progress_cb(done, total):
        if progress_callback:
            progress_callback(done, total)

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            _run_benchmark_impl(
                token_lengths=lengths, concurrency=concurrency,
                base_url=url, model=model, api_key=api_key,
                timeout=timeout, output_tokens=output_tokens,
                save_json=save_json, save_plot=save_plot,
                progress_callback=_progress_cb,
            )
    except Exception as e:
        buf.write(f"\n❌ 压测出错: {e}\n")

    text_output = buf.getvalue()

    # Auto-save benchmark results to history
    try:
        results_path = Path(save_json)
        if results_path.exists():
            raw = json.loads(results_path.read_text(encoding="utf-8"))
            cfg = raw.get("config", {})
            stats = raw.get("summary", [])
            output_vals = [s["output"] for s in stats if s.get("output") is not None]
            total_tokens = sum(
                r.get("actual_output_tokens", 0)
                for entry in raw.get("details", [])
                for r in entry.get("results", [])
                if r.get("success")
            )
            avg_tps = round(sum(output_vals) / len(output_vals), 2) if output_vals else 0
            prompt_lengths = ", ".join(str(s["length"]) for s in stats)
            _append_bench_history({
                "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": cfg.get("model", model or ""),
                "tokens_per_sec": avg_tps,
                "total_tokens": total_tokens,
                "duration": round(cfg.get("total_duration", 0), 1),
                "prompt_length": prompt_lengths,
            })
    except Exception as e:
        logger.warning(f"保存压测历史失败: {e}")

    return text_output


def get_plot_path() -> Optional[str]:
    """Return the path to the benchmark plot image, or None if it doesn't exist."""
    plot_path = config.OUTPUT_DIR / "benchmarks" / "throughput_chart.png"
    return str(plot_path) if plot_path.exists() else None
