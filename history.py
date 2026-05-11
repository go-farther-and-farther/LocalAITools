"""
处理历史记录模块 — 每次任务完成后自动记录
"""
import json
import time
import threading
from pathlib import Path

HISTORY_FILE = Path(__file__).parent / "outputs" / "history.json"
MAX_ENTRIES = 100
_lock = threading.Lock()


def _load() -> list[dict]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def add_entry(tool: str, input_path: str, summary: str, source: str = "web") -> None:
    """添加一条历史记录"""
    with _lock:
        entries = _load()
        entries.insert(0, {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tool": tool,
            "input": str(input_path),
            "summary": summary,
            "source": source,
        })
        # 保留最近 MAX_ENTRIES 条
        entries = entries[:MAX_ENTRIES]
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def get_recent(count: int = 20) -> list[dict]:
    """获取最近 N 条记录"""
    return _load()[:count]
