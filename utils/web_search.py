"""Web search utilities."""
import logging
import threading
import time

logger = logging.getLogger("LocalAITools")

# ---------------------------------------------------------------------------
# Search result cache (thread-safe, TTL-based, LRU eviction)
# ---------------------------------------------------------------------------
_SEARCH_CACHE: dict[str, tuple[str | None, float]] = {}   # key -> (result, timestamp)
_CACHE_LOCK = threading.Lock()
_CACHE_MAX_SIZE = 100
_CACHE_TTL = 600  # seconds (10 minutes)


def _cache_key(query: str, max_results: int) -> str:
    """Build a deterministic cache key from search parameters."""
    return f"{query.strip().lower()}||{max_results}"


def _cache_get(key: str) -> str | None | object:
    """Return cached result if still valid, else _MISS sentinel."""
    with _CACHE_LOCK:
        entry = _SEARCH_CACHE.get(key)
        if entry is None:
            return _MISS
        result, ts = entry
        if time.time() - ts > _CACHE_TTL:
            # Expired — remove it
            del _SEARCH_CACHE[key]
            return _MISS
        return result


_MISS = object()  # sentinel: not found / expired


def _cache_put(key: str, result: str | None) -> None:
    """Insert a result into the cache, evicting the oldest entry if full."""
    with _CACHE_LOCK:
        if key in _SEARCH_CACHE:
            _SEARCH_CACHE[key] = (result, time.time())
            return
        # Evict oldest entry when at capacity (LRU-style)
        if len(_SEARCH_CACHE) >= _CACHE_MAX_SIZE:
            oldest_key = min(_SEARCH_CACHE, key=lambda k: _SEARCH_CACHE[k][1])
            del _SEARCH_CACHE[oldest_key]
        _SEARCH_CACHE[key] = (result, time.time())


def clear_search_cache() -> None:
    """Manually clear the entire search cache."""
    with _CACHE_LOCK:
        _SEARCH_CACHE.clear()
    logger.debug("Search cache cleared")


def _do_web_search(query: str, max_results: int = 5) -> str | None:
    """联网搜索，优先用 Bing 中国版，失败回退到 DuckDuckGo。

    Results are cached in-memory for 10 minutes.
    """
    key = _cache_key(query, max_results)

    cached = _cache_get(key)
    if cached is not _MISS:
        logger.debug("Search cache HIT for query: %s", query)
        return cached

    logger.debug("Search cache MISS for query: %s", query)

    results = _search_bing_cn(query, max_results)
    if results is not None:
        _cache_put(key, results)
        return results
    results = _search_ddg(query, max_results)
    if results is not None:
        _cache_put(key, results)
        return results

    # Cache None results too so we don't re-hit failing backends
    _cache_put(key, None)
    return None

def _search_bing_cn(query: str, max_results: int = 5) -> str | None:
    """Bing 中国版搜索 (cn.bing.com)"""
    try:
        import requests
        from lxml import html
    except ImportError:
        return None

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        resp = requests.get("https://cn.bing.com/search",
                            params={"q": query}, headers=headers, timeout=10)
        resp.encoding = "utf-8"
        tree = html.fromstring(resp.text)
        items = tree.xpath('//li[contains(@class, "b_algo")]')
        results = []
        for r in items[:max_results]:
            title_els = r.xpath(".//h2//text()") or r.xpath(".//a//text()")
            title = " ".join(title_els).strip() if title_els else ""
            snippet_els = r.xpath(
                './/p[contains(@class, "b_lineclamp") or contains(@class, "b_snippet")]//text()'
            )
            snippet = " ".join(snippet_els).strip() if snippet_els else ""
            if not snippet:
                snippet_els = r.xpath('.//div[contains(@class, "b_caption")]//p//text()')
                snippet = " ".join(snippet_els).strip() if snippet_els else ""
            link_els = r.xpath(".//h2//a/@href")
            link = link_els[0] if link_els else ""
            if title:
                results.append(f"【{title}】\n{snippet}\n来源: {link}")
        return "\n\n".join(results) if results else None
    except Exception as e:
        logger.warning(f"Bing 搜索失败: {e}")
        return None

def _search_ddg(query: str, max_results: int = 5) -> str | None:
    """DuckDuckGo 搜索（通过 ddgs 库）"""
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"【{r.get('title', '')}】\n{r.get('body', '')}\n来源: {r.get('href', '')}")
        return "\n\n".join(results) if results else None
    except ImportError:
        logger.warning("ddgs 未安装，DuckDuckGo 搜索不可用")
        return None
    except Exception as e:
        logger.warning(f"DuckDuckGo 搜索失败: {e}")
        return None

