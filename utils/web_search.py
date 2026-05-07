"""Web search utilities."""
import logging
logger = logging.getLogger("LocalAITools")

def _do_web_search(query, max_results=5):
    """联网搜索，优先用 Bing 中国版，失败回退到 DuckDuckGo"""
    results = _search_bing_cn(query, max_results)
    if results is not None:
        return results
    results = _search_ddg(query, max_results)
    if results is not None:
        return results
    return None

def _search_bing_cn(query, max_results=5):
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

def _search_ddg(query, max_results=5):
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

