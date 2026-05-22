"""
Web search: DuckDuckGo results -> parallel HTTP fetch -> markdownify -> LLM synthesis.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from ddgs import DDGS
from dotenv import load_dotenv
from markdownify import markdownify as html_to_markdown

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")

WEB_SEARCH_MAX_RESULTS = max(1, min(5, int(os.getenv("WEB_SEARCH_MAX_RESULTS", "2"))))
WEB_SEARCH_DDGS_TIMEOUT = float(os.getenv("WEB_SEARCH_DDGS_TIMEOUT", "10"))
WEB_SEARCH_HTTP_TIMEOUT = float(os.getenv("WEB_SEARCH_HTTP_TIMEOUT", "12"))
WEB_SEARCH_MAX_RESPONSE_BYTES = int(os.getenv("WEB_SEARCH_MAX_RESPONSE_BYTES", "2000000"))
WEB_SEARCH_MAX_EXTRACT_CHARS = int(os.getenv("WEB_SEARCH_MAX_EXTRACT_CHARS", "8000"))
WEB_SEARCH_MAX_TOTAL_INPUT_CHARS = int(os.getenv("WEB_SEARCH_MAX_TOTAL_INPUT_CHARS", "20000"))
WEB_SEARCH_SUMMARY_MODEL = os.getenv("WEB_SEARCH_SUMMARY_MODEL", "gpt-4o-mini")
WEB_SEARCH_SUMMARY_MAX_TOKENS = int(os.getenv("WEB_SEARCH_SUMMARY_MAX_TOKENS", "1400"))
WEB_SEARCH_SUMMARY_TEMPERATURE = float(os.getenv("WEB_SEARCH_SUMMARY_TEMPERATURE", "0.35"))

# Chrome on Windows — many sites block non-browser or bot-like User-Agent strings.
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
WEB_SEARCH_USER_AGENT = os.getenv("WEB_SEARCH_USER_AGENT", "").strip() or _DEFAULT_USER_AGENT

# Referer sent on page fetches (mimics traffic from a search engine).
WEB_SEARCH_REFERER = os.getenv("WEB_SEARCH_REFERER", "https://www.google.com/").strip()
# google = fixed referer above; origin = target site's origin (/{scheme}://{host}/).
WEB_SEARCH_REFERER_MODE = os.getenv("WEB_SEARCH_REFERER_MODE", "google").strip().lower()

WEB_SEARCH_LOG_PREVIEW_CHARS = max(
    0, int(os.getenv("WEB_SEARCH_LOG_PREVIEW_CHARS", "600"))
)

_STRIP_TAGS = ["script", "style", "noscript", "svg", "iframe", "head", "meta", "link"]

_BOT_BLOCK_MARKERS = (
    "cf-browser-verification",
    "challenge-platform",
    "captcha-delivery",
    "access denied",
    "please enable javascript",
    "unusual traffic",
    "verify you are human",
    "bot detection",
)


def _log_summary_preview(summary: str | None) -> None:
    if not summary:
        print("[web_search] summary: (none)")
        return
    n = len(summary)
    if WEB_SEARCH_LOG_PREVIEW_CHARS <= 0:
        print(f"[web_search] summary length={n} chars (preview disabled)")
        return
    preview = summary[: WEB_SEARCH_LOG_PREVIEW_CHARS]
    if len(summary) > WEB_SEARCH_LOG_PREVIEW_CHARS:
        preview += "…"
    print(f"[web_search] summary ({n} chars):\n{preview}")


def _referer_for_url(url: str) -> str | None:
    parsed = urlparse(url)
    if WEB_SEARCH_REFERER_MODE == "origin":
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}/"
        return None
    if WEB_SEARCH_REFERER_MODE == "google" and WEB_SEARCH_REFERER:
        return WEB_SEARCH_REFERER
    if WEB_SEARCH_REFERER:
        return WEB_SEARCH_REFERER
    return None


def _fetch_headers(url: str) -> dict[str, str]:
    """Browser-like headers to reduce bot blocks on page fetches."""
    referer = _referer_for_url(url)
    headers: dict[str, str] = {
        "User-Agent": WEB_SEARCH_USER_AGENT,
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "max-age=0",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site" if referer else "none",
        "Sec-Fetch-User": "?1",
        "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "DNT": "1",
    }
    if referer:
        headers["Referer"] = referer
    if PUBLIC_BASE_URL.strip():
        headers["From"] = PUBLIC_BASE_URL.strip()
    return headers


def _looks_like_bot_block(html: str) -> bool:
    sample = html[:12000].lower()
    return any(marker in sample for marker in _BOT_BLOCK_MARKERS)


def _extract_main_text(html: str) -> str | None:
    if _looks_like_bot_block(html):
        return None
    try:
        text = html_to_markdown(
            html,
            heading_style="ATX",
            bullets="-",
            strip=_STRIP_TAGS,
        )
    except Exception:
        return None
    if not text or not text.strip():
        return None
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text


def _cap_text(s: str, max_chars: int) -> str:
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + "\n\n[truncated…]"


def _search_ddgs(query: str, max_results: int) -> list[dict[str, str]]:
    with DDGS() as ddgs:
        rows = list(ddgs.text(query, max_results=max_results))
    items: list[dict[str, str]] = []
    for row in rows[:max_results]:
        items.append(
            {
                "title": (row.get("title") or "").strip(),
                "snippet": (row.get("body") or "").strip(),
                "url": (row.get("href") or "").strip(),
            }
        )
    return items


async def _fetch_one(
    client: httpx.AsyncClient,
    item: dict[str, str],
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    url = item.get("url") or ""
    out: dict[str, Any] = {
        "title": item.get("title", ""),
        "url": url,
        "snippet": item.get("snippet", ""),
        "extracted_text": None,
        "fetch_error": None,
    }
    if not url or urlparse(url).scheme not in ("http", "https"):
        out["fetch_error"] = "invalid_or_missing_url"
        return out

    async with sem:
        try:
            resp = await client.get(
                url,
                headers=_fetch_headers(url),
                follow_redirects=True,
                timeout=WEB_SEARCH_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            if len(resp.content) > WEB_SEARCH_MAX_RESPONSE_BYTES:
                out["fetch_error"] = "response_too_large"
                return out
            ctype = (resp.headers.get("content-type") or "").lower()
            if "html" not in ctype and "xml" not in ctype:
                out["fetch_error"] = "non_html_content"
                return out
            html = resp.text
        except Exception as exc:
            out["fetch_error"] = str(exc)
            return out

    if _looks_like_bot_block(html):
        out["fetch_error"] = "bot_block_or_challenge_page"
        return out

    extracted = await asyncio.to_thread(_extract_main_text, html)
    if extracted:
        out["extracted_text"] = _cap_text(extracted, WEB_SEARCH_MAX_EXTRACT_CHARS)
    else:
        out["fetch_error"] = out.get("fetch_error") or "extraction_empty"

    return out


async def _summarize_with_openai(query: str, numbered_sources: list[tuple[int, dict[str, Any]]]) -> tuple[str | None, str | None]:
    """Returns (summary, error_message)."""
    if not OPENAI_API_KEY:
        return None, "OPENAI_API_KEY not set; skipping summarization."

    blocks: list[str] = []
    total_chars = 0
    for idx, src in numbered_sources:
        title = src.get("title") or ""
        url = src.get("url") or ""
        snippet = src.get("snippet") or ""
        ext = src.get("extracted_text") or ""
        part = (
            f"[{idx}] {title}\nURL: {url}\n"
            f"Search snippet: {snippet}\n"
            f"Extracted page content (markdown):\n{ext if ext else '(none — rely on snippet only)'}\n"
        )
        if total_chars + len(part) > WEB_SEARCH_MAX_TOTAL_INPUT_CHARS:
            part = _cap_text(part, WEB_SEARCH_MAX_TOTAL_INPUT_CHARS - total_chars)
        blocks.append(part)
        total_chars += len(part)
        if total_chars >= WEB_SEARCH_MAX_TOTAL_INPUT_CHARS:
            break

    combined = "\n---\n".join(blocks)
    system = (
        "You synthesize web page extracts for a voice assistant answering a caller. "
        "Be precise and preserve important details."
    )
    user = (
        f'User search query: "{query}"\n\n'
        "Below are numbered sources. Write a synthesis that helps answer the query.\n\n"
        "Rules:\n"
        "- Preserve concrete facts: numbers, dates, names, times, statistics, units, and hedges "
        '("about", "approximately", "reportedly", "as of …").\n'
        "- Extracts are full-page markdown; ignore nav/footer boilerplate but keep tables, "
        "lists, and showtime-style data when present.\n"
        "- If sources conflict, state that briefly.\n"
        "- Do not invent facts. If the extracts are insufficient, say what is missing.\n"
        "- Use short paragraphs or labeled bullets. Prefer clarity over brevity when facts matter.\n"
        "- Reference sources as [1], [2] when attributing specific claims.\n\n"
        f"{combined}"
    )

    payload = {
        "model": WEB_SEARCH_SUMMARY_MODEL,
        "temperature": WEB_SEARCH_SUMMARY_TEMPERATURE,
        "max_tokens": WEB_SEARCH_SUMMARY_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    try:
        async with httpx.AsyncClient(
            timeout=60.0,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
        ) as api_client:
            r = await api_client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
        choices = data.get("choices") or []
        if not choices:
            return None, "OpenAI returned no choices."
        msg = (choices[0].get("message") or {}).get("content") or ""
        msg = msg.strip()
        if not msg:
            return None, "Empty summary from model."
        return msg, None
    except Exception as exc:
        return None, str(exc)


def _fallback_digest(numbered: list[tuple[int, dict[str, Any]]]) -> str:
    """Unsummarized cap when the LLM step fails; keeps snippets + short extracts."""
    parts: list[str] = []
    budget = WEB_SEARCH_MAX_TOTAL_INPUT_CHARS
    for idx, src in numbered:
        title = src.get("title") or ""
        url = src.get("url") or ""
        snip = src.get("snippet") or ""
        ext = src.get("extracted_text") or ""
        chunk = f"[{idx}] {title}\n{url}\nSnippet: {snip}"
        if ext:
            chunk += f"\nExtract (truncated):\n{_cap_text(ext, min(4000, budget // max(1, len(numbered))))}"
        chunk += "\n"
        if len(chunk) > budget:
            chunk = _cap_text(chunk, budget)
        parts.append(chunk)
        budget -= len(chunk)
        if budget <= 0:
            break
    return "\n---\n".join(parts).strip()


async def run_web_search(query: str) -> dict[str, Any]:
    """
    DDGS → fetch + markdownify → optional OpenAI synthesis into `summary`.
    """
    print(f"[web_search] start query={query!r} max_results={WEB_SEARCH_MAX_RESULTS}")
    try:
        rows = await asyncio.wait_for(
            asyncio.to_thread(_search_ddgs, query, WEB_SEARCH_MAX_RESULTS),
            timeout=WEB_SEARCH_DDGS_TIMEOUT,
        )
    except Exception as exc:
        print(f"[web_search] DDGS failed: {exc}")
        return {"query": query, "error": str(exc), "results": [], "summary": None}

    if not rows:
        print("[web_search] DDGS returned no rows")
        return {
            "query": query,
            "results": [],
            "summary": None,
            "note": "No search results returned.",
        }

    for i, row in enumerate(rows, start=1):
        u = row.get("url") or ""
        t = (row.get("title") or "")[:100]
        print(f"[web_search] ddgs[{i}] {u} — {t}")

    limits = httpx.Limits(max_connections=5, max_keepalive_connections=5)
    sem = asyncio.Semaphore(WEB_SEARCH_MAX_RESULTS)

    async with httpx.AsyncClient(
        limits=limits,
        follow_redirects=True,
    ) as client:
        fetched = await asyncio.gather(
            *[_fetch_one(client, row, sem) for row in rows],
            return_exceptions=False,
        )

    numbered: list[tuple[int, dict[str, Any]]] = [
        (i, item) for i, item in enumerate(fetched, start=1)
    ]

    results_out: list[dict[str, Any]] = []
    for _, src in numbered:
        results_out.append(
            {
                "title": src.get("title"),
                "url": src.get("url"),
                "snippet": src.get("snippet"),
                "had_extracted_text": bool(src.get("extracted_text")),
                "fetch_error": src.get("fetch_error"),
            }
        )

    for idx, src in numbered:
        ext = src.get("extracted_text") or ""
        err = src.get("fetch_error")
        print(
            f"[web_search] fetched[{idx}] extracted_chars={len(ext)} "
            f"had_text={bool(ext)} fetch_error={err!r}"
        )

    digest = _fallback_digest(numbered) if numbered else ""

    summary: str | None = None
    summary_error: str | None = None
    attempted_llm = False

    if numbered and OPENAI_API_KEY:
        attempted_llm = True
        summary, summary_error = await _summarize_with_openai(query, numbered)

    if not OPENAI_API_KEY:
        summary_error = (
            "OPENAI_API_KEY not set; summary field uses capped snippets and extracts only."
        )

    llm_ok = bool(summary and summary.strip())
    if not llm_ok and digest:
        summary = digest

    out: dict[str, Any] = {
        "query": query,
        "summary": summary,
        "results": results_out,
    }
    if summary_error:
        out["summary_error"] = summary_error
    if attempted_llm and not llm_ok:
        out["note"] = (
            "Summarization step did not return usable text; summary uses capped raw excerpts."
        )

    if llm_ok:
        path = "llm_summary"
    elif attempted_llm:
        path = "fallback_digest_after_llm_failed"
    elif not OPENAI_API_KEY:
        path = "fallback_digest_no_api_key"
    else:
        path = "empty"
    print(
        f"[web_search] done query={query!r} "
        f"path={path} attempted_llm={attempted_llm} llm_ok={llm_ok}"
    )
    if summary_error:
        print(f"[web_search] summary_error: {summary_error}")
    _log_summary_preview(summary)

    return out
