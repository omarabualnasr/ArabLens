"""Keyless web search with a dedicated, article-focused news path.

DDGS retrieves public search results. Query routing, URL validation, ranking,
news relevance/freshness filtering, answer construction, caching, and source
presentation are implemented locally for this project. No AI API is used.
"""
from __future__ import annotations

import html
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any, Protocol
from urllib.parse import urlparse
from pathlib import Path

_NLP_DIR = Path(__file__).resolve().parents[1] / "Custom NLP"
if str(_NLP_DIR) not in sys.path:
    sys.path.insert(0, str(_NLP_DIR))
from nlp_engine import CustomNLP  # noqa: E402

_MAX_QUERY_LENGTH = 240
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "could", "did",
    "do", "does", "for", "from", "how", "i", "in", "is", "it", "its",
    "me", "of", "on", "or", "please", "the", "their", "there", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "you", "your", "tell", "explain", "find", "search",
    "web", "online", "latest", "latets", "current", "currently", "today",
    "yesterday", "recent", "newest", "now", "news", "headlines", "breaking",
    "show", "give", "me", "about", "happening", "going", "on", "please",
    "up", "are", "there", "any", "what's", "whats",
}
_NEWS_TERMS = re.compile(r"\b(news|headlines|breaking news)\b", re.IGNORECASE)
_FRESHNESS_PATTERNS = re.compile(
    r"\b(latest|current|currently|today|yesterday|tomorrow|right now|these days|"
    r"this week|this month|recent|newest|breaking|news|price|pricing|stock|weather|"
    r"exchange rate|release date|version|who is the (?:current|new)|as of 20\d\d|20\d\d)\b",
    re.IGNORECASE,
)
_GENERIC_NEWS_SLUGS = {
    "latest", "headlines", "topics", "topic", "news", "world", "world-news",
    "breaking-news", "home", "latest-news", "top-stories", "politics", "business",
    "technology", "sports", "us", "uk", "middle-east", "international", "live",
    "search", "section", "category", "tag", "feed", "index", "all",
}


class WebSearchError(RuntimeError):
    """Base class for user-safe web-search failures."""


class SearchRateLimited(WebSearchError):
    """Raised when upstream search backends throttle requests."""


class SearchProviderUnavailable(WebSearchError):
    """Raised for missing dependencies, network, or provider errors."""


class _TextOnlyHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def clean_snippet(value: Any, limit: int = 520) -> str:
    """Convert a provider snippet to safe, compact plain text."""
    if not isinstance(value, str):
        return ""
    parser = _TextOnlyHTML()
    try:
        parser.feed(value)
        parser.close()
        text = " ".join(parser.parts)
    except Exception:
        text = re.sub(r"<[^>]*>", " ", value)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit].rstrip()


def _safe_web_url(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) > 2_048:
        return None
    try:
        parsed = urlparse(value)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return value


def _words(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9]{2,}", text.lower()) if word not in _STOP_WORDS]


def parse_published_date(value: Any) -> datetime | None:
    """Parse provider timestamps into timezone-aware UTC datetimes."""
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    lowered = raw.lower()
    now = datetime.now(timezone.utc)
    if lowered in {"now", "just now", "today"}:
        return now
    if lowered == "yesterday":
        return now - timedelta(days=1)
    relative = re.fullmatch(r"(\d+)\s+(minute|hour|day)s?\s+ago", lowered)
    if relative:
        amount = int(relative.group(1))
        unit = relative.group(2)
        delta = timedelta(minutes=amount) if unit == "minute" else timedelta(hours=amount) if unit == "hour" else timedelta(days=amount)
        return now - delta
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_article_url(value: str) -> bool:
    """Reject homepages, topic hubs, tags and category pages as news stories."""
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    path = parsed.path.strip("/").lower()
    if not path:
        return False
    if parsed.netloc.lower().endswith("news.google.com") and (path.startswith("topics/") or path.startswith("search")):
        return False
    if path.startswith(("topics/", "topic/", "category/", "categories/", "tag/", "tags/", "section/", "sections/", "author/", "search/")):
        return False
    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        return False
    last = segments[-1].replace(".html", "").replace(".htm", "")
    if last in _GENERIC_NEWS_SLUGS:
        return False
    # A single short generic segment is almost always a section, not a story.
    if len(segments) == 1 and last in _GENERIC_NEWS_SLUGS:
        return False
    return True


def build_news_query(question: str) -> str | None:
    """Detect news intent and turn conversational/typo-heavy wording into a topic.

    A broad request such as 'what is the latets news today?' becomes 'world news'.
    A focused request such as 'latest AI news' becomes 'AI'.
    """
    text = " ".join((question or "").split()).strip()
    if not text or not _NEWS_TERMS.search(text):
        return None
    cleaned = re.sub(r"\b(?:breaking\s+)?news\b|\bheadlines\b", " ", text, flags=re.IGNORECASE)
    tokens = _words(cleaned)
    # Remove common misspellings/variants of recency wording as well.
    topic = " ".join(tokens[:8]).strip()
    if not topic or topic.lower() in {"latest", "current", "today", "recent"}:
        return "world news"
    return topic[:120]


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    description: str
    domain: str
    published_at: str | None = None
    publisher: str | None = None

    def to_public_dict(self) -> dict[str, str]:
        payload = {
            "title": self.title,
            "url": self.url,
            "description": self.description,
            "domain": self.domain,
        }
        if self.published_at:
            payload["published_at"] = self.published_at
        if self.publisher:
            payload["publisher"] = self.publisher
        return payload


class SearchProvider(Protocol):
    def search(self, query: str, count: int = 8) -> list[SearchResult]: ...


class DDGSSearchProvider:
    """Keyless metasearch provider using DDGS text and dedicated news endpoints."""

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = max(3.0, min(float(timeout), 20.0))
        self.region = os.environ.get("TINY_AI_SEARCH_REGION", "wt-wt").strip() or "wt-wt"
        self.backend = os.environ.get("TINY_AI_SEARCH_BACKEND", "auto").strip() or "auto"

    @property
    def available(self) -> bool:
        try:
            from ddgs import DDGS  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _raise_provider_error(exc: Exception) -> None:
        exception_name = type(exc).__name__.lower()
        message = str(exc).lower()
        if "ratelimit" in exception_name or "rate limit" in message or "too many requests" in message or "429" in message:
            raise SearchRateLimited(
                "Free search sources are temporarily limiting requests. Wait a little and try again."
            ) from exc
        raise SearchProviderUnavailable(
            "Free web-search sources could not be reached. They may be temporarily unavailable or blocking automated requests."
        ) from exc

    def _client_class(self):
        try:
            from ddgs import DDGS
            return DDGS
        except ImportError as exc:
            raise SearchProviderUnavailable(
                "The free web-search package is missing. Run: py -m pip install -r requirements.txt"
            ) from exc

    def search(self, query: str, count: int = 8) -> list[SearchResult]:
        query = " ".join(query.split()).strip()
        if not query or len(query) > _MAX_QUERY_LENGTH:
            raise ValueError(f"Search queries must contain 1–{_MAX_QUERY_LENGTH} characters.")
        DDGS = self._client_class()
        try:
            raw_results = DDGS(timeout=int(self.timeout)).text(
                query,
                region=self.region,
                safesearch="moderate",
                max_results=max(1, min(int(count), 10)),
                backend=self.backend,
            )
        except Exception as exc:
            self._raise_provider_error(exc)

        results: list[SearchResult] = []
        seen: set[str] = set()
        for item in raw_results if isinstance(raw_results, (list, tuple)) else []:
            if not isinstance(item, dict):
                continue
            url = _safe_web_url(item.get("href") or item.get("url"))
            if not url or url in seen:
                continue
            parsed = urlparse(url)
            title = clean_snippet(item.get("title"), 180) or parsed.netloc
            description = clean_snippet(item.get("body") or item.get("description"), 900)
            seen.add(url)
            results.append(SearchResult(title, url, description, parsed.netloc.lower()))
            if len(results) >= count:
                break
        return results

    def search_news(self, query: str, count: int = 10, timelimit: str = "d") -> list[SearchResult]:
        """Use DDGS's dedicated news search, preserving publisher and publication date."""
        query = " ".join(query.split()).strip()
        if not query or len(query) > _MAX_QUERY_LENGTH:
            raise ValueError(f"Search queries must contain 1–{_MAX_QUERY_LENGTH} characters.")
        if timelimit not in {"d", "w", "m"}:
            raise ValueError("News timelimit must be d, w, or m.")
        DDGS = self._client_class()
        try:
            raw_results = DDGS(timeout=int(self.timeout)).news(
                query=query,
                region=self.region,
                safesearch="moderate",
                timelimit=timelimit,
                max_results=max(1, min(int(count), 20)),
                backend=self.backend,
            )
        except Exception as exc:
            self._raise_provider_error(exc)

        results: list[SearchResult] = []
        seen: set[str] = set()
        for item in raw_results if isinstance(raw_results, (list, tuple)) else []:
            if not isinstance(item, dict):
                continue
            url = _safe_web_url(item.get("url") or item.get("href"))
            if not url or url in seen or not _is_article_url(url):
                continue
            parsed = urlparse(url)
            title = clean_snippet(item.get("title"), 220)
            description = clean_snippet(item.get("body") or item.get("description"), 1_000)
            if not title or not (description or item.get("date")):
                continue
            publisher = clean_snippet(item.get("source") or item.get("publisher"), 100) or parsed.netloc.lower()
            published = item.get("date") or item.get("published_at")
            published_at = str(published).strip() if published else None
            seen.add(url)
            results.append(SearchResult(title, url, description, parsed.netloc.lower(), published_at, publisher))
            if len(results) >= count:
                break
        return results


@dataclass(frozen=True)
class WebSearchAnswer:
    answer: str
    sources: tuple[SearchResult, ...]
    query: str


class WebSearchService:
    """Locally ranks web results and builds concise, source-backed responses."""

    def __init__(self, provider: SearchProvider | None = None, *, cache_seconds: int = 600) -> None:
        self.provider = provider or DDGSSearchProvider()
        self.cache_seconds = max(0, int(cache_seconds))
        self._cache: dict[str, tuple[float, list[SearchResult]]] = {}
        self._news_cache: dict[str, tuple[float, list[SearchResult]]] = {}
        self._lock = threading.RLock()

    @property
    def available(self) -> bool:
        return bool(getattr(self.provider, "available", True))

    def _cached(self, cache: dict[str, tuple[float, list[SearchResult]]], key: str) -> list[SearchResult] | None:
        now = time.monotonic()
        with self._lock:
            cached = cache.get(key)
            if cached and now - cached[0] < self.cache_seconds:
                return list(cached[1])
        return None

    def _store(self, cache: dict[str, tuple[float, list[SearchResult]]], key: str, results: list[SearchResult]) -> None:
        with self._lock:
            cache[key] = (time.monotonic(), list(results))
            if len(cache) > 128:
                oldest = min(cache, key=lambda item: cache[item][0])
                cache.pop(oldest, None)

    def search(self, query: str) -> list[SearchResult]:
        normalized = " ".join(query.lower().split())
        cached = self._cached(self._cache, normalized)
        if cached is not None:
            return cached
        results = self.provider.search(query, count=8)
        ranked = self._rank(query, results)
        self._store(self._cache, normalized, ranked)
        return ranked

    def search_news(self, query: str, *, timelimit: str = "d") -> list[SearchResult]:
        normalized = " ".join(query.lower().split())
        cache_key = f"{timelimit}:{normalized}"
        cached = self._cached(self._news_cache, cache_key)
        if cached is not None:
            return cached
        search_news = getattr(self.provider, "search_news", None)
        if not callable(search_news):
            raise SearchProviderUnavailable("The configured search provider does not support dedicated news search.")
        raw_results = search_news(query, count=16, timelimit=timelimit)
        ranked = self._rank_news(query, raw_results, timelimit=timelimit)
        self._store(self._news_cache, cache_key, ranked)
        return ranked

    @staticmethod
    def _rank(query: str, results: list[SearchResult]) -> list[SearchResult]:
        query_terms = set(_words(query))
        if not query_terms:
            return results

        scored: list[tuple[tuple[float, int], SearchResult]] = []
        for result in results:
            title_terms = set(_words(result.title))
            body_terms = set(_words(result.description))
            matched_terms = query_terms & (title_terms | body_terms)
            if not matched_terms:
                # Do not show unrelated results just because the provider returned them.
                continue
            weighted_matches = 2.5 * len(query_terms & title_terms) + len(query_terms & body_terms)
            coverage = len(matched_terms) / max(1, len(query_terms))
            scored.append(((weighted_matches + coverage, -len(result.description)), result))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [result for _, result in scored]

    @staticmethod
    def _rank_news(query: str, results: list[SearchResult], *, timelimit: str) -> list[SearchResult]:
        now = datetime.now(timezone.utc)
        cutoff = now - (timedelta(days=2) if timelimit == "d" else timedelta(days=8 if timelimit == "w" else 32))
        query_terms = set(_words(query))
        candidates: list[tuple[float, SearchResult]] = []
        seen_titles: set[str] = set()
        for result in results:
            if not _is_article_url(result.url) or not result.title.strip():
                continue
            published = parse_published_date(result.published_at)
            if published and (published > now + timedelta(hours=12) or published < cutoff):
                continue
            if timelimit == "d" and published is None:
                # A current-news request must not silently promote undated search pages.
                continue
            normalized_title = re.sub(r"[^a-z0-9]+", " ", result.title.lower()).strip()
            if normalized_title in seen_titles:
                continue
            seen_titles.add(normalized_title)
            title_terms = set(_words(result.title))
            body_terms = set(_words(result.description))
            overlap = len(query_terms & (title_terms | body_terms))
            if query_terms and overlap == 0 and query.strip().lower() != "world news":
                continue
            topic_score = overlap / max(1, len(query_terms))
            recency_score = 0.0
            if published:
                age_hours = max(0.0, (now - published).total_seconds() / 3600)
                recency_score = max(0.0, 1.0 - age_hours / (48 if timelimit == "d" else 168))
            score = topic_score * 3.0 + recency_score + (0.25 if result.description else 0)
            candidates.append((score, result))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [result for _, result in candidates[:8]]

    @staticmethod
    def _rewrite_question_query(question: str) -> tuple[str, str, tuple[str, ...]]:
        """Turn a conversational question into a focused search phrase."""
        analysis = CustomNLP().analyze(question)
        focus = analysis.focus.strip()
        terms = analysis.focus_tokens or analysis.expanded_keywords
        if not focus or not terms:
            return question, focus, tuple(terms)
        if analysis.intent == "definition":
            return f"{focus} definition", focus, tuple(terms)
        if analysis.intent in {"explanation", "why"}:
            return f"{focus} explanation how it works", focus, tuple(terms)
        if analysis.intent == "procedure":
            return f"{focus} how to", focus, tuple(terms)
        return question, focus, tuple(terms)

    @staticmethod
    def _sentence_candidates(query: str, results: list[SearchResult]) -> list[tuple[float, str, int]]:
        """Rank factual sentences from result descriptions for a direct answer."""
        analysis = CustomNLP().analyze(query)
        focus_terms = set(analysis.focus_tokens or analysis.expanded_keywords)
        if not focus_terms:
            focus_terms = set(_words(query))
        candidates: list[tuple[float, str, int]] = []
        seen: set[str] = set()
        for result_index, result in enumerate(results):
            title_terms = set(_words(result.title))
            description_sentences = re.split(r"(?<=[.!?])\s+|\s+[•|]\s+", result.description)
            previous_sentence_relevant = False
            for sentence in description_sentences:
                clean = re.sub(r"^[-•*\s]+", "", sentence).strip()
                clean = re.sub(r"\s+", " ", clean)
                if len(clean) < 24 or len(clean) > 420:
                    previous_sentence_relevant = False
                    continue
                lowered = clean.lower()
                if any(phrase in lowered for phrase in (
                    "click here", "read more", "subscribe", "sign up", "cookie policy",
                    "privacy policy", "all rights reserved", "search results for",
                )):
                    previous_sentence_relevant = False
                    continue
                terms = set(_words(clean))
                overlap_count = len(focus_terms & terms)
                overlap = overlap_count / max(1, len(focus_terms))
                title_overlap = len(focus_terms & title_terms) / max(1, len(focus_terms))
                contextual = overlap_count == 0 and previous_sentence_relevant
                if overlap_count == 0 and title_overlap == 0 and not contextual:
                    previous_sentence_relevant = False
                    continue
                definition = bool(re.search(r"\b(is|are|means|refers to|is defined as|describes)\b", lowered))
                word_count = len(clean.split())
                readability = 1.0 - min(abs(word_count - 19) / 38, 0.65)
                score = overlap * 2.2 + title_overlap * 0.45 + (0.55 if definition else 0.0) + readability * 0.25
                if contextual:
                    score = max(score, 0.72)
                normalized = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
                if normalized not in seen:
                    seen.add(normalized)
                    candidates.append((score, clean, result_index))
                previous_sentence_relevant = overlap_count > 0 or title_overlap > 0
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates

    @staticmethod
    def _direct_answer(query: str, results: list[SearchResult]) -> str:
        """Compose a concise answer paragraph instead of printing search excerpts."""
        analysis = CustomNLP().analyze(query)
        focus = analysis.focus.strip()
        candidates = WebSearchService._sentence_candidates(query, results)
        if not candidates:
            if focus:
                return f"I found pages about {focus}, but their descriptions did not contain enough clear information to answer confidently. The sources are available below."
            return "I found some relevant pages, but their descriptions did not contain enough clear information to answer confidently."

        # Keep supporting context from the same top result together where
        # possible. This creates a readable answer rather than a list of snippets.
        first_score, first_sentence, first_result = candidates[0]
        chosen = [first_sentence]
        chosen_sets = [set(_words(first_sentence))]
        same_result = next((item for item in candidates[1:] if item[2] == first_result), None)
        second = same_result or next((item for item in candidates[1:] if item[2] != first_result), None)
        if second:
            _, sentence, _ = second
            words = set(_words(sentence))
            overlap = len(words & chosen_sets[0]) / max(1, len(words | chosen_sets[0]))
            if overlap <= 0.72:
                if focus and re.match(rf"^{re.escape(focus)}\s+", sentence, flags=re.I):
                    sentence = re.sub(rf"^{re.escape(focus)}\s+", "It ", sentence, count=1, flags=re.I)
                chosen.append(sentence)
        answer = " ".join(chosen)
        if answer and answer[-1] not in ".!?…":
            answer += "."
        return answer or "I couldn't form a reliable answer from the search results. Please try a more specific question."

    def answer(self, query: str) -> WebSearchAnswer:
        question = " ".join(query.split()).strip()
        if not question:
            raise ValueError("A search query is required.")
        search_query, _, _ = self._rewrite_question_query(question)
        results = self.search(search_query)
        sources = tuple(results[:5])
        if not results:
            answer = f"I searched for “{search_query}”, but no useful pages were returned. Try rephrasing the question or making it more specific."
            return WebSearchAnswer(answer, (), search_query)
        answer = self._direct_answer(question, results)
        return WebSearchAnswer(answer, sources, search_query)

    def answer_news(self, topic: str, *, timelimit: str = "d") -> WebSearchAnswer:
        """Return a real headline roundup, not generic search/homepage excerpts."""
        topic = " ".join((topic or "world news").split()).strip() or "world news"
        results = self.search_news(topic, timelimit=timelimit)
        sources = tuple(results[:5])
        if not results:
            period = "today" if timelimit == "d" else "this week" if timelimit == "w" else "this month"
            answer = (
                f"I couldn't find reliable article-level headlines for {topic!r} {period}. "
                "The search provider did not return enough dated news articles, so I won't present generic news homepages as headlines. "
                "Try a more specific topic, such as 'AI news', 'UAE news', or 'technology news'."
            )
            return WebSearchAnswer(answer, (), topic)

        now = datetime.now(timezone.utc)
        date_label = now.strftime("%A, %d %B %Y")
        period_label = "past 24 hours" if timelimit == "d" else "past week" if timelimit == "w" else "past month"
        lines = [f"Here are the latest {topic} headlines I found for {date_label} (published within the {period_label}):"]
        for index, result in enumerate(results[:5], start=1):
            summary = result.description.strip()
            if len(summary) > 300:
                summary = summary[:297].rsplit(" ", 1)[0] + "…"
            published = parse_published_date(result.published_at)
            when = published.strftime("%d %b %Y, %H:%M UTC") if published else "publication time unavailable"
            publisher = result.publisher or result.domain
            lines.append(f"\n{index}. **{result.title}**\n{summary or 'Open the article for details.'}\nSource: {publisher} · {when}")
        lines.append("\nThese are summaries of search-result descriptions, not full-article analysis. Open the sources to read the complete reporting.")
        return WebSearchAnswer("\n".join(lines), sources, topic)


def extract_explicit_search_query(message: str) -> str | None:
    """Return the requested query for explicit search commands, else None."""
    text = message.strip()
    patterns = (
        r"^/search\s+(.+)$",
        r"^search\s*:\s*(.+)$",
        r"^search\s+the\s+web\s+for\s+(.+)$",
        r"^search\s+web\s+for\s+(.+)$",
        r"^look\s+up\s+(?:online\s+)?(.+)$",
        r"^find\s+online\s+(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()[:_MAX_QUERY_LENGTH]
    return None


def question_needs_fresh_web_data(question: str) -> bool:
    """Heuristic routing for questions whose answers change over time."""
    return bool(_FRESHNESS_PATTERNS.search(question))


def load_dotenv(path: str) -> None:
    """Load simple KEY=VALUE entries without a third-party dotenv dependency."""
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
