"""Native date/time tool for Tiny AI.

Uses Python's clock and the standard-library zoneinfo database. It does not
call a website, search provider, LLM, or external API.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


CITY_TIMEZONES = {
    "abu dhabi": "Asia/Dubai",
    "dubai": "Asia/Dubai",
    "uae": "Asia/Dubai",
    "united arab emirates": "Asia/Dubai",
    "london": "Europe/London",
    "uk": "Europe/London",
    "united kingdom": "Europe/London",
    "new york": "America/New_York",
    "nyc": "America/New_York",
    "los angeles": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles",
    "chicago": "America/Chicago",
    "toronto": "America/Toronto",
    "vancouver": "America/Vancouver",
    "paris": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "tokyo": "Asia/Tokyo",
    "singapore": "Asia/Singapore",
    "hong kong": "Asia/Hong_Kong",
    "india": "Asia/Kolkata",
    "mumbai": "Asia/Kolkata",
    "delhi": "Asia/Kolkata",
    "sydney": "Australia/Sydney",
}

# Keep matching specific. Questions such as "what is the best time to visit?"
# are not clock requests and must continue to the normal QA engine.
_TIME_PATTERNS = (
    r"\bwhat\s+time\s+(?:is\s+it|is\s+it\s+now)\b",
    r"\bwhat(?:\s+is|'s)\s+(?:the\s+)?time\b",
    r"\bcurrent\s+time\b",
    r"\btime\s+(?:now|right\s+now)\b",
    r"\btell\s+me\s+(?:the\s+)?(?:current\s+)?time\b",
    r"\bcan\s+you\s+(?:tell\s+me\s+)?(?:the\s+)?time\b",
)
_DATE_PATTERNS = (
    r"\bwhat(?:\s+is|'s)\s+(?:the\s+)?date\b",
    r"\bwhat\s+date\s+is\s+(?:it|today|tomorrow|yesterday)\b",
    r"\bwhat\s+day\s+(?:(?:is|was)\s+)?(?:it|today|tomorrow|yesterday)\b",
    r"\bwhat(?:\s+is|'s)\s+today(?:'s)?\b",
    r"\btoday'?s\s+date\b",
    r"\bcurrent\s+date\b",
    r"\bdate\s+today\b",
    r"\btell\s+me\s+(?:the\s+)?(?:date|day)\b",
)
_DATETIME_PATTERNS = (
    r"\bdate\s+and\s+time\b",
    r"\btime\s+and\s+date\b",
    r"\bcurrent\s+datetime\b",
    r"\bwhat(?:\s+is|'s)\s+(?:the\s+)?date\s+and\s+time\b",
    r"\bwhat\s+day\s+and\s+time\s+is\s+it\b",
)


@dataclass(frozen=True)
class DateTimeToolResult:
    answer: str
    operation: str
    timezone: str
    iso_datetime: str
    source: str = "tool"
    tool: str = "date_time"

    def to_dict(self) -> dict[str, str]:
        return {
            "answer": self.answer,
            "source": self.source,
            "tool": self.tool,
            "operation": self.operation,
            "timezone": self.timezone,
            "datetime": self.iso_datetime,
        }


class DateTimeTool:
    """Resolve straightforward current date/time questions from the live clock."""

    def __init__(
        self,
        default_timezone: str | None = None,
        *,
        clock: Callable[[ZoneInfo], datetime] | None = None,
    ) -> None:
        self.default_timezone = (
            default_timezone
            or os.environ.get("TINY_AI_TIMEZONE", "Asia/Dubai")
        ).strip()
        self._clock = clock or datetime.now

    @staticmethod
    def classify(question: str) -> str | None:
        """Return ``datetime``, ``time``, ``date``, or None for a non-clock query.

        Matching is intentionally constrained to direct clock/date requests so
        ordinary questions about time complexity, date formats, etc. are not
        accidentally intercepted by this tool.
        """
        text = re.sub(r"\s+", " ", str(question or "").lower().replace("’", "'")).strip()
        text = re.sub(r"[?!.,]+$", "", text).strip()
        if not text:
            return None

        place_suffix = r"(?:\s+(?:in|for)\s+[a-z0-9_+ /-]+)?"
        datetime_patterns = (
            rf"^(?:what(?:\s+is|'s)\s+)?(?:the\s+)?(?:current\s+)?date\s+and\s+time(?:\s+(?:today|now|right\s+now))?{place_suffix}$",
            rf"^(?:what(?:\s+is|'s)\s+)?(?:the\s+)?time\s+and\s+date(?:\s+(?:today|now|right\s+now))?{place_suffix}$",
            rf"^current\s+datetime{place_suffix}$",
            rf"^today'?s\s+date\s+and\s+time{place_suffix}$",
            rf"^what(?:\s+is|'s)\s+today'?s\s+date\s+and\s+time{place_suffix}$",
            rf"^tell\s+me\s+(?:the\s+)?date\s+and\s+time{place_suffix}$",
            rf"^(?:get|show)(?:\s+me)?\s+(?:the\s+)?(?:current\s+)?date\s+and\s+time{place_suffix}$",
        )
        if any(re.fullmatch(pattern, text) for pattern in datetime_patterns):
            return "datetime"

        time_patterns = (
            rf"^what\s+time\s+is\s+it(?:\s+(?:now|right\s+now))?{place_suffix}$",
            rf"^what(?:\s+is|'s)\s+(?:the\s+)?time(?:\s+(?:now|right\s+now))?{place_suffix}$",
            rf"^current\s+time(?:\s+(?:now|right\s+now))?{place_suffix}$",
            rf"^local\s+time{place_suffix}$",
            rf"^what(?:\s+is|'s)\s+(?:(?:the|my)\s+)?(?:current\s+|local\s+)?time{place_suffix}$",
            rf"^(?:get|show)(?:\s+me)?\s+(?:the\s+)?(?:current\s+|local\s+)?time{place_suffix}$",
            rf"^time\s+(?:now|right\s+now){place_suffix}$",
            rf"^tell\s+me\s+(?:the\s+)?(?:current\s+)?time{place_suffix}$",
            rf"^can\s+you\s+(?:tell\s+me\s+)?(?:the\s+)?time{place_suffix}$",
            rf"^can\s+you\s+tell\s+me\s+what\s+time\s+it\s+is{place_suffix}$",
        )
        if any(re.fullmatch(pattern, text) for pattern in time_patterns):
            return "time"

        date_patterns = (
            rf"^what(?:\s+is|'s)\s+(?:the\s+)?date(?:\s+(?:today|now|right\s+now))?{place_suffix}$",
            r"^what\s+date\s+is\s+(?:it|today|tomorrow|yesterday)$",
            r"^what\s+day\s+(?:(?:is|was)\s+)?(?:it|today|tomorrow|yesterday)(?:\s+today)?$",
            r"^what(?:\s+is|'s)\s+today(?:'s)?(?:\s+date)?$",
            r"^today'?s\s+date$",
            r"^current\s+date$",
            rf"^date\s+today{place_suffix}$",
            rf"^current\s+day{place_suffix}$",
            rf"^what(?:\s+is|'s)\s+(?:the\s+)?(?:current\s+)?day(?:\s+today)?{place_suffix}$",
            rf"^(?:get|show)(?:\s+me)?\s+(?:the\s+)?(?:current\s+|today's\s+)?date{place_suffix}$",
            rf"^tell\s+me\s+(?:the\s+)?(?:date|day){place_suffix}$",
        )
        if any(re.fullmatch(pattern, text) for pattern in date_patterns):
            return "date"
        return None

    @staticmethod
    def _explicit_timezone(question: str) -> str | None:
        original = str(question or "")
        text = original.lower()
        # Preserve IANA zone capitalization (Europe/London is case-sensitive).
        iana = re.search(r"\b(?:in|for)\s+([A-Za-z]+/[A-Za-z0-9_+\-/]+)\b", original, re.I)
        if iana:
            return iana.group(1)
        # Resolve a concise list of common city names and aliases.
        for place in sorted(CITY_TIMEZONES, key=len, reverse=True):
            if re.search(rf"\b(?:in|for)\s+{re.escape(place)}\b", text):
                return CITY_TIMEZONES[place]
        return None

    @staticmethod
    def _relative_day(question: str) -> int:
        text = str(question or "").lower()
        if re.search(r"\btomorrow\b", text):
            return 1
        if re.search(r"\byesterday\b", text):
            return -1
        return 0

    @staticmethod
    def _get_zone(timezone_name: str, fallback: str) -> ZoneInfo:
        try:
            return ZoneInfo(timezone_name)
        except (ZoneInfoNotFoundError, ValueError, TypeError):
            try:
                return ZoneInfo(fallback)
            except (ZoneInfoNotFoundError, ValueError, TypeError):
                # A system-local fallback remains better than failing the tool.
                local_zone = datetime.now().astimezone().tzinfo
                if isinstance(local_zone, ZoneInfo):
                    return local_zone
                return ZoneInfo("UTC")

    @staticmethod
    def _format_date(value: date) -> str:
        # %-d is not supported by Windows' strftime implementation.
        return value.strftime("%A, %B %d, %Y").replace(" 0", " ")

    @staticmethod
    def _format_time(value: datetime) -> str:
        return value.strftime("%I:%M %p").lstrip("0")

    @staticmethod
    def _offset_label(value: datetime) -> str:
        offset = value.strftime("%z") or "+0000"
        if len(offset) == 5:
            offset = f"{offset[:3]}:{offset[3:]}"
        return f"UTC{offset}"

    def execute(self, question: str, *, client_timezone: str | None = None) -> DateTimeToolResult | None:
        operation = self.classify(question)
        if operation is None:
            return None

        explicit_timezone = self._explicit_timezone(question)
        requested_timezone = explicit_timezone or client_timezone or self.default_timezone
        # ZoneInfo expects an IANA name. Bound incoming browser values and fall
        # back safely if the browser sends an unknown or malformed identifier.
        if not isinstance(requested_timezone, str) or len(requested_timezone) > 100:
            requested_timezone = self.default_timezone
        zone = self._get_zone(requested_timezone, self.default_timezone)
        now = self._clock(zone)
        if now.tzinfo is None:
            now = now.replace(tzinfo=zone)
        else:
            now = now.astimezone(zone)

        relative_date = now.date() + timedelta(days=self._relative_day(question))
        place_label = zone.key
        abbreviation = now.tzname() or place_label
        offset = self._offset_label(now)

        if operation == "time":
            answer = (
                f"It's {self._format_time(now)} in {place_label} "
                f"({abbreviation}, {offset})."
            )
        elif operation == "date":
            answer = f"{self._format_date(relative_date)} in {place_label}."
        else:
            answer = (
                f"It's {self._format_date(relative_date)} and the time is "
                f"{self._format_time(now)} in {place_label} "
                f"({abbreviation}, {offset})."
            )

        return DateTimeToolResult(
            answer=answer,
            operation=operation,
            timezone=place_label,
            iso_datetime=now.isoformat(timespec="seconds"),
        )


__all__ = ["DateTimeTool", "DateTimeToolResult"]
