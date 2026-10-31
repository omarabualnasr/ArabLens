"""Local text metrics and extraction utilities for Tiny AI."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_URL = re.compile(r"(?i)\bhttps?://[^\s<>\"]+")
_COMMON = {"the", "and", "for", "that", "with", "this", "from", "are", "was", "were", "you", "your", "have", "has", "had", "not", "but", "they", "them", "their", "about", "into", "then", "than", "when", "what", "which", "will", "would", "could", "should", "can", "all", "any", "our", "out", "its", "it's", "a", "an", "of", "to", "in", "on", "as", "at", "by", "be", "is", "it", "or", "we", "i"}


@dataclass(frozen=True)
class TextAnalyzerResult:
    answer: str
    operation: str
    source: str = "tool"
    tool: str = "text_analyzer"

    def to_dict(self) -> dict[str, str]:
        return {"answer": self.answer, "source": self.source, "tool": self.tool, "operation": self.operation}


class TextAnalyzer:
    """Perform explicit text operations; never rewrite arbitrary user questions."""

    _PREFIXES = (
        (r"^analy[sz]e\s+(?:this\s+)?text\s*:\s*", "analyze"),
        (r"^(?:word\s+count|count\s+words)\s*:\s*", "word_count"),
        (r"^(?:character\s+count|count\s+characters)\s*:\s*", "character_count"),
        (r"^extract\s+(?:email\s+addresses|emails)\s*(?:from)?\s*:\s*", "emails"),
        (r"^extract\s+(?:urls|links)\s*(?:from)?\s*:\s*", "urls"),
        (r"^uppercase\s*:\s*", "uppercase"),
        (r"^lowercase\s*:\s*", "lowercase"),
        (r"^title\s+case\s*:\s*", "title_case"),
        (r"^slugify\s*:\s*", "slugify"),
    )

    @staticmethod
    def _metrics(text: str) -> str:
        words = _WORD.findall(text)
        sentences = [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
        paragraphs = [part for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]
        lines = text.splitlines() or ([text] if text else [])
        top_terms = Counter(word.casefold() for word in words if word.casefold() not in _COMMON and len(word) > 2)
        frequent = ", ".join(f"{word} ({count})" for word, count in top_terms.most_common(5)) or "None"
        reading_minutes = max(1, round(len(words) / 225)) if words else 0
        characters_without_whitespace = len(re.sub(r"\s", "", text))
        return (
            f"Words: {len(words)}\nCharacters: {len(text)} (including spaces)\n"
            f"Characters without whitespace: {characters_without_whitespace}\n"
            f"Sentences: {len(sentences)}\nParagraphs: {len(paragraphs)}\nLines: {len(lines)}\n"
            f"Estimated reading time: {reading_minutes} minute(s)\nMost frequent useful terms: {frequent}"
        )

    def execute(self, question: str) -> TextAnalyzerResult | None:
        text = str(question or "").strip()
        for pattern, operation in self._PREFIXES:
            match = re.match(pattern, text, flags=re.I | re.S)
            if not match:
                continue
            content = text[match.end():].strip()
            if not content:
                return None
            if len(content) > 20_000:
                return TextAnalyzerResult("That text is too long to analyze in one request (maximum 20,000 characters).", operation)
            if operation == "analyze": answer = self._metrics(content)
            elif operation == "word_count": answer = f"Word count: {len(_WORD.findall(content))}."
            elif operation == "character_count":
                no_whitespace = len(re.sub(r"\s", "", content))
                answer = f"Character count: {len(content)} including spaces; {no_whitespace} excluding whitespace."
            elif operation == "emails":
                values = list(dict.fromkeys(_EMAIL.findall(content)))
                answer = "Email addresses found:\n" + ("\n".join(f"• {value}" for value in values) if values else "None.")
            elif operation == "urls":
                values = [value.rstrip(".,);]") for value in dict.fromkeys(_URL.findall(content))]
                answer = "URLs found:\n" + ("\n".join(f"• {value}" for value in values) if values else "None.")
            elif operation == "uppercase": answer = content.upper()
            elif operation == "lowercase": answer = content.lower()
            elif operation == "title_case": answer = content.title()
            else:
                slug = re.sub(r"[^a-z0-9]+", "-", content.casefold()).strip("-")
                answer = slug
            return TextAnalyzerResult(answer, operation)
        return None


__all__ = ["TextAnalyzer", "TextAnalyzerResult"]
