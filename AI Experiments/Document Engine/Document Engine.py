"""Custom document ingestion, extraction, indexing, and retrieval for Tiny AI.

The parsers extract text only; all chunking, NLP analysis, ranking, and answer
composition are implemented in this project. No pretrained model or external
AI service is involved.
"""
from __future__ import annotations

import html
import json
import math
import re
import sys
import uuid
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

from pypdf import PdfReader

_NLP_DIR = Path(__file__).resolve().parents[1] / "Custom NLP"
if str(_NLP_DIR) not in sys.path:
    sys.path.insert(0, str(_NLP_DIR))
from nlp_engine import CustomNLP  # noqa: E402
_REASONING_DIR = Path(__file__).resolve().parents[1] / "Custom Reasoning Engine"
if str(_REASONING_DIR) not in sys.path:
    sys.path.insert(0, str(_REASONING_DIR))
from reasoning_engine import CustomReasoningEngine  # noqa: E402

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".html", ".htm"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_EXTRACTED_CHARS = 400_000
MAX_CHUNKS_PER_DOCUMENT = 180
MAX_DOCX_UNCOMPRESSED_BYTES = 24 * 1024 * 1024
MAX_DOCX_ENTRIES = 300
CHUNK_CHARS = 1_250
CHUNK_OVERLAP = 180


class DocumentUploadError(ValueError):
    """Raised when a document cannot safely be accepted or extracted."""


@dataclass(frozen=True)
class ExtractedSection:
    text: str
    page: int | None = None
    heading: str | None = None


@dataclass(frozen=True)
class DocumentChunk:
    index: int
    text: str
    page: int | None = None
    heading: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    filename: str
    extension: str
    uploaded_at: str
    size_bytes: int
    character_count: int
    word_count: int
    page_count: int | None
    chunks: tuple[DocumentChunk, ...]
    topics: tuple[str, ...]

    def to_storage_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "extension": self.extension,
            "uploaded_at": self.uploaded_at,
            "size_bytes": self.size_bytes,
            "character_count": self.character_count,
            "word_count": self.word_count,
            "page_count": self.page_count,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "topics": list(self.topics),
        }

    def to_public_dict(self) -> dict[str, object]:
        return {
            "id": self.document_id,
            "filename": self.filename,
            "type": self.extension.lstrip("."),
            "uploaded_at": self.uploaded_at,
            "size_bytes": self.size_bytes,
            "character_count": self.character_count,
            "word_count": self.word_count,
            "page_count": self.page_count,
            "chunk_count": len(self.chunks),
            "topics": list(self.topics),
        }


@dataclass(frozen=True)
class DocumentAnswer:
    answer: str
    confidence: float
    sources: tuple[dict[str, object], ...]
    document_ids: tuple[str, ...]


class _VisibleHTMLText(HTMLParser):
    """Small standard-library HTML text extractor that ignores scripts/styles."""

    HIDDEN_TAGS = {"script", "style", "noscript", "svg", "head"}
    BLOCK_TAGS = {"p", "div", "br", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.HIDDEN_TAGS:
            self.hidden_depth += 1
        elif not self.hidden_depth and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.HIDDEN_TAGS and self.hidden_depth:
            self.hidden_depth -= 1
        elif not self.hidden_depth and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth and data.strip():
            self.parts.append(data.strip())


def _clean_text(value: str) -> str:
    value = value.replace("\x00", " ").replace("\ufeff", " ")
    value = html.unescape(value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[\t\f\v ]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _decode_text(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def _split_section(section: ExtractedSection, start_index: int) -> list[DocumentChunk]:
    text = _clean_text(section.text)
    if not text:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    raw_chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) <= CHUNK_CHARS:
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= CHUNK_CHARS:
                current = candidate
                continue
            if current:
                raw_chunks.append(current)
            current = paragraph
            continue

        if current:
            raw_chunks.append(current)
            current = ""
        # Prefer sentence boundaries. Fall back to a hard character window for
        # long tables, code, or unpunctuated text.
        remaining = paragraph
        while len(remaining) > CHUNK_CHARS:
            cut = max(remaining.rfind(". ", 0, CHUNK_CHARS), remaining.rfind("? ", 0, CHUNK_CHARS), remaining.rfind("! ", 0, CHUNK_CHARS))
            if cut < CHUNK_CHARS // 2:
                cut = CHUNK_CHARS
            else:
                cut += 1
            raw_chunks.append(remaining[:cut].strip())
            remaining = remaining[max(0, cut - CHUNK_OVERLAP):].strip()
        current = remaining
    if current:
        raw_chunks.append(current)

    return [
        DocumentChunk(index=start_index + offset, text=chunk, page=section.page, heading=section.heading)
        for offset, chunk in enumerate(raw_chunks)
        if chunk
    ]


class DocumentEngine:
    """Persistent text document store with custom lexical retrieval and QA."""

    def __init__(self, storage_dir: str | Path, *, nlp: CustomNLP | None = None) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.nlp = nlp or CustomNLP()
        self.reasoner = CustomReasoningEngine(self.nlp)
        self.documents: dict[str, DocumentRecord] = {}
        self._vocabulary: set[str] = set()
        self._load()

    def _load(self) -> None:
        for path in sorted(self.storage_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                chunks = tuple(DocumentChunk(**chunk) for chunk in payload.get("chunks", []))
                record = DocumentRecord(
                    document_id=str(payload["document_id"]),
                    filename=str(payload["filename"]),
                    extension=str(payload["extension"]),
                    uploaded_at=str(payload["uploaded_at"]),
                    size_bytes=int(payload["size_bytes"]),
                    character_count=int(payload["character_count"]),
                    word_count=int(payload["word_count"]),
                    page_count=payload.get("page_count"),
                    chunks=chunks,
                    topics=tuple(str(topic) for topic in payload.get("topics", [])),
                )
                if record.document_id == path.stem and record.chunks:
                    self.documents[record.document_id] = record
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                # One damaged document index must not prevent the rest of the
                # assistant's knowledge base from loading.
                continue
        self._rebuild_vocabulary()

    def _rebuild_vocabulary(self) -> None:
        self._vocabulary = set()
        for record in self.documents.values():
            self._vocabulary.update(self.nlp.analyze(record.filename).expanded_keywords)
            for chunk in record.chunks:
                self._vocabulary.update(self.nlp.analyze(chunk.text).expanded_keywords)

    @staticmethod
    def _extract_docx(payload: bytes) -> list[ExtractedSection]:
        try:
            with zipfile.ZipFile(__import__("io").BytesIO(payload)) as archive:
                infos = archive.infolist()
                if len(infos) > MAX_DOCX_ENTRIES:
                    raise DocumentUploadError("This Word document contains too many internal files.")
                if sum(info.file_size for info in infos) > MAX_DOCX_UNCOMPRESSED_BYTES:
                    raise DocumentUploadError("This Word document expands beyond the safe extraction limit.")
                if "word/document.xml" not in archive.namelist():
                    raise DocumentUploadError("This DOCX file does not contain a readable document body.")
                root = ET.fromstring(archive.read("word/document.xml"))
        except DocumentUploadError:
            raise
        except (zipfile.BadZipFile, KeyError, ET.ParseError, OSError) as exc:
            raise DocumentUploadError("The Word document is damaged or could not be opened.") from exc

        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        body = root.find("w:body", ns)
        if body is None:
            return []
        sections: list[ExtractedSection] = []
        current_heading: str | None = None
        for child in body:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                text = "".join(node.text or "" for node in child.findall(".//w:t", ns)).strip()
                if not text:
                    continue
                style = child.find("w:pPr/w:pStyle", ns)
                style_name = style.attrib.get(f"{{{ns['w']}}}val", "") if style is not None else ""
                if style_name.lower().startswith("heading") or style_name.lower() == "title":
                    current_heading = text
                sections.append(ExtractedSection(text, heading=current_heading))
            elif tag == "tbl":
                rows: list[str] = []
                for row in child.findall(".//w:tr", ns):
                    cells = [" ".join("".join(node.text or "" for node in cell.findall(".//w:t", ns)).split()) for cell in row.findall("w:tc", ns)]
                    rows.append(" | ".join(cell for cell in cells if cell))
                if rows:
                    sections.append(ExtractedSection("\n".join(rows), heading=current_heading))
        return sections

    @staticmethod
    def _extract_pdf(payload: bytes) -> list[ExtractedSection]:
        try:
            from io import BytesIO
            reader = PdfReader(BytesIO(payload), strict=False)
            if reader.is_encrypted:
                try:
                    unlocked = reader.decrypt("")
                except Exception:
                    unlocked = 0
                if not unlocked:
                    raise DocumentUploadError("This PDF is password-protected. Remove its password before uploading.")
            if len(reader.pages) > 250:
                raise DocumentUploadError("This PDF has more than 250 pages. Please split it into smaller files.")
            sections = [ExtractedSection(page.extract_text() or "", page=index + 1) for index, page in enumerate(reader.pages)]
            return sections
        except DocumentUploadError:
            raise
        except Exception as exc:
            raise DocumentUploadError("The PDF could not be read. Please check that it is a valid, text-based PDF.") from exc

    @staticmethod
    def _extract_html(payload: bytes) -> list[ExtractedSection]:
        parser = _VisibleHTMLText()
        parser.feed(_decode_text(payload))
        parser.close()
        return [ExtractedSection("\n".join(parser.parts))]

    def extract(self, filename: str, payload: bytes) -> tuple[list[ExtractedSection], int | None]:
        extension = Path(filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise DocumentUploadError(f"Unsupported file type '{extension or 'none'}'. Supported formats: {supported}.")
        if not payload:
            raise DocumentUploadError("The selected document is empty.")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise DocumentUploadError("Each document must be 15 MB or smaller.")

        if extension == ".pdf":
            sections = self._extract_pdf(payload)
            page_count: int | None = len(sections)
        elif extension == ".docx":
            sections = self._extract_docx(payload)
            page_count = None
        elif extension in {".html", ".htm"}:
            sections = self._extract_html(payload)
            page_count = None
        else:
            sections = [ExtractedSection(_decode_text(payload))]
            page_count = None

        cleaned_sections = [ExtractedSection(_clean_text(section.text), section.page, section.heading) for section in sections]
        total_chars = sum(len(section.text) for section in cleaned_sections)
        if total_chars > MAX_EXTRACTED_CHARS:
            raise DocumentUploadError("This document contains too much extracted text. Please split it into smaller files.")
        if total_chars < 20:
            raise DocumentUploadError(
                "I couldn't find readable text in this document. Scanned/image-only PDFs need OCR, which is not included yet."
            )
        return cleaned_sections, page_count

    @staticmethod
    def _safe_filename(filename: str) -> str:
        name = Path(filename.replace("\\", "/")).name
        name = re.sub(r"[^\w.() -]+", "_", name, flags=re.UNICODE).strip(" .")
        if not name:
            name = "uploaded-document"
        return name[:140]

    def _make_chunks(self, sections: Iterable[ExtractedSection]) -> tuple[DocumentChunk, ...]:
        chunks: list[DocumentChunk] = []
        for section in sections:
            chunks.extend(_split_section(section, len(chunks)))
            if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
                raise DocumentUploadError("This document is too long to index safely. Please split it into smaller files.")
        return tuple(chunks)

    def _topics(self, filename: str, chunks: tuple[DocumentChunk, ...]) -> tuple[str, ...]:
        from collections import Counter
        counter: Counter[str] = Counter()
        for chunk in chunks:
            analysis = self.nlp.analyze(chunk.text)
            counter.update(token for token in analysis.keywords if len(token) > 2)
        for token in self.nlp.analyze(Path(filename).stem).keywords:
            counter[token] += 4
        return tuple(word for word, _ in counter.most_common(8))

    def ingest(self, filename: str, payload: bytes) -> DocumentRecord:
        safe_name = self._safe_filename(filename)
        sections, page_count = self.extract(safe_name, payload)
        chunks = self._make_chunks(sections)
        if not chunks:
            raise DocumentUploadError("The document contains no text that can be indexed.")
        words = sum(len(self.nlp.tokenize(section.text)) for section in sections)
        document_id = uuid.uuid4().hex
        record = DocumentRecord(
            document_id=document_id,
            filename=safe_name,
            extension=Path(safe_name).suffix.lower(),
            uploaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            size_bytes=len(payload),
            character_count=sum(len(section.text) for section in sections),
            word_count=words,
            page_count=page_count,
            chunks=chunks,
            topics=self._topics(safe_name, chunks),
        )
        destination = self.storage_dir / f"{document_id}.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(record.to_storage_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(destination)
        self.documents[document_id] = record
        self._rebuild_vocabulary()
        return record

    def list_documents(self) -> list[dict[str, object]]:
        return [record.to_public_dict() for record in sorted(self.documents.values(), key=lambda item: item.uploaded_at, reverse=True)]

    def delete(self, document_id: str) -> bool:
        record = self.documents.pop(document_id, None)
        if record is None:
            return False
        (self.storage_dir / f"{document_id}.json").unlink(missing_ok=True)
        self._rebuild_vocabulary()
        return True

    @staticmethod
    def _char_ngrams(text: str, n: int = 3) -> set[str]:
        compact = re.sub(r"\s+", " ", text.lower()).strip()
        if len(compact) < n:
            return {compact} if compact else set()
        return {compact[index:index + n] for index in range(len(compact) - n + 1)}

    def _score_chunk(self, question: str, query_analysis: object, record: DocumentRecord, chunk: DocumentChunk) -> float:
        candidate = self.nlp.analyze(chunk.text, vocabulary=self._vocabulary)
        query_focus = set(getattr(query_analysis, "focus_tokens"))
        query_terms = query_focus or set(getattr(query_analysis, "expanded_keywords"))
        candidate_terms = set(candidate.expanded_keywords)
        if not query_terms:
            return 0.0
        shared = query_terms & candidate_terms
        coverage = len(shared) / max(1, len(query_terms))
        jaccard = len(shared) / max(1, len(query_terms | candidate_terms))
        focus_coverage = len(query_focus & candidate_terms) / max(1, len(query_focus)) if query_focus else 0.0
        phrase = 1.0 if len(getattr(query_analysis, "focus")) >= 4 and getattr(query_analysis, "focus").lower() in chunk.text.lower() else 0.0
        title_analysis = self.nlp.analyze(Path(record.filename).stem)
        title_terms = set(title_analysis.expanded_keywords)
        title_score = len(query_terms & title_terms) / max(1, len(query_terms))
        query_ngrams = self._char_ngrams(" ".join(sorted(query_terms)))
        chunk_ngrams = self._char_ngrams(" ".join(sorted(candidate_terms)))
        char_score = len(query_ngrams & chunk_ngrams) / max(1, len(query_ngrams | chunk_ngrams))
        return min(1.0, 0.32 * coverage + 0.17 * jaccard + 0.23 * focus_coverage + 0.16 * phrase + 0.08 * title_score + 0.04 * char_score)

    @staticmethod
    def _sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [part.strip(" •\t") for part in parts if len(part.strip()) >= 28]

    def _compose_answer(self, question: str, ranked: list[tuple[float, DocumentRecord, DocumentChunk]]) -> str:
        analysis = self.nlp.analyze(question, vocabulary=self._vocabulary)
        if analysis.intent == "summary" or re.search(r"\b(summarize|summary|overview|key points|main points)\b", analysis.normalized):
            selected: list[str] = []
            seen: set[str] = set()
            # For a summary request, cover the beginning and the strongest
            # distinct chunks rather than repeating one highly scored passage.
            candidates = sorted(ranked, key=lambda item: (item[2].index, -item[0]))[:8]
            candidates += sorted(ranked, key=lambda item: -item[0])[:8]
            for _, _, chunk in candidates:
                for sentence in self._sentences(chunk.text):
                    key = self.nlp.normalize(sentence)
                    if key not in seen:
                        selected.append(sentence)
                        seen.add(key)
                    if len(selected) >= 6:
                        break
                if len(selected) >= 6:
                    break
            if not selected:
                selected = [ranked[0][2].text[:600].strip()]
            return "Here's a summary of the uploaded document:\n\n" + "\n".join(f"• {sentence}" for sentence in selected)

        query_terms = set(analysis.focus_tokens) or set(analysis.expanded_keywords)
        sentence_scores: list[tuple[float, int, str]] = []
        for rank, (_, _, chunk) in enumerate(ranked):
            for sentence_index, sentence in enumerate(self._sentences(chunk.text)):
                sentence_analysis = self.nlp.analyze(sentence, vocabulary=self._vocabulary)
                terms = set(sentence_analysis.expanded_keywords)
                shared = query_terms & terms
                overlap = len(shared) / max(1, len(query_terms))
                precision = len(shared) / max(1, len(terms))
                phrase = 0.22 if len(analysis.focus) >= 4 and analysis.focus.lower() in sentence.lower() else 0.0
                score = 0.70 * overlap + 0.12 * precision + phrase + 0.03 * (1.0 / (1 + rank))
                if score > 0:
                    sentence_scores.append((score, rank * 1000 + sentence_index, sentence))
        sentence_scores.sort(key=lambda item: (-item[0], item[1]))
        chosen: list[str] = []
        seen: set[str] = set()
        if sentence_scores:
            best_score = sentence_scores[0][0]
            normalized_question = analysis.normalized
            count_limit = 1 if re.search(r"\b(how many|how much|how long|how early|who|when|where)\b", normalized_question) else 2
            for score, _, sentence in sentence_scores:
                if score < max(0.16, best_score * 0.58):
                    continue
                key = self.nlp.normalize(sentence)
                if key in seen:
                    continue
                chosen.append(sentence)
                seen.add(key)
                if len(chosen) >= count_limit:
                    break
        if not chosen:
            chosen = [ranked[0][2].text[:550].strip()]
        return " ".join(chosen)

    def answer(self, question: str, *, top_k: int = 4) -> DocumentAnswer | None:
        if not self.documents:
            return None
        analysis = self.nlp.analyze(question, vocabulary=self._vocabulary)
        summary_request = analysis.intent == "summary" or bool(re.search(
            r"\b(summarize|summary|overview|key points|main points)\b", analysis.normalized
        )) or bool(re.search(
            r"\b(what is|what's|tell me|explain)\b.*\b(in|about|inside|contain|include)\b.*\b(document|file|pdf|report)\b|"
            r"\bwhat does (?:the|this|my) (?:document|file|pdf|report) (?:say|contain|include)\b",
            analysis.normalized,
        ))
        document_reference = bool(re.search(r"\b(document|file|pdf|report|uploaded|attachment|this text|the text)\b", analysis.normalized))
        if not analysis.expanded_keywords and not summary_request:
            return None
        candidate_documents = list(self.documents.values())
        if summary_request and candidate_documents:
            focus_terms = set(analysis.focus_tokens) - {"document", "file", "pdf", "report", "summary", "summarize", "overview"}
            title_matches = []
            for record in candidate_documents:
                title_terms = set(self.nlp.analyze(Path(record.filename).stem).expanded_keywords)
                overlap = len(focus_terms & title_terms)
                if overlap:
                    title_matches.append((overlap, record))
            if title_matches:
                strongest = max(item[0] for item in title_matches)
                candidate_documents = [record for score, record in title_matches if score == strongest]
            else:
                candidate_documents = [max(candidate_documents, key=lambda item: item.uploaded_at)]
        ranked: list[tuple[float, DocumentRecord, DocumentChunk]] = []
        passage_records = [
            (record, chunk)
            for record in candidate_documents
            for chunk in record.chunks
        ]
        passage_texts = [
            f"{record.filename} {chunk.heading or ''} {chunk.text}"
            for record, chunk in passage_records
        ]
        bm25_scores = self.reasoner.rank_texts(question, passage_texts)
        for (record, chunk), bm25_score in zip(passage_records, bm25_scores):
            score = self._score_chunk(question, analysis, record, chunk)
            score = 0.88 * score + 0.12 * bm25_score
            if summary_request and document_reference:
                score = max(score, 0.22 + (0.03 if chunk.index == 0 else 0.0))
            if score >= 0.18:
                ranked.append((score, record, chunk))
        if not ranked:
            # A summary request can still work on the most recently added file.
            if summary_request and self.documents:
                record = max(self.documents.values(), key=lambda item: item.uploaded_at)
                ranked = [(0.2, record, chunk) for chunk in record.chunks[:8]]
            else:
                return None
        ranked.sort(key=lambda item: (-item[0], item[1].uploaded_at, item[2].index))
        selected = ranked[:max(1, min(8, top_k * 2))]
        answer = self._compose_answer(question, selected)
        confidence = min(0.99, max(score for score, _, _ in selected))
        source_items: list[dict[str, object]] = []
        document_ids: list[str] = []
        seen_sources: set[tuple[str, int | None]] = set()
        for score, record, chunk in selected[:top_k]:
            key = (record.document_id, chunk.page)
            if key in seen_sources:
                continue
            seen_sources.add(key)
            if record.document_id not in document_ids:
                document_ids.append(record.document_id)
            source_items.append({
                "source_type": "document",
                "document_id": record.document_id,
                "title": record.filename,
                "publisher": "Uploaded document",
                "page": chunk.page,
                "section": chunk.heading,
                "excerpt": chunk.text[:320],
                "score": round(score, 4),
            })
        return DocumentAnswer(answer, confidence, tuple(source_items), tuple(document_ids))
