"""Serve the custom Tiny AI engine and its Frontend chat interface."""
from __future__ import annotations

import json
import importlib.util
import os
import re
from email.parser import BytesParser
from email.policy import default as email_policy
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import uuid
from collections import deque
from time import monotonic, time

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "Frontend"
WEB_DIR = FRONTEND_DIR  # Keep the existing static-route code readable.
AI_EXPERIMENTS_DIR = ROOT / "AI Experiments"
ENGINE_DIR = AI_EXPERIMENTS_DIR / "Tiny AI Engine"
SEARCH_DIR = AI_EXPERIMENTS_DIR / "Web Search"
NLP_DIR = AI_EXPERIMENTS_DIR / "Custom NLP"
REASONING_DIR = AI_EXPERIMENTS_DIR / "Custom Reasoning Engine"
DATE_TIME_DIR = AI_EXPERIMENTS_DIR / "Date Time Tool"
DOCUMENT_DIR = AI_EXPERIMENTS_DIR / "Document Engine"
MEMORY_DIR = AI_EXPERIMENTS_DIR / "Memory Context Engine"
UNIT_CONVERTER_DIR = AI_EXPERIMENTS_DIR / "Unit Converter"
DATE_CALCULATOR_DIR = AI_EXPERIMENTS_DIR / "Date Calculator"
TEXT_ANALYZER_DIR = AI_EXPERIMENTS_DIR / "Text Analyzer"
JSON_TOOLS_DIR = AI_EXPERIMENTS_DIR / "JSON Tools"
SCIENTIFIC_MATH_DIR = AI_EXPERIMENTS_DIR / "Scientific Math"
NEURAL_MODEL_CHECKPOINT = AI_EXPERIMENTS_DIR / "Mini GPT" / "Checkpoints" / "Neural Model.npz"


def _load_local_module(module_name: str, module_path: Path):
    """Load a Python module from a filename that may contain spaces."""
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    if not module_path.is_file():
        raise FileNotFoundError(f"Required application module is missing: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load application module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


# Keep Python import aliases stable while making every visible filename readable.
# Dependencies are registered first because the custom modules import one another.
for _module_dir in (
    ENGINE_DIR, SEARCH_DIR, NLP_DIR, REASONING_DIR, DATE_TIME_DIR, DOCUMENT_DIR,
    MEMORY_DIR, UNIT_CONVERTER_DIR, DATE_CALCULATOR_DIR, TEXT_ANALYZER_DIR,
    JSON_TOOLS_DIR, SCIENTIFIC_MATH_DIR,
):
    if str(_module_dir) not in sys.path:
        sys.path.insert(0, str(_module_dir))

_load_local_module("nlp_engine", NLP_DIR / "NLP Engine.py")
_load_local_module("reasoning_engine", REASONING_DIR / "Reasoning Engine.py")
_search_module = _load_local_module("web_search", SEARCH_DIR / "Web Search.py")
_engine_module = _load_local_module("tiny_ai_engine", ENGINE_DIR / "Tiny AI Engine.py")
_date_time_module = _load_local_module("date_time_tool", DATE_TIME_DIR / "Date Time Tool.py")
_document_module = _load_local_module("document_engine", DOCUMENT_DIR / "Document Engine.py")
_memory_module = _load_local_module("memory_context_engine", MEMORY_DIR / "Memory Context Engine.py")
_unit_module = _load_local_module("unit_converter", UNIT_CONVERTER_DIR / "Unit Converter.py")
_date_calculator_module = _load_local_module("date_calculator", DATE_CALCULATOR_DIR / "Date Calculator.py")
_text_module = _load_local_module("text_analyzer", TEXT_ANALYZER_DIR / "Text Analyzer.py")
_json_module = _load_local_module("json_tools", JSON_TOOLS_DIR / "JSON Tools.py")
_math_module = _load_local_module("scientific_math", SCIENTIFIC_MATH_DIR / "Scientific Math.py")

DDGSSearchProvider = _search_module.DDGSSearchProvider
SearchProviderUnavailable = _search_module.SearchProviderUnavailable
SearchRateLimited = _search_module.SearchRateLimited
WebSearchService = _search_module.WebSearchService
build_news_query = _search_module.build_news_query
extract_explicit_search_query = _search_module.extract_explicit_search_query
load_dotenv = _search_module.load_dotenv
TinyAIEngine = _engine_module.TinyAIEngine
DateTimeTool = _date_time_module.DateTimeTool
DocumentEngine = _document_module.DocumentEngine
DocumentUploadError = _document_module.DocumentUploadError
MAX_UPLOAD_BYTES = _document_module.MAX_UPLOAD_BYTES
MemoryContextEngine = _memory_module.MemoryContextEngine
MemoryContextError = _memory_module.MemoryContextError
UnitConverter = _unit_module.UnitConverter
DateCalculator = _date_calculator_module.DateCalculator
TextAnalyzer = _text_module.TextAnalyzer
JSONTools = _json_module.JSONTools
ScientificMath = _math_module.ScientificMath

# Load optional locale/backend preferences. No search API secret is required.
load_dotenv(str(ROOT / ".env"))

ENGINE = TinyAIEngine()
ENGINE_LOCK = threading.RLock()
WEB_SEARCH = WebSearchService(DDGSSearchProvider(), cache_seconds=600)
DATE_TIME_TOOL = DateTimeTool()
DOCUMENT_ENGINE = DocumentEngine(os.environ.get("TINY_AI_DOCUMENTS_DIR", str(ROOT / "Data" / "Documents")))
DOCUMENT_LOCK = threading.RLock()
MEMORY_CONTEXT = MemoryContextEngine(os.environ.get("TINY_AI_MEMORY_PATH", str(ROOT / "Data" / "Memory" / "Memory Context.json")))
MEMORY_LOCK = threading.RLock()
UNIT_CONVERTER = UnitConverter()
DATE_CALCULATOR = DateCalculator()
TEXT_ANALYZER = TextAnalyzer()
JSON_TOOLS = JSONTools()
SCIENTIFIC_MATH = ScientificMath()
MAX_BODY_BYTES = 16_384
MAX_TOTAL_UPLOAD_BYTES = 26 * 1024 * 1024
MAX_FILES_PER_UPLOAD = 5
ALLOW_DOCUMENT_UPLOAD = os.environ.get("TINY_AI_ALLOW_DOCUMENT_UPLOAD", "true").strip().lower() in {"1", "true", "yes", "on"}
# Memory is enabled for local/private use. Hosted environments keep it off
# by default unless explicitly protected and enabled by the operator.
HOSTED_ENVIRONMENT = bool(os.environ.get("PORT") or os.environ.get("RENDER") or os.environ.get("DYNO") or os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("FLY_APP_NAME"))
ALLOW_MEMORY = os.environ.get("TINY_AI_ALLOW_MEMORY", "false" if HOSTED_ENVIRONMENT else "true").strip().lower() in {"1", "true", "yes", "on"}
ALLOW_TEACHING = os.environ.get("TINY_AI_ALLOW_TEACHING", "true").strip().lower() in {"1", "true", "yes", "on"}
TOOL_NAMES = ("calculator", "unit_converter", "date_time", "date_calculator", "scientific_math", "text_analyzer", "json_tools", "web_search", "document_reader", "memory_context")
# Tools are opt-in. A client must explicitly enable each tool per request.
DEFAULT_TOOL_SETTINGS = {
    "calculator": False,
    "unit_converter": False,
    "date_time": False,
    "date_calculator": False,
    "scientific_math": False,
    "text_analyzer": False,
    "json_tools": False,
    "web_search": False,
    "document_reader": False,
    "memory_context": False,
}
SEARCH_LIMIT_PER_MINUTE = max(1, int(os.environ.get("TINY_AI_SEARCH_LIMIT_PER_MINUTE", "10")))
SEARCH_LIMIT_PER_DAY = max(1, int(os.environ.get("TINY_AI_SEARCH_LIMIT_PER_DAY", "25")))
_SEARCH_REQUESTS: dict[str, deque[float]] = {}
_SEARCH_RATE_LOCK = threading.Lock()
_SEARCH_DAY = int(time() // 86_400)
_SEARCH_DAILY_COUNT = 0


def _allow_search_request(client_ip: str) -> bool:
    """Apply per-client and global daily limits to avoid overloading public search backends."""
    global _SEARCH_DAY, _SEARCH_DAILY_COUNT
    now = monotonic()
    day = int(time() // 86_400)
    with _SEARCH_RATE_LOCK:
        if day != _SEARCH_DAY:
            _SEARCH_DAY = day
            _SEARCH_DAILY_COUNT = 0
        if _SEARCH_DAILY_COUNT >= SEARCH_LIMIT_PER_DAY:
            return False
        window = _SEARCH_REQUESTS.setdefault(client_ip, deque())
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= SEARCH_LIMIT_PER_MINUTE:
            return False
        window.append(now)
        _SEARCH_DAILY_COUNT += 1
        if len(_SEARCH_REQUESTS) > 1_000:
            for address in list(_SEARCH_REQUESTS):
                if not _SEARCH_REQUESTS[address] or now - _SEARCH_REQUESTS[address][-1] >= 60:
                    _SEARCH_REQUESTS.pop(address, None)
        return True

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


def _valid_client_id(value: object) -> bool:
    """Accept only UUID-shaped browser identifiers; never trust arbitrary paths."""
    if not isinstance(value, str) or len(value) > 64:
        return False
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def _validated_tool_settings(payload: dict[str, object], legacy_web_search: bool) -> dict[str, bool]:
    """Merge per-message tool preferences with safe defaults and validate types."""
    settings = dict(DEFAULT_TOOL_SETTINGS)
    settings["web_search"] = legacy_web_search
    supplied = payload.get("enabled_tools")
    if supplied is None:
        return settings
    if not isinstance(supplied, dict):
        raise ValueError("enabled_tools must be an object of tool names and true/false values.")
    unknown = set(supplied) - set(TOOL_NAMES)
    if unknown:
        raise ValueError(f"Unknown tool setting: {sorted(unknown)[0]}.")
    for name, enabled in supplied.items():
        if not isinstance(enabled, bool):
            raise ValueError(f"Tool setting '{name}' must be true or false.")
        settings[name] = enabled
    return settings


class TinyAIRequestHandler(BaseHTTPRequestHandler):
    server_version = "TinyAI/1.0"

    def log_message(self, format_string: str, *args: object) -> None:
        # Keep routine request logs compact for local use and hosting logs.
        print(f"[TinyAI] {self.address_string()} - {format_string % args}")

    def _send_bytes(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate" )
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, status: int, data: dict[str, object]) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, payload, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path == "/api/health":
            with ENGINE_LOCK:
                count = len(ENGINE.entries)
            self._send_json(200, {
                "status": "ok",
                "name": "Tiny AI",
                "knowledge_entries": count,
                "teaching_enabled": ALLOW_TEACHING,
                "web_search_enabled": WEB_SEARCH.available,
                "web_search_provider": "DDGS web + dedicated news search (no API key)" if WEB_SEARCH.available else None,
                "nlp_engine": "custom-rule-based",
                "reasoning_engine": "custom-bm25-evidence-reasoner",
                "neural_language_model": {
                    "architecture": "rope-rmsnorm-gqa-swiglu-decoder-v1",
                    "implementation": "custom-python-numpy-autograd",
                    "checkpoint_available": NEURAL_MODEL_CHECKPOINT.is_file(),
                    "served_by_chat": False,
                    "training_command": "py \"AI Experiments/Mini GPT/Train Mini GPT.py\" --preset small --steps 1500",
                },
                "reasoning_features": ["bm25-evidence-ranking", "compound-question-decomposition", "topic-inheritance", "evidence-diversity", "extractive-answer-composition", "explicit-taxonomy-relations", "shared-document-passage-ranking"],
                "native_tools": ["calculator", "unit_converter", "date_time", "date_calculator", "scientific_math", "text_analyzer", "json_tools", "web_search", "document_reader", "memory_context"],
                "available_tools": {
                    "calculator": True,
                    "unit_converter": True,
                    "date_time": True,
                    "date_calculator": True,
                    "scientific_math": True,
                    "text_analyzer": True,
                    "json_tools": True,
                    "web_search": WEB_SEARCH.available,
                    "document_reader": True,
                    "memory_context": ALLOW_MEMORY,
                },
                "date_time_tool": True,
                "unit_converter": True,
                "document_upload_enabled": ALLOW_DOCUMENT_UPLOAD,
                "documents_indexed": len(DOCUMENT_ENGINE.documents),
                "document_formats": ["pdf", "docx", "txt", "md", "csv", "html", "htm"],
                "document_engine": "custom-extractive-retrieval",
                "memory_enabled": ALLOW_MEMORY,
                "memory_engine": "custom-persistent-profile-and-conversation-context",
                "memory_features": ["explicit-memory-commands", "conservative-profile-capture", "per-user-memory-isolation", "per-conversation-history", "pronoun-resolution", "elliptical-follow-ups", "conversation-recap", "memory-deletion"],
                "nlp_features": ["normalization", "tokenization", "light-lemmatization", "typo-correction", "intent-detection", "keyword-extraction", "phrase-matching", "question-focus-extraction", "concept-expansion", "focus-aware-retrieval", "character-ngram-matching", "question-planning", "document-aware-retrieval", "dynamic-knowledge-inventory", "conversation-context-resolution", "persistent-memory-retrieval", "bm25-evidence-ranking", "compound-question-decomposition", "evidence-grounded-composition"],
            })
            return

        if path in {"/api/conversation", "/api/memory"}:
            if not ALLOW_MEMORY:
                self._send_json(403, {"error": "Memory is disabled on this hosted instance."})
                return
            query = parse_qs(urlparse(self.path).query)
            user_id = (query.get("user_id") or [""])[0]
            conversation_id = (query.get("conversation_id") or [""])[0]
            if not _valid_client_id(user_id) or (path == "/api/conversation" and not _valid_client_id(conversation_id)):
                self._send_json(400, {"error": "A valid user_id and conversation_id are required."})
                return
            with MEMORY_LOCK:
                stats = MEMORY_CONTEXT.stats(user_id)
                if path == "/api/conversation":
                    self._send_json(200, {
                        "messages": MEMORY_CONTEXT.get_history(user_id, conversation_id),
                        "memory_count": stats["memory_count"],
                        "conversation_id": conversation_id,
                    })
                else:
                    self._send_json(200, {"memories": MEMORY_CONTEXT.list_memories(user_id), **stats})
            return

        allowed_files = {
            "/": WEB_DIR / "index.html",
            "/index.html": WEB_DIR / "index.html",
            "/styles.css": WEB_DIR / "styles.css",
            "/app.js": WEB_DIR / "app.js",
        }
        file_path = allowed_files.get(path)
        if file_path is None or not file_path.is_file():
            self._send_json(404, {"error": "Not found."})
            return
        self._send_bytes(200, file_path.read_bytes(), CONTENT_TYPES[file_path.suffix])

    def _handle_document_upload(self) -> None:
        if not ALLOW_DOCUMENT_UPLOAD:
            self._send_json(403, {"error": "Document uploads are disabled on this hosted instance. For private documents, run the app locally or explicitly enable uploads on a protected deployment."})
            return
        try:
            body_size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid upload length."})
            return
        if body_size < 1 or body_size > MAX_TOTAL_UPLOAD_BYTES:
            self._send_json(413 if body_size > MAX_TOTAL_UPLOAD_BYTES else 400, {"error": "Upload is empty or exceeds the 26 MB total request limit."})
            return
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            self._send_json(400, {"error": "Documents must be uploaded as multipart/form-data."})
            return
        try:
            raw = self.rfile.read(body_size)
            header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
            message = BytesParser(policy=email_policy).parsebytes(header + raw)
            if not message.is_multipart():
                raise ValueError("The multipart upload could not be read.")
            file_parts = []
            form_fields: dict[str, str] = {}
            for part in message.iter_parts():
                if part.get_content_disposition() != "form-data":
                    continue
                field_name = part.get_param("name", header="content-disposition")
                if field_name in {"files", "file"} and part.get_filename():
                    file_parts.append(part)
                elif field_name in {"user_id", "conversation_id", "enabled_tools"}:
                    field_value = part.get_content()
                    form_fields[field_name] = str(field_value).strip() if field_value is not None else ""
            raw_tool_settings = form_fields.get("enabled_tools")
            if raw_tool_settings:
                try:
                    parsed_upload_settings = json.loads(raw_tool_settings)
                except json.JSONDecodeError as exc:
                    raise ValueError("The upload tool settings are invalid.") from exc
                if not isinstance(parsed_upload_settings, dict):
                    raise ValueError("The upload tool settings are invalid.")
                upload_tool_settings = _validated_tool_settings({"enabled_tools": parsed_upload_settings}, False)
            else:
                upload_tool_settings = dict(DEFAULT_TOOL_SETTINGS)
            if not upload_tool_settings["document_reader"]:
                raise PermissionError("Document Reader is disabled in the chat Tools menu.")
            upload_memory_enabled = ALLOW_MEMORY and upload_tool_settings["memory_context"]
            upload_user_id = form_fields.get("user_id")
            upload_conversation_id = form_fields.get("conversation_id")
            if ALLOW_MEMORY and upload_user_id and not _valid_client_id(upload_user_id):
                raise ValueError("The user identifier is invalid.")
            if ALLOW_MEMORY and upload_conversation_id and not _valid_client_id(upload_conversation_id):
                raise ValueError("The conversation identifier is invalid.")
        except PermissionError as exc:
            self._send_json(403, {"error": str(exc)})
            return
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"error": str(exc) or "The upload form is invalid."})
            return
        if not file_parts:
            self._send_json(400, {"error": "Choose at least one document to upload."})
            return
        if len(file_parts) > MAX_FILES_PER_UPLOAD:
            self._send_json(413, {"error": "Upload up to five documents at a time."})
            return

        uploaded: list[dict[str, object]] = []
        errors: list[dict[str, str]] = []
        with DOCUMENT_LOCK:
            for part in file_parts:
                filename = part.get_filename() or "document"
                payload = part.get_payload(decode=True) or b""
                if len(payload) > MAX_UPLOAD_BYTES:
                    errors.append({"filename": Path(filename).name, "error": "Each document must be 15 MB or smaller."})
                    continue
                try:
                    record = DOCUMENT_ENGINE.ingest(filename, payload)
                    uploaded.append(record.to_public_dict())
                except DocumentUploadError as exc:
                    errors.append({"filename": Path(filename).name, "error": str(exc)})
                except OSError as exc:
                    errors.append({"filename": Path(filename).name, "error": "The document could not be saved to the local knowledge store."})
                    print(f"[TinyAI] Document save failed: {type(exc).__name__}: {exc}")

        if not uploaded:
            message_text = errors[0]["error"] if errors else "No document could be processed."
            self._send_json(422, {"error": message_text, "errors": errors})
            return
        lines = []
        for document in uploaded:
            page_text = f", {document['page_count']} pages" if document.get("page_count") else ""
            lines.append(
                f"• {document['filename']}: extracted {document['word_count']} words, "
                f"created {document['chunk_count']} searchable sections{page_text}."
            )
        answer = "I've added this document content to my local knowledge and search index:\n\n" + "\n".join(lines)
        if errors:
            answer += "\n\nSome files could not be added:\n" + "\n".join(f"• {item['filename']}: {item['error']}" for item in errors)
        answer += "\n\nYou can now ask me questions about the uploaded content or ask me to summarize it."
        response_data: dict[str, object] = {"answer": answer, "documents": uploaded, "errors": errors, "source": "document_upload"}
        if upload_memory_enabled and upload_user_id and upload_conversation_id:
            with MEMORY_LOCK:
                upload_summary = "Uploaded document: " + ", ".join(str(item["filename"]) for item in uploaded)
                MEMORY_CONTEXT.append_turn(upload_user_id, upload_conversation_id, "user", upload_summary)
                MEMORY_CONTEXT.append_turn(upload_user_id, upload_conversation_id, "assistant", answer)
                response_data.update({
                    "conversation_id": upload_conversation_id,
                    "memory_enabled": True,
                    "memory_count": MEMORY_CONTEXT.stats(upload_user_id)["memory_count"],
                })
        self._send_json(200, response_data)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        request_path = urlparse(self.path).path
        if request_path == "/api/upload":
            self._handle_document_upload()
            return
        if request_path != "/api/chat":
            self._send_json(404, {"error": "Not found."})
            return

        try:
            body_size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid request length."})
            return
        if body_size < 1 or body_size > MAX_BODY_BYTES:
            self._send_json(413 if body_size > MAX_BODY_BYTES else 400, {"error": "Message is empty or too large."})
            return

        try:
            payload = json.loads(self.rfile.read(body_size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "Request must contain valid JSON."})
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("message"), str):
            self._send_json(400, {"error": "Send a message as a string."})
            return
        web_search_requested = payload.get("web_search", False)
        if not isinstance(web_search_requested, bool):
            self._send_json(400, {"error": "web_search must be true or false."})
            return
        try:
            tool_settings = _validated_tool_settings(payload, web_search_requested)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        web_search_requested = tool_settings["web_search"]
        memory_enabled_for_request = ALLOW_MEMORY and tool_settings["memory_context"]

        client_timezone = payload.get("timezone")
        if client_timezone is not None and (not isinstance(client_timezone, str) or len(client_timezone) > 100):
            self._send_json(400, {"error": "timezone must be a valid timezone name string."})
            return

        user_id = payload.get("user_id")
        conversation_id = payload.get("conversation_id")
        if user_id is not None and not _valid_client_id(user_id):
            self._send_json(400, {"error": "user_id must be a UUID generated by the chat client."})
            return
        if conversation_id is not None and not _valid_client_id(conversation_id):
            self._send_json(400, {"error": "conversation_id must be a UUID generated by the chat client."})
            return
        # Older clients remain functional, but only the browser's persistent IDs
        # enable cross-message and cross-conversation memory.
        user_id = user_id or str(uuid.uuid4())
        conversation_id = conversation_id or str(uuid.uuid4())

        message = payload["message"].strip()
        if not message:
            self._send_json(400, {"error": "Please type a question."})
            return
        if len(message) > 1_000:
            self._send_json(413, {"error": "Please keep messages under 1,000 characters."})
            return

        history: list[dict[str, str]] = []
        context_used = False
        context_topic: str | None = None
        context_source: str | None = None
        memory_saved: dict[str, object] | None = None
        effective_question = message
        if memory_enabled_for_request:
            with MEMORY_LOCK:
                history = MEMORY_CONTEXT.get_history(user_id, conversation_id, limit=40)
                memory_command = MEMORY_CONTEXT.handle_memory_command(user_id, message)
                if memory_command is not None:
                    MEMORY_CONTEXT.append_turn(user_id, conversation_id, "user", message)
                    MEMORY_CONTEXT.append_turn(user_id, conversation_id, "assistant", str(memory_command["answer"]))
                    memory_command.update({"conversation_id": conversation_id, "memory_enabled": True})
                    self._send_json(200, memory_command)
                    return
                history_answer = MEMORY_CONTEXT.answer_history_question(message, history)
                if history_answer is not None:
                    MEMORY_CONTEXT.append_turn(user_id, conversation_id, "user", message)
                    MEMORY_CONTEXT.append_turn(user_id, conversation_id, "assistant", history_answer)
                    self._send_json(200, {
                        "answer": history_answer, "source": "conversation", "known": True,
                        "context_used": True, "conversation_id": conversation_id,
                        "memory_enabled": True, "memory_count": MEMORY_CONTEXT.stats(user_id)["memory_count"],
                    })
                    return
                profile_memories = MEMORY_CONTEXT.list_memories(user_id)
                resolution = MEMORY_CONTEXT.resolve_follow_up(message, history, memories=profile_memories)
                effective_question = str(resolution["query"])
                context_used = bool(resolution["context_used"])
                context_topic = str(resolution["topic"]) if resolution.get("topic") else None
                context_source = str(resolution["context_source"]) if resolution.get("context_source") else None
                memory_saved = MEMORY_CONTEXT.append_turn(user_id, conversation_id, "user", message)

        def send_chat_response(data: dict[str, object], status: int = 200) -> None:
            if status == 200 and memory_enabled_for_request:
                answer_text = data.get("answer")
                if isinstance(answer_text, str):
                    with MEMORY_LOCK:
                        MEMORY_CONTEXT.append_turn(user_id, conversation_id, "assistant", answer_text)
                        stats = MEMORY_CONTEXT.stats(user_id)
                    data.setdefault("conversation_id", conversation_id)
                    data.setdefault("memory_enabled", True)
                    data.setdefault("memory_count", stats["memory_count"])
                    data.setdefault("context_used", context_used)
                    if context_topic:
                        data.setdefault("context_topic", context_topic)
                    if context_source:
                        data.setdefault("context_source", context_source)
                    if memory_saved:
                        data.setdefault("memory_saved", True)
            self._send_json(status, data)

        # Teaching is supported by the same local engine used in terminal mode.
        if message.lower().startswith("teach:"):
            if not ALLOW_TEACHING:
                self._send_json(403, {"error": "Teaching is disabled on this hosted instance. Add facts to AI Experiments/Tiny AI Engine/Knowledge.json in GitHub and redeploy."})
                return
            teaching = message[6:].strip()
            if "=>" not in teaching:
                self._send_json(400, {"error": "Use: teach: question => answer"})
                return
            question, answer = (part.strip() for part in teaching.split("=>", 1))
            if not question or not answer or len(question) > 300 or len(answer) > 2_000:
                self._send_json(400, {"error": "Please provide a question (max 300 characters) and answer (max 2,000 characters)."})
                return
            try:
                with ENGINE_LOCK:
                    ENGINE.add_knowledge(question, answer, save=True)
                send_chat_response({
                    "answer": "I've saved that question and answer in my knowledge file. Ask me about it now.",
                    "source": "learning",
                    "known": True,
                })
            except (OSError, ValueError) as exc:
                self._send_json(500, {"error": f"I couldn't save that information: {exc}"})
            return

        explicit_query = extract_explicit_search_query(effective_question)
        if explicit_query is not None and not web_search_requested:
            send_chat_response({
                "answer": "Web Search is off. Open the Tools menu beside the message box, enable Web Search, then send your request again.",
                "source": "web_search_disabled",
                "known": False,
                "sources": [],
                "web_search": False,
            })
            return

        question = explicit_query or effective_question

        # Native tools run before knowledge retrieval. The browser supplies its
        # timezone so hosted servers report the visitor's local time, not UTC.
        date_time_result = DATE_TIME_TOOL.execute(question, client_timezone=client_timezone) if tool_settings["date_time"] else None
        if date_time_result is not None:
            send_chat_response({
                **date_time_result.to_dict(),
                "known": True,
                "matched_question": None,
                "confidence": 1.0,
                "sources": [],
                "web_search": False,
            })
            return

        date_calculation = DATE_CALCULATOR.execute(question) if tool_settings["date_calculator"] else None
        if date_calculation is not None:
            send_chat_response({**date_calculation.to_dict(), "known": True, "confidence": 1.0, "sources": [], "web_search": False})
            return

        unit_conversion = UNIT_CONVERTER.execute(question) if tool_settings["unit_converter"] else None
        if unit_conversion is not None:
            send_chat_response({
                **unit_conversion.to_dict(),
                "known": True,
                "confidence": 1.0,
                "sources": [],
                "web_search": False,
            })
            return

        scientific_result = SCIENTIFIC_MATH.execute(question) if tool_settings["scientific_math"] else None
        if scientific_result is not None:
            send_chat_response({**scientific_result.to_dict(), "known": True, "confidence": 1.0, "sources": [], "web_search": False})
            return

        text_result = TEXT_ANALYZER.execute(question) if tool_settings["text_analyzer"] else None
        if text_result is not None:
            send_chat_response({**text_result.to_dict(), "known": True, "confidence": 1.0, "sources": [], "web_search": False})
            return

        json_result = JSON_TOOLS.execute(question) if tool_settings["json_tools"] else None
        if json_result is not None:
            send_chat_response({**json_result.to_dict(), "known": True, "confidence": 1.0, "sources": [], "web_search": False})
            return

        with ENGINE_LOCK:
            local_response = ENGINE.answer(question, allow_calculator=tool_settings["calculator"])

        # Documents become a second local knowledge source. Explicit references
        # to an uploaded file take priority; otherwise document retrieval is used
        # when the curated Q&A knowledge base cannot answer confidently.
        explicit_document_context = bool(re.search(
            r"\b(document|file|pdf|docx|report|attachment|uploaded|according to|in this text)\b",
            question,
            flags=re.IGNORECASE,
        ))
        if tool_settings["document_reader"] and (explicit_document_context or (not local_response.known and not web_search_requested)):
            with DOCUMENT_LOCK:
                document_response = DOCUMENT_ENGINE.answer(question)
            if document_response is not None:
                send_chat_response({
                    "answer": document_response.answer,
                    "source": "document",
                    "known": True,
                    "confidence": document_response.confidence,
                    "document_ids": list(document_response.document_ids),
                    "sources": list(document_response.sources),
                    "web_search": False,
                })
                return

        if local_response.source == "capabilities" and tool_settings["document_reader"]:
            with DOCUMENT_LOCK:
                documents = DOCUMENT_ENGINE.list_documents()
            if documents:
                document_lines = ["", "Uploaded document knowledge:"]
                document_lines.extend(
                    f"• {item['filename']} ({item['chunk_count']} searchable sections)"
                    for item in documents
                )
                local_response = type(local_response)(
                    answer=local_response.answer + "\n" + "\n".join(document_lines),
                    confidence=local_response.confidence,
                    matched_question=local_response.matched_question,
                    source=local_response.source,
                    known=local_response.known,
                )

        # Web search is strictly opt-in. Unknown questions and freshness words
        # never trigger external requests unless the user enables the toggle.
        if web_search_requested and local_response.source not in {"capabilities", "tool"}:
            if not WEB_SEARCH.available:
                send_chat_response({
                    "answer": "The free web-search package is not installed. Run `py -m pip install -r requirements.txt` and restart the app.",
                    "source": "search_setup",
                    "known": False,
                    "sources": [],
                    "web_search": True,
                })
                return
            if not _allow_search_request(self.client_address[0]):
                send_chat_response({
                    "answer": "Web search is being used too frequently from this connection. Please wait a minute and try again.",
                    "source": "search_rate_limited",
                    "known": False,
                    "sources": [],
                    "web_search": True,
                })
                return
            try:
                news_topic = build_news_query(question)
                if news_topic:
                    web_answer = WEB_SEARCH.answer_news(news_topic, timelimit="d")
                else:
                    web_answer = WEB_SEARCH.answer(question)
                send_chat_response({
                    "answer": web_answer.answer,
                    "source": "web",
                    "search_type": "news" if news_topic else "web",
                    "known": bool(web_answer.sources),
                    "matched_question": None,
                    "search_query": web_answer.query,
                    "sources": [item.to_public_dict() for item in web_answer.sources],
                    "web_search": True,
                })
                return
            except SearchRateLimited:
                send_chat_response({
                    "answer": "Free search sources are temporarily limiting requests. Please wait a little and try again.",
                    "source": "search_rate_limited",
                    "known": False,
                    "sources": [],
                    "web_search": True,
                })
                return
            except (SearchProviderUnavailable, ValueError) as exc:
                send_chat_response({
                    "answer": "I couldn't complete the web search just now. Please try again in a moment.",
                    "source": "web_error",
                    "known": False,
                    "sources": [],
                    "web_search": True,
                })
                print(f"[TinyAI] Web search failed: {type(exc).__name__}: {exc}")
                return

        send_chat_response({
            "answer": local_response.answer,
            "source": local_response.source,
            "known": local_response.known,
            "matched_question": local_response.matched_question,
            "confidence": local_response.confidence,
            "sources": [],
            "web_search": False,
        })


def create_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), TinyAIRequestHandler)


def main() -> None:
    # Keep the memory-enabled local app private to this computer. Hosting
    # platforms provide PORT and require the service to listen on all interfaces.
    host = "0.0.0.0" if HOSTED_ENVIRONMENT else "127.0.0.1"
    port = int(os.environ.get("PORT", "8000"))
    server = create_server(host, port)
    local_url = f"http://127.0.0.1:{port}"
    print(f"Chat interface: {local_url}")
    print("Press Ctrl+C to stop the server.")
    # Open the browser automatically for local runs. Hosted environments set RENDER or PORT.
    if not os.environ.get("RENDER") and not os.environ.get("PORT"):
        threading.Timer(0.8, lambda: webbrowser.open(local_url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Tiny AI...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
