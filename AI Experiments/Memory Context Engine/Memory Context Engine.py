"""Custom persistent memory and multi-turn conversation context for Tiny AI.

The subsystem is deliberately implemented with the Python standard library and
this project's own CustomNLP. It stores user-approved/low-risk profile facts and
separate per-conversation message histories in one local JSON database.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


_NLP_DIR = Path(__file__).resolve().parents[1] / "Custom NLP"
import sys
if str(_NLP_DIR) not in sys.path:
    sys.path.insert(0, str(_NLP_DIR))
from nlp_engine import CustomNLP  # noqa: E402

DEFAULT_MEMORY_PATH = Path(__file__).resolve().parents[2] / "Data" / "Memory" / "Memory Context.json"
MAX_FACT_LENGTH = 500
MAX_MESSAGE_LENGTH = 4_000


class MemoryContextError(ValueError):
    """Raised when a memory or conversation operation is invalid."""


class MemoryContextEngine:
    """Persistent user memory plus bounded multi-turn context resolution."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        max_facts: int = 100,
        max_turns: int = 60,
        max_conversations: int = 40,
    ) -> None:
        self.path = Path(path) if path else DEFAULT_MEMORY_PATH
        self.max_facts = max(1, int(max_facts))
        self.max_turns = max(4, int(max_turns))
        self.max_conversations = max(1, int(max_conversations))
        self._lock = threading.RLock()
        self._data = self._load()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _empty_data() -> dict[str, Any]:
        return {"version": 1, "users": {}}

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty_data()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MemoryContextError(f"Could not read the memory database: {exc}") from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("users", {}), dict):
            raise MemoryContextError("The memory database has an invalid structure.")
        raw.setdefault("version", 1)
        raw.setdefault("users", {})
        # Repair malformed user records conservatively instead of discarding data.
        for user in raw["users"].values():
            if not isinstance(user, dict):
                continue
            if not isinstance(user.get("facts"), list):
                user["facts"] = []
            if not isinstance(user.get("conversations"), dict):
                user["conversations"] = {}
        return raw

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self._data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def _user(self, user_id: str, *, create: bool = True) -> dict[str, Any]:
        if not isinstance(user_id, str) or not user_id or len(user_id) > 100:
            raise MemoryContextError("A valid user identifier is required.")
        users = self._data["users"]
        if user_id not in users:
            if not create:
                return {"facts": [], "conversations": {}}
            users[user_id] = {"facts": [], "conversations": {}}
        user = users[user_id]
        user.setdefault("facts", [])
        user.setdefault("conversations", {})
        return user

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        return set(CustomNLP().analyze(text).expanded_keywords)

    @staticmethod
    def _clean_fact(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" \t\r\n.,;:-")
        if not cleaned:
            raise MemoryContextError("Tell me what you want me to remember.")
        if len(cleaned) > MAX_FACT_LENGTH:
            raise MemoryContextError(f"A memory must be {MAX_FACT_LENGTH} characters or fewer.")
        return cleaned

    def remember(self, user_id: str, text: str, *, category: str = "note", source: str = "explicit") -> dict[str, Any]:
        """Add or refresh one user memory, de-duplicating by normalized wording."""
        fact_text = self._clean_fact(text)
        category = category if category in {"profile", "preference", "project", "note"} else "note"
        normalized = CustomNLP.normalize(fact_text)
        with self._lock:
            user = self._user(user_id)
            existing = next((item for item in user["facts"] if CustomNLP.normalize(str(item.get("text", ""))) == normalized), None)
            now = self._now()
            if existing:
                existing["updated_at"] = now
                existing["recall_count"] = int(existing.get("recall_count", 0)) + 1
                result = dict(existing)
            else:
                result = {
                    "id": uuid.uuid4().hex[:12],
                    "text": fact_text,
                    "category": category,
                    "source": source,
                    "created_at": now,
                    "updated_at": now,
                    "recall_count": 0,
                }
                user["facts"].append(result)
                user["facts"] = user["facts"][-self.max_facts:]
            self._save()
            return dict(result)

    def _auto_memory_candidate(self, message: str) -> tuple[str, str] | None:
        """Extract only stable, first-person profile/preferences/project statements."""
        value = re.sub(r"\s+", " ", message).strip()
        patterns: tuple[tuple[str, str], ...] = (
            ("profile", r"^(?:my name is|call me)\s+(.{1,120})[.!]?$"),
            ("preference", r"^(?:i prefer|i usually prefer|i like|i love|my preferred [a-z ]+ is)\s+(.{1,220})[.!]?$"),
            ("preference", r"^(?:please always|from now on,? please|i want you to always)\s+(.{1,220})[.!]?$"),
            ("project", r"^(?:i am|i'm|we are|we're)\s+(?:building|developing|creating|working on|making)\s+(.{1,220})[.!]?$"),
            ("project", r"^(?:my project is|the project i'm working on is)\s+(.{1,220})[.!]?$"),
            ("profile", r"^(?:i work as|my role is)\s+(.{1,120})[.!]?$"),
        )
        for category, pattern in patterns:
            match = re.match(pattern, value, re.IGNORECASE)
            if match:
                captured = match.group(1).strip(" .,!?")
                if not captured or CustomNLP.normalize(captured) in {"it", "this", "that", "something", "stuff", "one", "things"}:
                    return None
                if category == "profile" and re.match(r"^(?:my name is|call me)", value, re.I):
                    fact = f"The user's name is {captured}."
                elif category == "preference":
                    prefix = "The user prefers" if re.match(r"^(?:i prefer|i usually prefer|my preferred)", value, re.I) else "The user likes"
                    if re.match(r"^(?:please always|from now on|i want you to always)", value, re.I):
                        fact = f"The user requests: {captured}."
                    else:
                        fact = f"{prefix} {captured}."
                elif category == "project":
                    fact = f"The user is working on {captured}."
                else:
                    fact = f"The user's role is {captured}."
                return category, fact
        return None

    def auto_remember(self, user_id: str, message: str) -> dict[str, Any] | None:
        candidate = self._auto_memory_candidate(message)
        if candidate is None:
            return None
        category, text = candidate
        return self.remember(user_id, text, category=category, source="automatic-profile")

    def list_memories(self, user_id: str) -> list[dict[str, Any]]:
        with self._lock:
            user = self._user(user_id, create=False)
            return [dict(item) for item in user.get("facts", []) if isinstance(item, dict)]

    def forget(self, user_id: str, target: str | None = None) -> int:
        """Delete all memories or only entries matching the supplied phrase."""
        with self._lock:
            user = self._user(user_id)
            facts = user["facts"]
            if not target:
                removed = len(facts)
                user["facts"] = []
            elif CustomNLP.normalize(target) in {"that", "it", "the last one", "the last memory"}:
                removed = 1 if facts else 0
                if facts:
                    facts.pop()
            else:
                wanted = self._content_tokens(target)
                scored: list[tuple[float, int]] = []
                for index, item in enumerate(facts):
                    available = self._content_tokens(str(item.get("text", "")))
                    overlap = len(wanted & available) / max(1, len(wanted))
                    if overlap >= 0.34:
                        scored.append((overlap, index))
                indices = {index for _, index in scored}
                removed = len(indices)
                user["facts"] = [item for index, item in enumerate(facts) if index not in indices]
            if removed:
                self._save()
            return removed

    def handle_memory_command(self, user_id: str, message: str) -> dict[str, Any] | None:
        """Recognize explicit remember/list/forget commands without web search."""
        normalized = CustomNLP.normalize(message).strip(" .!?\t")
        facts = self.list_memories(user_id)
        if re.search(r"\b(what do you remember about me|what have you remembered|show (?:me )?(?:my )?memories|list (?:my )?memories|what do you know about me)\b", normalized):
            if not facts:
                answer = "I don't have any saved memories about you yet. Tell me something worth remembering, for example: ‘Remember that I prefer concise answers.’"
            else:
                lines = ["Here is what I currently remember:"]
                lines.extend(f"• {item['text']}" for item in facts)
                lines.append("\nYou can say ‘forget [a detail]’ or ‘forget everything you remember about me’ to remove memories.")
                answer = "\n".join(lines)
            return {"answer": answer, "source": "memory", "known": True, "memory_count": len(facts)}

        recall_category = None
        if re.search(r"\b(what is my name|what's my name|who am i)\b", normalized):
            recall_category = "profile"
        elif re.search(r"\b(what project am i working on|what am i working on|what am i building|what project did i tell you about|what do you remember about my project)\b", normalized):
            recall_category = "project"
        elif re.search(r"\b(what are my preferences|what do i prefer|what do i like)\b", normalized):
            recall_category = "preference"
        if recall_category:
            matching = [item for item in facts if item.get("category") == recall_category]
            if recall_category == "profile":
                matching = [item for item in facts if "name is" in str(item.get("text", "")).lower()]
            answer = "\n".join(["Here is what I remember:", *[f"• {item['text']}" for item in matching]]) if matching else "I don't have that saved yet. You can tell me directly or say ‘remember that ...’."
            return {"answer": answer, "source": "memory", "known": bool(matching), "memory_count": len(facts)}

        if re.search(r"\b(forget everything|forget all memories|clear all memory|delete all memories|forget what you know about me)\b", normalized):
            removed = self.forget(user_id)
            return {"answer": f"I've deleted {removed} saved memor{'y' if removed == 1 else 'ies'}. Your conversation history is separate and has not been deleted.", "source": "memory", "known": True, "memory_count": 0}

        forget_match = re.match(r"^(?:please\s+)?forget(?:\s+(?:that|it))?(?:\s+(.+))?$", message.strip(), re.IGNORECASE)
        if forget_match and not re.match(r"^(?:everything|all|what you know)", (forget_match.group(1) or ""), re.I):
            target = (forget_match.group(1) or "that").strip(" .!?\t")
            removed = self.forget(user_id, target or None)
            if removed:
                answer = f"I've removed {removed} matching memor{'y' if removed == 1 else 'ies'}."
            else:
                answer = "I couldn't find a saved memory matching that description."
            return {"answer": answer, "source": "memory", "known": True, "memory_count": len(self.list_memories(user_id))}

        remember_match = re.match(r"^(?:please\s+)?(?:i want you to )?remember(?:\s+that)?\s*[:,-]?\s*(.+)$", message.strip(), re.IGNORECASE)
        if remember_match:
            value = remember_match.group(1).strip()
            # Normalize recognizable preferences/profile/project statements into
            # useful memory categories; keep other explicit notes verbatim.
            candidate = self._auto_memory_candidate(value)
            category, stored_value = candidate if candidate else ("note", value)
            fact = self.remember(user_id, stored_value, category=category, source="explicit")
            return {"answer": f"I'll remember this: {fact['text']}", "source": "memory", "known": True, "memory_count": len(self.list_memories(user_id)), "memory_saved": True}
        return None

    def append_turn(self, user_id: str, conversation_id: str, role: str, content: str) -> dict[str, Any] | None:
        if role not in {"user", "assistant"}:
            raise MemoryContextError("Conversation role must be user or assistant.")
        content = str(content or "").strip()
        if not content:
            return None
        content = content[:MAX_MESSAGE_LENGTH]
        with self._lock:
            user = self._user(user_id)
            conversations = user["conversations"]
            conversation = conversations.setdefault(conversation_id, {"messages": [], "updated_at": self._now()})
            if not isinstance(conversation, dict):
                conversation = {"messages": [], "updated_at": self._now()}
                conversations[conversation_id] = conversation
            messages = conversation.setdefault("messages", [])
            messages.append({"role": role, "content": content, "timestamp": self._now()})
            conversation["messages"] = messages[-self.max_turns:]
            conversation["updated_at"] = self._now()
            # Bound the number of stored chats by least-recently-updated time.
            if len(conversations) > self.max_conversations:
                oldest = sorted(conversations, key=lambda key: conversations[key].get("updated_at", ""))
                for old_id in oldest[:len(conversations) - self.max_conversations]:
                    if old_id != conversation_id:
                        conversations.pop(old_id, None)
            self._save()
        if role == "user":
            return self.auto_remember(user_id, content)
        return None

    def get_history(self, user_id: str, conversation_id: str, *, limit: int | None = None) -> list[dict[str, str]]:
        with self._lock:
            user = self._user(user_id, create=False)
            conversation = user.get("conversations", {}).get(conversation_id, {})
            messages = conversation.get("messages", []) if isinstance(conversation, dict) else []
            selected = messages[-limit:] if limit else messages
            return [
                {"role": str(item.get("role", "")), "content": str(item.get("content", "")), "timestamp": str(item.get("timestamp", ""))}
                for item in selected if isinstance(item, dict) and item.get("role") in {"user", "assistant"}
            ]

    @staticmethod
    def _is_follow_up(message: str) -> bool:
        normalized = CustomNLP.normalize(message).strip(" .!?\t")
        if not normalized:
            return False
        # Idiomatic standalone clock questions contain "it" but are not
        # references to an earlier message.
        if normalized in {"what time is it", "what day is it", "what date is it"}:
            return False
        if re.search(r"\b(it|that|this|they|them|those|these)\b", normalized):
            return True
        if re.search(r"\b(?:which|the|another|same)\s+ones?\b|\bsame\b", normalized):
            return True
        if re.match(r"^(?:what about|what if)\b", normalized):
            return True
        if re.fullmatch(r"(?:and\s+)?(?:why|how|when|where|which)(?:\s+(?:many|much))?", normalized):
            return True
        if re.fullmatch(r"(?:and\s+)?how many\s+[a-z]+", normalized):
            return True
        if re.search(r"\b(tell me more|explain more|go on|continue|elaborate|expand on that|give me an example|another example|make it simpler|simplify it|say that again|repeat that)\b", normalized):
            return True
        return normalized in {"why", "how", "when", "where", "what", "which", "more", "again", "same"}

    @staticmethod
    def _topic_from_text(text: str) -> str:
        value = CustomNLP.normalize(text)
        value = re.sub(r"^(?:i am|i'm|we are|we're)\s+(?:building|developing|creating|working on|making)\s+", "", value)
        value = re.sub(r"^(?:my project is|the project i am working on is)\s+", "", value)
        focus = CustomNLP.extract_focus(value)
        focus = re.sub(r"\b(?:please|thanks|thank|again|more|detail|details|example|examples)\b", "", focus)
        focus = re.sub(r"\s+", " ", focus).strip(" .,!?:;-\t")
        tokens = CustomNLP().analyze(focus).keywords
        if not tokens:
            tokens = CustomNLP().analyze(value).keywords
        if len(tokens) > 7:
            # Retain a compact trailing topic phrase after stripping query frames.
            tokens = tokens[-7:]
        return " ".join(tokens)[:100]

    def resolve_follow_up(
        self,
        message: str,
        history: Iterable[dict[str, Any]],
        memories: Iterable[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        """Resolve pronouns and elliptical follow-ups against chat or saved project context."""
        turns = list(history)
        if not self._is_follow_up(message):
            return {"query": message, "context_used": False, "topic": None, "context_source": None}
        previous_user = [str(item.get("content", "")) for item in turns if item.get("role") == "user"]
        topic = ""
        context_source = "conversation"
        # Prefer the nearest root topic, skipping later short contextual follow-ups.
        for candidate in reversed(previous_user[-12:]):
            if candidate.lower().startswith(("upload document:", "upload documents:", "uploaded document:", "uploaded documents:")) or self._is_follow_up(candidate):
                continue
            topic = self._topic_from_text(candidate)
            if topic:
                break
        # In a new chat, a recent saved project can act as a longer-term anchor.
        if not topic:
            project_memories = [item for item in memories if item.get("category") == "project"]
            project_memories.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
            for item in project_memories:
                saved_text = str(item.get("text", ""))
                saved_text = re.sub(r"^the user is working on\s+", "", saved_text, flags=re.IGNORECASE)
                saved_text = re.sub(r"^the project is\s+", "", saved_text, flags=re.IGNORECASE)
                topic = self._topic_from_text(saved_text)
                if topic:
                    context_source = "memory"
                    break
        if not topic:
            return {"query": message, "context_used": False, "topic": None, "context_source": None}

        normalized = CustomNLP.normalize(message).strip()
        if re.search(r"\b(tell me more|explain more|go on|continue|elaborate|expand on that)\b", normalized):
            query = f"Explain {topic} in more detail."
        elif re.search(r"\b(give me an example|another example|example please)\b", normalized):
            query = f"Give an example of {topic}."
        elif re.fullmatch(r"(?:and\s+)?why\??", normalized):
            query = f"Why is {topic} important?"
        elif re.fullmatch(r"(?:and\s+)?how\??", normalized):
            query = f"How does {topic} work?"
        elif re.fullmatch(r"(?:and\s+)?when\??", normalized):
            query = f"When does {topic} happen?"
        elif re.fullmatch(r"(?:and\s+)?where\??", normalized):
            query = f"Where is {topic}?"
        elif re.fullmatch(r"(?:and\s+)?what\??", normalized):
            query = f"What is {topic}?"
        elif re.match(r"^what about\s+(.+)$", normalized):
            remainder = re.sub(r"^what about\s+", "", message, flags=re.IGNORECASE).rstrip(" ?")
            query = f"What about {remainder} for {topic}?"
        elif re.match(r"^(?:how many|how much|what is|when is|where is|which)\b", normalized) and len(normalized.split()) <= 6:
            query = f"{message.rstrip(' ?')} for {topic}?"
        elif len(normalized.split()) <= 5 and not re.search(r"\b(?:it|that|this|they|them|those|these|one|ones)\b", normalized):
            query = f"{message.rstrip(' ?')} for {topic}?"
        else:
            query = re.sub(r"\b(?:it|that|this|they|them|those|these|one|ones)\b", topic, message, flags=re.IGNORECASE)
            query = re.sub(r"\s+", " ", query).strip()
        return {
            "query": query,
            "context_used": True,
            "topic": topic,
            "context_source": context_source,
            "memory_used": context_source == "memory",
        }

    def answer_history_question(self, message: str, history: Iterable[dict[str, Any]]) -> str | None:
        normalized = CustomNLP.normalize(message).strip(" .!?\t")
        turns = list(history)
        if not turns:
            return None
        if re.search(r"\b(repeat that|repeat your last answer|say that again|what did you just say)\b", normalized):
            last_answer = next((str(item.get("content", "")) for item in reversed(turns) if item.get("role") == "assistant"), "")
            return last_answer or None
        if re.search(r"\b(what did i ask|what was my last question|what did i say earlier|what were we discussing|what are we talking about|summarize this conversation|recap our conversation|summarize our discussion)\b", normalized):
            recent = turns[-10:]
            lines = ["Here is the recent conversation:"]
            for item in recent:
                role = "You" if item.get("role") == "user" else "Me"
                content = re.sub(r"\s+", " ", str(item.get("content", ""))).strip()
                if len(content) > 280:
                    content = content[:277] + "..."
                if content:
                    lines.append(f"• {role}: {content}")
            return "\n".join(lines)
        return None

    def relevant_memories(self, user_id: str, message: str, *, limit: int = 4) -> list[dict[str, Any]]:
        """Return memories with meaningful lexical overlap for downstream context."""
        query_tokens = self._content_tokens(message)
        if not query_tokens:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for fact in self.list_memories(user_id):
            fact_tokens = self._content_tokens(str(fact.get("text", "")))
            overlap = len(query_tokens & fact_tokens)
            if overlap:
                score = overlap / max(1, min(len(query_tokens), len(fact_tokens)))
                if score >= 0.2:
                    scored.append((score, fact))
        scored.sort(key=lambda item: (item[0], item[1].get("updated_at", "")), reverse=True)
        return [dict(item) for _, item in scored[:limit]]

    def stats(self, user_id: str) -> dict[str, int]:
        with self._lock:
            user = self._user(user_id, create=False)
            conversations = user.get("conversations", {})
            return {
                "memory_count": len(user.get("facts", [])),
                "conversation_count": len(conversations),
                "message_count": sum(len(item.get("messages", [])) for item in conversations.values() if isinstance(item, dict)),
            }
