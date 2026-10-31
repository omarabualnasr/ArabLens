"""A small, custom question-answering engine built with Python and NumPy.

This is a transparent retrieval-and-reasoning system, not a pretrained language
model. It combines custom NLP, TF-IDF/cosine matching, BM25 evidence ranking,
question decomposition, evidence selection, and safe arithmetic.
"""

from __future__ import annotations

import ast
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


_NLP_DIR = Path(__file__).resolve().parents[1] / "Custom NLP"
if str(_NLP_DIR) not in sys.path:
    sys.path.insert(0, str(_NLP_DIR))
from nlp_engine import CustomNLP  # noqa: E402
_REASONING_DIR = Path(__file__).resolve().parents[1] / "Custom Reasoning Engine"
if str(_REASONING_DIR) not in sys.path:
    sys.path.insert(0, str(_REASONING_DIR))
from reasoning_engine import CustomReasoningEngine, RankedEvidence  # noqa: E402


DEFAULT_KNOWLEDGE_PATH = Path(__file__).resolve().parent / "Knowledge.json"

# Small stopword list: remove question scaffolding while retaining meaningful terms.
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "could", "do",
    "does", "for", "from", "how", "i", "in", "is", "it", "me", "of",
    "on", "or", "please", "the", "their", "there", "these", "this", "to",
    "was", "what", "when", "where", "which", "who", "why", "will", "with",
    "would", "you", "your", "tell", "explain", "define", "about", "mean",
}


@dataclass(frozen=True)
class KnowledgeEntry:
    question: str
    answer: str
    keywords: tuple[str, ...] = ()
    alternate_questions: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "KnowledgeEntry":
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            raise ValueError("Each knowledge entry needs a non-empty question and answer.")
        return cls(
            question=question,
            answer=answer,
            keywords=tuple(str(x).strip() for x in item.get("keywords", []) if str(x).strip()),
            alternate_questions=tuple(
                str(x).strip() for x in item.get("alternate_questions", []) if str(x).strip()
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "keywords": list(self.keywords),
            "alternate_questions": list(self.alternate_questions),
        }


@dataclass(frozen=True)
class EngineResponse:
    answer: str
    confidence: float
    matched_question: str | None = None
    source: str = "knowledge"
    known: bool = True


class _SafeMathEvaluator(ast.NodeVisitor):
    """Evaluate a deliberately small arithmetic language without eval()."""

    _BINARY_OPERATORS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a**b,
        ast.Mod: lambda a, b: a % b,
    }
    _UNARY_OPERATORS = {ast.UAdd: lambda x: x, ast.USub: lambda x: -x}

    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Only numbers are allowed in calculations.")
        value = float(node.value)
        if not math.isfinite(value):
            raise ValueError("The number must be finite.")
        return value

    def visit_BinOp(self, node: ast.BinOp) -> float:
        operation = self._BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("That operator is not supported.")
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise ValueError("Powers above 8 are not supported.")
        result = operation(left, right)
        if not math.isfinite(result) or abs(result) > 1e100:
            raise ValueError("The result is outside the supported range.")
        return result

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        operation = self._UNARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("That unary operator is not supported.")
        return operation(self.visit(node.operand))

    def generic_visit(self, node: ast.AST) -> float:
        raise ValueError("Only basic arithmetic is supported.")


def _normalise_text(text: str) -> str:
    """Compatibility wrapper around the custom NLP normalizer."""
    return CustomNLP.normalize(text)


def _tokens(text: str) -> list[str]:
    """Compatibility wrapper returning custom NLP's content lemmas."""
    return list(CustomNLP().analyze(text).keywords)


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-10):
        return str(int(round(value)))
    return f"{value:.10g}"


class TinyAIEngine:
    """A local QA engine using custom NLP, TF-IDF, BM25 evidence ranking and safe arithmetic."""

    def __init__(
        self,
        knowledge: Iterable[KnowledgeEntry | dict[str, Any]] | None = None,
        *,
        knowledge_path: str | Path | None = None,
        confidence_threshold: float = 0.16,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        self.confidence_threshold = float(confidence_threshold)
        self.knowledge_path = Path(knowledge_path) if knowledge_path else DEFAULT_KNOWLEDGE_PATH
        self.nlp = CustomNLP()
        self.reasoner = CustomReasoningEngine(self.nlp)

        if knowledge is not None:
            loaded_entries = [self._coerce_entry(item) for item in knowledge]
        else:
            loaded_entries = self._load_entries(self.knowledge_path)
        self.entries = self._deduplicate_entries(loaded_entries)
        if not self.entries:
            raise ValueError("The knowledge base is empty. Add at least one question and answer.")

        self._rebuild_index()

    @staticmethod
    def _coerce_entry(item: KnowledgeEntry | dict[str, Any]) -> KnowledgeEntry:
        if isinstance(item, KnowledgeEntry):
            return item
        if isinstance(item, dict):
            return KnowledgeEntry.from_dict(item)
        raise TypeError("Knowledge items must be KnowledgeEntry objects or dictionaries.")

    @staticmethod
    def _load_entries(path: Path) -> list[KnowledgeEntry]:
        if not path.exists():
            raise FileNotFoundError(f"Knowledge file not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid knowledge JSON in {path}: {exc}") from exc
        if not isinstance(payload, list):
            raise ValueError("Knowledge file must contain a JSON list.")
        return [KnowledgeEntry.from_dict(item) for item in payload]

    def _entry_search_text(self, entry: KnowledgeEntry) -> str:
        # Search questions, alternate phrasings and keywords, never answer prose.
        parts = [entry.question, *entry.alternate_questions, *entry.keywords]
        return " ".join(parts)

    @staticmethod
    def _deduplicate_entries(entries: list[KnowledgeEntry]) -> list[KnowledgeEntry]:
        """Remove repeated question/answer rows that otherwise distort IDF."""
        unique: dict[tuple[str, str], KnowledgeEntry] = {}
        for entry in entries:
            key = (CustomNLP.normalize(entry.question), entry.answer.strip())
            unique.setdefault(key, entry)
        return list(unique.values())

    def _rebuild_index(self) -> None:
        raw_documents = [self._entry_search_text(entry) for entry in self.entries]
        raw_analyses = [self.nlp.analyze(text) for text in raw_documents]
        vocabulary = {token for analysis in raw_analyses for token in analysis.expanded_keywords}
        self.nlp.vocabulary.update(vocabulary)
        # Re-analyze using the complete knowledge vocabulary so typo correction
        # can be applied consistently to incoming questions.
        self.reasoner.fit(self.entries)
        self._entry_analyses = [
            {
                "questions": [
                    self.nlp.analyze(variant, vocabulary=vocabulary)
                    for variant in [entry.question, *entry.alternate_questions]
                ],
                "keywords": [
                    self.nlp.analyze(keyword, vocabulary=vocabulary)
                    for keyword in entry.keywords
                ],
            }
            for entry in self.entries
        ]
        self._documents = [list(analysis.expanded_keywords) for analysis in raw_analyses]
        self._vocabulary = sorted({token for document in self._documents for token in document})
        self._term_to_column = {term: index for index, term in enumerate(self._vocabulary)}

        if not self._vocabulary:
            self._matrix = np.zeros((len(self.entries), 0), dtype=np.float64)
            return

        document_count = len(self._documents)
        document_frequency = np.zeros(len(self._vocabulary), dtype=np.float64)
        term_frequency = np.zeros((document_count, len(self._vocabulary)), dtype=np.float64)

        for row, document in enumerate(self._documents):
            for token in document:
                term_frequency[row, self._term_to_column[token]] += 1.0
            for token in set(document):
                document_frequency[self._term_to_column[token]] += 1.0

        # Smooth IDF; log scaling limits rare-term dominance in a small knowledge base.
        inverse_document_frequency = np.log(
            (1.0 + document_count) / (1.0 + document_frequency)
        ) + 1.0
        weighted = term_frequency * inverse_document_frequency
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        self._matrix = weighted / np.maximum(norms, 1e-12)

    def _retrieve_candidates(self, question: str, *, limit: int = 6) -> list[RankedEvidence]:
        query = self.nlp.analyze(question, vocabulary=self._vocabulary)
        query_plan = self.nlp.plan(question, vocabulary=self._vocabulary)
        query_focus = set(query_plan["focus_tokens"])
        query_tokens = list(query.expanded_keywords)
        if not query_tokens or not self._vocabulary:
            return []

        query_vector = np.zeros(len(self._vocabulary), dtype=np.float64)
        for token in query_tokens:
            column = self._term_to_column.get(token)
            if column is not None:
                query_vector[column] += 1.0
        norm = float(np.linalg.norm(query_vector))
        if norm <= 1e-12:
            return []
        query_vector /= norm
        cosine_scores = self._matrix @ query_vector

        base_scores = np.zeros(len(self.entries), dtype=np.float64)
        for index, variants in enumerate(self._entry_analyses):
            question_score = max(
                (self.nlp.similarity(query, candidate) for candidate in variants["questions"]),
                default=0.0,
            )
            keyword_score = max(
                (self.nlp.similarity(query, candidate) for candidate in variants["keywords"]),
                default=0.0,
            )
            phrase_intent_score = 0.90 * question_score + 0.10 * keyword_score
            focus_scores = [
                len(query_focus & set(candidate.focus_tokens)) / max(1, len(query_focus))
                for candidate in variants["questions"]
            ]
            focus_score = max(focus_scores, default=0.0)
            intent_alignment = max((
                1.0 if query_plan["intent"] == candidate.intent else
                0.45 if {query_plan["intent"], candidate.intent} <= {"definition", "explanation", "general_question", "procedure"} else 0.0
                for candidate in variants["questions"]
            ), default=0.0)
            base_scores[index] = (
                0.20 * cosine_scores[index]
                + 0.63 * phrase_intent_score
                + 0.10 * focus_score
                + 0.07 * intent_alignment
            )

        ranked = self.reasoner.rank(question, self.entries, base_scores.tolist())
        return ranked[:max(1, min(20, limit))]

    def _retrieve(self, question: str) -> tuple[int, float] | None:
        """Compatibility API returning the strongest ranked evidence item."""
        ranked = self._retrieve_candidates(question, limit=1)
        if not ranked:
            return None
        return ranked[0].index, ranked[0].score

    @staticmethod
    def _try_calculation(question: str) -> str | None:
        cleaned = question.strip().lower()
        cleaned = re.sub(r"^(what is|calculate|compute|solve|please calculate)\s+", "", cleaned)
        cleaned = cleaned.rstrip("?!. ")
        if not cleaned or not re.fullmatch(r"[0-9+\-*/%().\s]+", cleaned):
            return None
        try:
            tree = ast.parse(cleaned, mode="eval")
            value = _SafeMathEvaluator().visit(tree)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
            return None
        return _format_number(value)

    @staticmethod
    def _knowledge_category(entry: KnowledgeEntry) -> str:
        question = CustomNLP.normalize(entry.question)
        keywords = " ".join(entry.keywords).lower()
        combined = f"{question} {keywords}"
        if any(term in question for term in ("tiny ai engine", "this project", "limitations of this", "programming language is this project")):
            return "About this assistant"
        if "space" in question or "astronomy" in keywords or "outer space" in keywords:
            return "Space & astronomy"
        if any(term in combined for term in (
            "tokenizer", "token", "attention", "transformer", "gpt", "natural language",
            "text normalization", "lemmatization", "intent detection", "keyword extraction",
            "custom nlp", "language model",
        )):
            return "Language, NLP & Transformers"
        if any(term in question for term in (
            "neural network", "neuron", "weight", "bias", "loss function", "gradient descent",
            "backpropagation", "learning rate",
        )):
            return "Neural networks & learning"
        if any(term in question for term in ("python", "numpy", "programming language")):
            return "Python & programming"
        return "AI & machine learning"

    def _knowledge_inventory_response(self) -> EngineResponse:
        """Describe the exact prompts represented by the current local knowledge file."""
        groups: dict[str, list[str]] = {
            "AI & machine learning": [],
            "Neural networks & learning": [],
            "Language, NLP & Transformers": [],
            "Python & programming": [],
            "Space & astronomy": [],
            "About this assistant": [],
        }
        for entry in self.entries:
            category = self._knowledge_category(entry)
            groups.setdefault(category, []).append(entry.question)

        lines = [
            f"I currently have {len(self.entries)} question-and-answer entries in my local knowledge base. "
            "These are the exact questions I can answer from that knowledge:",
        ]
        for category, questions in groups.items():
            if not questions:
                continue
            lines.extend(("", f"{category} ({len(questions)})"))
            lines.extend(f"• {question}" for question in questions)
        lines.extend((
            "",
            "Built-in tools:",
            "• Calculator: safe arithmetic expressions, including parentheses and common operators.",
            "• Unit converter: offline conversions for length, mass, volume, time, speed, area, storage, and temperature.",
            "• Date & time: current local time, today's date, weekday, tomorrow/yesterday, and common city timezones.",
            "• Date calculator: calculate date gaps, weekdays, business days, and add or subtract days, weeks, months, and years.",
            "• Scientific math: evaluate square roots, powers, trigonometry (degrees), logarithms, and solve linear or quadratic equations in x.",
            "• Text analyzer: count words and characters, estimate reading time, identify frequent terms, extract email addresses and URLs, and transform text case or slugs.",
            "• JSON tools: validate, format, minify, and inspect JSON safely.",
            "• Web Search: optional online lookup when enabled in the Tools menu in the chat composer.",
            "• Document reader: upload PDF, Word (DOCX), TXT, Markdown, CSV, or HTML files, then ask questions or request summaries from their extracted text.",
            "• Memory: remember explicitly saved facts, stable preferences, and project details across separate chats; ask 'What do you remember about me?' or say 'forget ...' to manage it.",
            "• Conversation context: connect follow-up questions to earlier messages, resolve references such as 'it' and 'that', repeat the previous answer, and recap the current conversation.",
            "• Custom reasoning: decompose multi-part questions, rank multiple evidence entries with BM25 plus the custom TF-IDF/NLP matcher, and compose evidence-grounded summaries.",
            "",
            "You can copy any question above or ask it in your own words. "
            "This list reflects my local knowledge exactly; it does not mean I know every subject. "
            "Turn on Web Search when you want me to look up something outside this list.",
        ))
        return EngineResponse(
            answer="\n".join(lines),
            confidence=1.0,
            source="capabilities",
            known=True,
        )

    @staticmethod
    def _handle_small_talk(question: str) -> EngineResponse | None:
        analysis = CustomNLP().analyze(question)
        normalised = re.sub(r"[^a-z0-9 ]", "", analysis.normalized).strip()
        tokens = set(analysis.lemmas)
        if analysis.intent == "greeting" and len(tokens) <= 5:
            return EngineResponse(
                "Hello! I'm Tiny AI. Ask me a simple question about AI, programming, or my knowledge base.",
                1.0,
                source="conversation",
            )
        if normalised in {"who are you", "what is your name", "tell me about yourself"}:
            return EngineResponse(
                "I'm a small, custom-built question-answering engine. I use a local knowledge base, "
                "custom natural-language processing, persistent memory, multi-turn conversation context, document understanding, "
                "a live date/time tool, a date calculator, a safe calculator, scientific math, an offline unit converter, text analysis, JSON tools, a custom evidence-reasoning engine, and optional keyless web search. "
                "I can remember details you explicitly save and connect follow-up questions to earlier messages in the same chat. "
                "Ask 'What is your knowledge?' to see the exact questions in my local knowledge base.",
                1.0,
                source="conversation",
            )
        if normalised in {"how are you", "how are you doing"}:
            return EngineResponse("I'm running and ready to answer a question!", 1.0, source="conversation")
        if analysis.intent == "gratitude" or tokens.intersection({"thanks", "thank"}):
            return EngineResponse("You're welcome!", 1.0, source="conversation")
        if analysis.intent == "farewell" or tokens.intersection({"bye", "goodbye"}):
            return EngineResponse("Goodbye!", 1.0, source="conversation")
        return None

    def answer(self, question: str, *, allow_calculator: bool = True) -> EngineResponse:
        if not isinstance(question, str):
            raise TypeError("question must be a string.")
        question = question.strip()
        if not question:
            return EngineResponse("Please type a question.", 0.0, source="validation", known=False)

        if self.nlp.analyze(question).intent == "capabilities":
            return self._knowledge_inventory_response()

        small_talk = self._handle_small_talk(question)
        if small_talk is not None:
            return small_talk

        # This assistant answers natural-language questions, not executable code.
        if re.search(r"__|\b(import|exec|eval)\b|[;{}]", question.lower()):
            return self._unknown_response()

        if allow_calculator:
            calculation = self._try_calculation(question)
            if calculation is not None:
                return EngineResponse(calculation, 1.0, source="calculator")

        # Resolve independent facets before selecting an answer. A compound
        # question is only marked fully known when each requested part is backed
        # by a sufficiently strong local match.
        facets = self.reasoner.decompose(question)
        if facets:
            facet_results: list[tuple[str, EngineResponse]] = []
            for facet in facets:
                result = self._retrieve(facet.query)
                if result is None:
                    return self._unknown_response()
                index, confidence = result
                if confidence < self.confidence_threshold:
                    return self._unknown_response(confidence)
                facet_results.append((facet.label, EngineResponse(
                    answer=self.entries[index].answer,
                    confidence=confidence,
                    matched_question=self.entries[index].question,
                    source="knowledge",
                    known=True,
                )))
            rendered: list[str] = []
            relationship = None
            if self.nlp.analyze(question).intent == "comparison" and len(facet_results) >= 2:
                relationship = self.reasoner.compare_relationship(
                    facet_results[0][0], facet_results[1][0], self.entries,
                )
            if relationship:
                rendered.append(f"**Key relationship**\n{relationship}")
            for label, result in facet_results:
                clean_label = label.title()
                clean_label = {"Ai": "AI", "Nlp": "NLP", "Ml": "ML", "Llm": "LLM"}.get(clean_label, clean_label)
                rendered.append(f"**{clean_label}**\n{result.answer}")
            confidence = min(item.confidence for _, item in facet_results)
            return EngineResponse(
                answer="\n\n".join(rendered),
                confidence=confidence,
                matched_question=None,
                source="reasoning",
                known=True,
            )

        ranked = self._retrieve_candidates(question, limit=8)
        if not ranked:
            return self._unknown_response()
        best = ranked[0]
        if best.score < self.confidence_threshold:
            return self._unknown_response(best.score)

        entry = self.entries[best.index]
        plan = self.nlp.plan(question, vocabulary=self._vocabulary)
        intent = str(plan.get("intent", "general_question"))
        relational_question = bool(re.search(
            r"\b(relationship|relate|related|connect|connection|affect|affects|impact|influence|help|helps|work together|combine|combined|interact)\b",
            question.lower(),
        ))
        if intent in {"summary", "list"}:
            strong_evidence = [
                item for item in ranked
                if item.score >= max(self.confidence_threshold, best.score * 0.62)
                and item.coverage >= max(0.18, best.coverage * 0.42)
            ]
            composed = self.reasoner.compose_evidence_answer(
                question, self.entries, strong_evidence or ranked[:1],
                style="bullets", limit=6 if intent == "summary" else 5,
            )
            if composed:
                return EngineResponse(
                    answer=composed,
                    confidence=best.score,
                    matched_question=entry.question,
                    source="reasoning",
                    known=True,
                )
        elif relational_question:
            # Only synthesize when there is independently strong supporting
            # evidence. The response identifies evidence; it does not invent a
            # causal link that is absent from the stored answers.
            strong = [item for item in ranked if item.score >= max(self.confidence_threshold, best.score * 0.70)]
            diverse = self.reasoner.diverse_evidence(question, strong, limit=3)
            if len(diverse) >= 2:
                composed = self.reasoner.compose_evidence_answer(
                    question, self.entries, diverse, style="bullets", limit=4,
                )
                if composed and len(composed.splitlines()) >= 2:
                    return EngineResponse(
                        answer="The local knowledge base contains these relevant pieces of evidence:\n" + composed,
                        confidence=best.score,
                        matched_question=entry.question,
                        source="reasoning",
                        known=True,
                    )

        return EngineResponse(
            answer=entry.answer,
            confidence=best.score,
            matched_question=entry.question,
            source="knowledge",
            known=True,
        )

    def _unknown_response(self, confidence: float = 0.0) -> EngineResponse:
        return EngineResponse(
            "Sorry, I don't know that. It's outside my knowledge range.",
            confidence=confidence,
            source="unknown",
            known=False,
        )

    def add_knowledge(
        self,
        question: str,
        answer: str,
        *,
        keywords: Iterable[str] = (),
        alternate_questions: Iterable[str] = (),
        save: bool = True,
    ) -> None:
        entry = KnowledgeEntry(
            question=question.strip(),
            answer=answer.strip(),
            keywords=tuple(str(item).strip() for item in keywords if str(item).strip()),
            alternate_questions=tuple(
                str(item).strip() for item in alternate_questions if str(item).strip()
            ),
        )
        if not entry.question or not entry.answer:
            raise ValueError("Both question and answer are required.")
        normalized_question = CustomNLP.normalize(entry.question)
        replaced = False
        updated_entries: list[KnowledgeEntry] = []
        for existing in self.entries:
            if CustomNLP.normalize(existing.question) == normalized_question:
                if not replaced:
                    updated_entries.append(entry)
                    replaced = True
            else:
                updated_entries.append(existing)
        if not replaced:
            updated_entries.append(entry)
        self.entries = updated_entries
        self._rebuild_index()
        if save:
            self.save_knowledge()

    def save_knowledge(self, path: str | Path | None = None) -> Path:
        destination = Path(path) if path else self.knowledge_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = [entry.to_dict() for entry in self.entries]
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return destination


def main() -> None:
    engine = TinyAIEngine()
    print("Tiny AI Engine — custom-built question answering")
    print("Ask a question, type 'teach: question => answer' to add knowledge, or type 'exit'.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTiny AI: Goodbye!")
            break
        if question.lower() in {"exit", "quit"}:
            print("Tiny AI: Goodbye!")
            break
        if question.lower() in {"help", "/help"}:
            print("Tiny AI: Ask a question, enter arithmetic, or teach me with:")
            print("         teach: What is a learning rate? => It controls the size of each training update.")
            continue
        if question.lower().startswith("teach:"):
            payload = question[6:].strip()
            if "=>" not in payload:
                print("Tiny AI: Use 'teach: question => answer'.")
                continue
            new_question, new_answer = payload.split("=>", 1)
            try:
                engine.add_knowledge(new_question, new_answer, save=True)
            except ValueError as exc:
                print(f"Tiny AI: {exc}")
            else:
                print("Tiny AI: Learned that answer. I'll use it in future questions.")
            continue
        response = engine.answer(question)
        print(f"Tiny AI: {response.answer}")
        if response.source == "knowledge":
            print(f"         (matched: {response.matched_question}; match score: {response.confidence:.2f})")


if __name__ == "__main__":
    main()
