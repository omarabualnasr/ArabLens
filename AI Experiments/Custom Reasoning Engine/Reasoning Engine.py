"""Experiment 022 — Custom evidence ranking and question reasoning.

This module is written from scratch for AI Experiment. It uses transparent
lexical statistics and the project's own NLP analysis; it does not use a
pretrained model, hosted model, external embeddings, or an AI API.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RankedEvidence:
    index: int
    score: float
    bm25_score: float
    coverage: float
    matched_terms: tuple[str, ...]


@dataclass(frozen=True)
class QuestionFacet:
    query: str
    label: str
    inherited_topic: bool = False


class CustomReasoningEngine:
    """A small, inspectable reasoning layer for retrieval and answer synthesis."""

    def __init__(self, nlp: object, *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and b between 0 and 1.")
        self.nlp = nlp
        self.k1 = float(k1)
        self.b = float(b)
        self._documents: list[list[str]] = []
        self._lengths: list[int] = []
        self._average_length = 0.0
        self._document_frequency: dict[str, int] = {}

    @staticmethod
    def _entry_text(entry: object) -> str:
        question = str(getattr(entry, "question", ""))
        alternatives = " ".join(str(item) for item in getattr(entry, "alternate_questions", ()))
        keywords = " ".join(str(item) for item in getattr(entry, "keywords", ()))
        answer = str(getattr(entry, "answer", ""))
        return " ".join((question, alternatives, keywords, answer))

    def fit(self, entries: Iterable[object]) -> None:
        """Build the BM25 statistics from locally stored knowledge entries."""
        self._documents = []
        self._document_frequency = {}
        for entry in entries:
            analysis = self.nlp.analyze(self._entry_text(entry))
            terms = list(analysis.expanded_keywords)
            self._documents.append(terms)
            for term in set(terms):
                self._document_frequency[term] = self._document_frequency.get(term, 0) + 1
        self._lengths = [len(document) for document in self._documents]
        self._average_length = sum(self._lengths) / max(1, len(self._lengths))

    def bm25_scores(self, query: str) -> list[float]:
        """Return a BM25 relevance score for every indexed knowledge entry."""
        if not self._documents:
            return []
        query_terms = tuple(dict.fromkeys(self.nlp.analyze(query).expanded_keywords))
        if not query_terms:
            return [0.0] * len(self._documents)
        count = len(self._documents)
        scores: list[float] = []
        for document, length in zip(self._documents, self._lengths):
            frequencies: dict[str, int] = {}
            for term in document:
                frequencies[term] = frequencies.get(term, 0) + 1
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                df = self._document_frequency.get(term, 0)
                # Robertson/Sparck Jones IDF with a positive floor for tiny corpora.
                idf = max(0.0, math.log(1.0 + (count - df + 0.5) / (df + 0.5)))
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / max(self._average_length, 1.0)
                )
                score += idf * (frequency * (self.k1 + 1.0)) / max(denominator, 1e-12)
            scores.append(score)
        return scores

    @staticmethod
    def normalise_scores(scores: Sequence[float]) -> list[float]:
        if not scores:
            return []
        maximum = max(scores)
        if maximum <= 1e-12:
            return [0.0 for _ in scores]
        return [min(1.0, max(0.0, score / maximum)) for score in scores]

    def rank(self, query: str, entries: Sequence[object], base_scores: Sequence[float]) -> list[RankedEvidence]:
        """Blend a caller's NLP/TF-IDF score with independent BM25 evidence."""
        if len(entries) != len(base_scores):
            raise ValueError("Entries and base scores must have the same length.")
        bm25 = self.normalise_scores(self.bm25_scores(query))
        query_terms = set(self.nlp.analyze(query).expanded_keywords)
        ranked: list[RankedEvidence] = []
        for index, entry in enumerate(entries):
            answer_terms = set(self.nlp.analyze(self._entry_text(entry)).expanded_keywords)
            matched = tuple(sorted(query_terms & answer_terms))
            coverage = len(matched) / max(1, len(query_terms))
            lexical = min(1.0, max(0.0, float(base_scores[index])))
            # The original query intent score remains the strongest component;
            # BM25 improves term-frequency and corpus-wide discrimination.
            final_score = 0.84 * lexical + 0.12 * bm25[index] + 0.04 * coverage
            ranked.append(RankedEvidence(
                index=index,
                score=min(1.0, final_score),
                bm25_score=bm25[index],
                coverage=coverage,
                matched_terms=matched,
            ))
        ranked.sort(key=lambda item: (-item.score, -item.coverage, item.index))
        return ranked

    def rank_texts(self, query: str, texts: Sequence[str]) -> list[float]:
        """BM25-normalise an arbitrary set of passages (for document retrieval)."""
        class _TextEntry:
            def __init__(self, value: str) -> None:
                self.question = ""
                self.alternate_questions: tuple[str, ...] = ()
                self.keywords: tuple[str, ...] = ()
                self.answer = value
        self.fit([_TextEntry(text) for text in texts])
        return self.normalise_scores(self.bm25_scores(query))

    def decompose(self, question: str) -> list[QuestionFacet]:
        """Split genuinely compound questions and resolve simple carried subjects."""
        normalized = self.nlp.normalize(question).strip()
        normalized = re.sub(r"\s+", " ", normalized)
        if not normalized:
            return []

        # Comparison questions need two independently retrievable subject facets.
        comparison = re.search(
            r"(?:what is the )?difference between\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)[?.!]*$|"
            r"^compare\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)[?.!]*$",
            normalized,
            flags=re.IGNORECASE,
        )
        if comparison:
            groups = comparison.groups()
            left_raw, right_raw = (groups[0], groups[1]) if groups[0] is not None else (groups[2], groups[3])
            left, right = self._clean_subject(left_raw), self._clean_subject(right_raw)
            if left and right and left != right:
                return [
                    QuestionFacet(f"What is {left}?", left),
                    QuestionFacet(f"What is {right}?", right),
                ]

        # Split only at coordinated question frames, never at every occurrence
        # of "and" (which would damage ordinary noun phrases).
        pieces = re.split(
            r"\s+(?:and|also|;|\?)\s*(?=(?:what|who|where|when|why|how|which|explain|describe|define|compare)\b)",
            normalized,
            flags=re.IGNORECASE,
        )
        if len(pieces) <= 1:
            # Independent sentence questions are also valid facets.
            pieces = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
        pieces = [part.strip(" ?.!;,:") for part in pieces if part.strip(" ?.!;,")]
        if len(pieces) <= 1:
            return []

        facets: list[QuestionFacet] = []
        inherited_topic = ""
        topic_aliases = {
            "ai": "artificial intelligence",
            "ml": "machine learning",
            "nlp": "natural language processing",
            "nn": "neural network",
            "llm": "large language model",
        }
        for piece in pieces[:4]:
            analysis = self.nlp.analyze(piece)
            focus = str(analysis.focus).strip()
            if not inherited_topic and focus:
                inherited_topic = topic_aliases.get(focus.lower(), focus)
            query = piece
            inherited = False
            if inherited_topic and re.search(r"\b(it|they|them|this|that|these|those)\b", piece):
                query = re.sub(r"\b(it|they|them|this|that|these|those)\b", inherited_topic, piece, count=2)
                inherited = True
            if len(self.nlp.analyze(query).keywords) < 1:
                continue
            label = str(self.nlp.analyze(query).focus or query).strip()
            facets.append(QuestionFacet(query=query, label=label, inherited_topic=inherited))
        return facets if len(facets) > 1 else []

    def compare_relationship(self, left_topic: str, right_topic: str, entries: Sequence[object]) -> str | None:
        """Find an explicit taxonomy fact connecting two compared concepts."""
        left_terms = set(self.nlp.analyze(left_topic).expanded_keywords)
        right_terms = set(self.nlp.analyze(right_topic).expanded_keywords)
        relationship_pattern = re.compile(
            r"^\s*(?P<subject>[A-Za-z][A-Za-z0-9 -]{1,80}?)\s+is\s+(?:a|an|the)\s+"
            r"(?P<kind>subfield|field|branch|part|type|form|subset)\s+of\s+"
            r"(?P<object>[A-Za-z][A-Za-z0-9 -]{1,80}?)(?=\s+(?:in which|where|that|which|who)\b|[,.;]|$)",
            re.IGNORECASE,
        )
        for entry in entries:
            answer = str(getattr(entry, "answer", ""))
            for sentence in re.split(r"(?<=[.!?])\s+", answer):
                match = relationship_pattern.search(sentence)
                if not match:
                    continue
                subject = match.group("subject").strip()
                object_name = re.split(r"\b(?:where|that|which|who)\b", match.group("object"), maxsplit=1, flags=re.IGNORECASE)[0].strip()
                subject_terms = set(self.nlp.analyze(subject).expanded_keywords)
                object_terms = set(self.nlp.analyze(object_name).expanded_keywords)
                left_is_subject = bool(left_terms & subject_terms) and bool(right_terms & object_terms)
                right_is_subject = bool(right_terms & subject_terms) and bool(left_terms & object_terms)
                if not (left_is_subject or right_is_subject):
                    continue
                specific = subject if left_is_subject else subject
                broad = object_name
                return (
                    f"The stored knowledge states that {specific} is a {match.group('kind')} of {broad}. "
                    f"So {broad} is the broader concept, while {specific} is a more specific area within it."
                )
        return None

    @staticmethod
    def _clean_subject(value: str) -> str:
        value = re.sub(r"^(?:the|a|an)\s+", "", value.strip())
        value = re.sub(r"\s+(?:is|are|was|were)$", "", value)
        return value.strip(" ?.!;,:")

    def diverse_evidence(
        self,
        question: str,
        ranked: Sequence[RankedEvidence],
        *,
        limit: int = 4,
    ) -> list[RankedEvidence]:
        """Choose evidence that covers different parts of a multi-concept query."""
        target = set(self.nlp.analyze(question).expanded_keywords)
        if not target:
            return list(ranked[:max(1, limit)])
        remaining = list(ranked)
        selected: list[RankedEvidence] = []
        covered: set[str] = set()
        while remaining and len(selected) < max(1, limit):
            choices: list[tuple[float, int, RankedEvidence, set[str]]] = []
            for item in remaining:
                new_terms = set(item.matched_terms) & target - covered
                if not new_terms:
                    continue
                gain = len(new_terms) / max(1, len(target))
                utility = 0.70 * item.score + 0.30 * gain
                choices.append((utility, len(new_terms), item, new_terms))
            if not choices:
                break
            choices.sort(key=lambda value: (-value[0], -value[1], value[2].index))
            _, _, chosen, new_terms = choices[0]
            selected.append(chosen)
            covered.update(new_terms)
            remaining = [item for item in remaining if item.index != chosen.index]
        return selected

    def answer_sentences(
        self,
        question: str,
        entries: Sequence[object],
        ranked: Sequence[RankedEvidence],
        *,
        limit: int = 5,
        score_floor: float = 0.12,
    ) -> list[tuple[float, int, str]]:
        """Extract non-duplicated answer sentences with direct lexical support."""
        query = self.nlp.analyze(question)
        query_terms = set(query.focus_tokens) or set(query.expanded_keywords)
        if not query_terms:
            return []
        evidence: list[tuple[float, int, str]] = []
        for rank_index, item in enumerate(ranked):
            entry = entries[item.index]
            answer = str(getattr(entry, "answer", ""))
            sentences = re.split(r"(?<=[.!?])\s+|\n+", answer)
            for sentence_index, sentence in enumerate(sentences):
                sentence = sentence.strip(" •\t-")
                if len(sentence) < 18:
                    continue
                sentence_analysis = self.nlp.analyze(sentence)
                terms = set(sentence_analysis.expanded_keywords)
                shared = query_terms & terms
                coverage = len(shared) / max(1, len(query_terms))
                precision = len(shared) / max(1, len(terms))
                phrase = 0.12 if query.focus and query.focus.lower() in sentence.lower() else 0.0
                score = 0.68 * coverage + 0.20 * precision + phrase + 0.02 / (1 + rank_index)
                if score >= score_floor:
                    evidence.append((score, item.index * 1000 + sentence_index, sentence))
        evidence.sort(key=lambda item: (-item[0], item[1]))
        selected: list[tuple[float, int, str]] = []
        seen: set[str] = set()
        for item in evidence:
            key = self.nlp.normalize(item[2])
            if key in seen:
                continue
            selected.append(item)
            seen.add(key)
            if len(selected) >= max(1, limit):
                break
        return selected

    def compose_evidence_answer(
        self,
        question: str,
        entries: Sequence[object],
        ranked: Sequence[RankedEvidence],
        *,
        style: str = "paragraph",
        limit: int = 5,
    ) -> str:
        """Build a concise answer solely from retrieved local evidence."""
        sentences = self.answer_sentences(question, entries, ranked, limit=limit)
        if not sentences:
            return ""
        if style == "bullets":
            return "\n".join(f"• {sentence}" for _, _, sentence in sentences)
        return " ".join(sentence for _, _, sentence in sentences)
