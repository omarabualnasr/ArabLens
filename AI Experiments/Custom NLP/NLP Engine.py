"""Custom, dependency-free natural-language processing for Tiny AI.

This is an educational lexical NLP pipeline written for this project. It does
not use spaCy, NLTK, transformers, hosted services, or pretrained embeddings.
It performs normalization, tokenization, light stemming, typo correction,
keyword/phrase extraction, simple intent detection, and lexical matching.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


_TOKEN_RE = re.compile(r"[\w]+(?:['’][\w]+)?", re.UNICODE)
_WORD_RE = re.compile(r"^[\w]+(?:['’][\w]+)?$", re.UNICODE)

_CONTRACTIONS = {
    "what's": "what is", "whats": "what is", "who's": "who is",
    "whos": "who is", "where's": "where is", "wheres": "where is",
    "when's": "when is", "whens": "when is", "it's": "it is",
    "that's": "that is", "thats": "that is", "there's": "there is",
    "theres": "there is", "i'm": "i am", "im": "i am",
    "you're": "you are", "youre": "you are", "we're": "we are",
    "they're": "they are", "isn't": "is not", "isnt": "is not",
    "aren't": "are not", "arent": "are not", "wasn't": "was not",
    "weren't": "were not", "don't": "do not", "dont": "do not",
    "doesn't": "does not", "doesnt": "does not", "didn't": "did not",
    "didnt": "did not", "can't": "cannot", "cant": "cannot",
    "couldn't": "could not", "couldnt": "could not", "wouldn't": "would not",
    "wouldnt": "would not", "shouldn't": "should not", "shouldnt": "should not",
    "won't": "will not", "wont": "will not", "we've": "we have",
    "they've": "they have", "i've": "i have", "you've": "you have",
    "i'll": "i will", "you'll": "you will", "we'll": "we will",
}

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "can", "could", "did", "do", "does", "doing", "for", "from",
    "had", "has", "have", "having", "he", "her", "hers", "him", "his",
    "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on",
    "or", "our", "ours", "please", "she", "that", "the", "their", "them",
    "there", "these", "they", "this", "those", "to", "up", "us", "was",
    "we", "were", "what", "when", "where", "which", "who", "whom", "whose",
    "why", "how", "will", "with", "would", "you", "your", "yours", "tell", "explain",
    "describe", "define", "mean", "meaning", "about", "know", "like", "please",
}

_COMMON_LEXICON = {
    "latest", "news", "today", "yesterday", "tomorrow", "world", "technology",
    "artificial", "intelligence", "machine", "learning", "neural", "network",
    "neuron", "model", "training", "weight", "bias", "loss", "gradient",
    "descent", "backpropagation", "backprop", "token", "tokenizer", "attention",
    "transformer", "language", "processing", "natural", "python", "numpy",
    "programming", "algorithm", "computer", "science", "data", "knowledge",
    "question", "answer", "created", "creator", "calculate", "calculation",
    "search", "engine", "assistant", "chatbot", "gpt", "large", "text",
    "embedding", "optimization", "optimizer", "derivative", "function", "matrix",
    "document", "documents", "uploaded", "upload", "extract", "extraction", "summary",
    "summarize", "overview", "paragraph", "chapter", "section", "page", "report",
    "contract", "invoice", "resume", "curriculum", "policy", "research", "table",
    "tensor", "autograd", "convolution", "classification", "probability", "simple",
    "example", "difference", "compare", "explain", "definition", "meaning",
}

_KNOWN_CORRECTIONS = {
    "latets": "latest", "lates": "latest", "teh": "the", "wierd": "weird",
    "recieve": "receive", "recieved": "received", "thier": "their",
    "neuralnetowrk": "neuralnetwork", "backpropogation": "backpropagation",
    "summery": "summary", "documet": "document", "docuemnt": "document",
    "reciept": "receipt", "reciepts": "receipts", "adress": "address",
}

_SYNONYM_EXPANSIONS = {
    "ai": ("artificial", "intelligence"),
    "ml": ("machine", "learning"),
    "nn": ("neural", "network"),
    "llm": ("large", "language", "model"),
    "nlp": ("natural", "language", "processing"),
    "backprop": ("backpropagation",),
    "chatbot": ("chat", "assistant"),
    "space": ("outer space", "cosmic space"),
    "cosmos": ("universe",),
    "universe": ("cosmos",),
    "web": ("internet", "online"),
    "internet": ("web", "online"),
    "meaning": ("definition",),
    "define": ("definition",),
    "doc": ("document", "file"),
    "docs": ("document", "files"),
    "pdf": ("document", "file"),
    "docx": ("document", "file"),
    "uploaded": ("upload", "document"),
    "summarize": ("summary", "overview", "main points"),
    "overview": ("summary", "main points"),
    "receipt": ("invoice", "payment"),
    "resume": ("cv", "curriculum vitae"),
}

_INTENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("summary", re.compile(r"\b(summarize|summary|overview|key takeaways|key points|main points)\b", re.I)),
    ("document_question", re.compile(r"\b(according to|in the (?:uploaded )?(?:document|file|pdf|report)|from the (?:document|file|pdf)|what does (?:this|the) (?:document|file|pdf) say)\b", re.I)),
    ("greeting", re.compile(r"^(hi|hello|hey|good morning|good afternoon|good evening|greetings)(\b|$)")),
    ("farewell", re.compile(r"^(bye|goodbye|see you|see ya)(\b|$)")),
    ("gratitude", re.compile(r"^(thanks|thank you|thx)(\b|$)")),
    ("capabilities", re.compile(
        r"^(?:what (?:is|are) your (?:knowledge(?: base| range)?|capabilities)|"
        r"what do you know(?: about)?(?: exactly)?|what can you (?:answer|do)|what can i ask you(?: about)?|"
        r"what (?:topics|subjects|areas) (?:do you know|can you cover|can you answer)|"
        r"what are your (?:topics|subjects|areas) of knowledge|what knowledge do you have|"
        r"what information do you have(?: in your knowledge base)?|how much do you know|"
        r"list (?:your )?(?:knowledge|topics|subjects)|show (?:me )?(?:your )?(?:knowledge|topics|subjects)|"
        r"tell me what you know|tell me about your knowledge|"
        r"what kind of (?:questions|topics) can you answer|"
        r"which (?:topics|subjects|questions) can i ask|"
        r"what tools (?:do you have|can you use|are available)|"
        r"which tools (?:do you have|can you use|are available)|"
        r"list (?:your )?tools)(?:[?!., ]*)$", re.I
    )),
    ("identity", re.compile(r"\b(who are you|what can you do|what is your name)\b")),
    ("search", re.compile(r"^(?:/search\s+|search\s*:|search\s+(?:the\s+)?web\s+for\s+|look\s+up\s+|find\s+online\s+)", re.I)),
    ("current_information", re.compile(r"\b(latest|current|currently|today|yesterday|right now|this week|recent|newest|breaking|news|price|weather|exchange rate)\b", re.I)),
    ("comparison", re.compile(r"\b(difference between|compare|versus|vs\.?|better than|区别)\b", re.I)),
    ("procedure", re.compile(r"\b(how (?:do i|can i|could i|should i|to)|steps to|guide to|way to|how can i)\b", re.I)),
    ("list", re.compile(r"\b(list|name|give me|what are the types|examples of|some examples)\b", re.I)),
    ("definition", re.compile(r"^(?:(?:please )?(?:what is|what are|define|meaning of)|what does .+ mean|can you explain|could you explain|tell me about|i want to know)\b", re.I)),
    ("explanation", re.compile(r"\b(explain|describe|how does|how do|why does|why do|tell me how|what does .+ do|what do .+ do)\b", re.I)),
    ("who", re.compile(r"^\s*who\b", re.I)),
    ("where", re.compile(r"^\s*where\b", re.I)),
    ("when", re.compile(r"^\s*when\b", re.I)),
    ("why", re.compile(r"^\s*why\b", re.I)),
)


@dataclass(frozen=True)
class NLPAnalysis:
    original: str
    normalized: str
    tokens: tuple[str, ...]
    lemmas: tuple[str, ...]
    keywords: tuple[str, ...]
    expanded_keywords: tuple[str, ...]
    bigrams: tuple[str, ...]
    intent: str
    question_word: str | None
    corrected: tuple[tuple[str, str], ...]
    entities: tuple[str, ...]
    focus: str
    focus_tokens: tuple[str, ...]


class CustomNLP:
    """Rule-based lexical NLP system, designed to be transparent and extensible."""

    def __init__(self, vocabulary: Iterable[str] = ()) -> None:
        self.vocabulary = set(_COMMON_LEXICON)
        self.vocabulary.update(self._clean_vocab_word(item) for item in vocabulary if item)

    @staticmethod
    def _clean_vocab_word(word: str) -> str:
        return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKC", str(word)).lower())

    @staticmethod
    def normalize(text: str) -> str:
        value = unicodedata.normalize("NFKC", str(text or ""))
        value = value.replace("’", "'").replace("‘", "'").replace("–", "-").replace("—", "-")
        value = value.lower().strip()
        # Expand contractions before tokenization so question frames are clearer.
        for contraction in sorted(_CONTRACTIONS, key=len, reverse=True):
            value = re.sub(rf"(?<!\w){re.escape(contraction)}(?!\w)", _CONTRACTIONS[contraction], value)
        value = re.sub(r"\s+", " ", value)
        return value

    @staticmethod
    def tokenize(text: str) -> tuple[str, ...]:
        return tuple(match.group(0).replace("’", "'").lower() for match in _TOKEN_RE.finditer(text))

    @staticmethod
    def lemmatize(token: str) -> str:
        word = token.lower()
        if len(word) <= 3 or not word.isalpha():
            return word
        irregular = {"children": "child", "people": "person", "men": "man", "women": "woman", "better": "good", "best": "good", "worse": "bad", "went": "go", "was": "be", "were": "be", "has": "have", "had": "have"}
        if word in irregular:
            return irregular[word]
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith("ing") and len(word) > 5:
            root = word[:-3]
            if len(root) > 2 and root[-1] == root[-2]:
                root = root[:-1]
            if root.endswith(("mak", "creat", "writ", "updat", "generat")):
                root += "e"
            return root
        if word.endswith("ed") and len(word) > 4:
            root = word[:-2]
            if len(root) > 2 and root[-1] == root[-2]:
                root = root[:-1]
            if root.endswith(("creat", "updat", "generat", "calculat", "us")):
                root += "e"
            return root
        if word.endswith("es") and len(word) > 4 and word[-3] in "sxz":
            return word[:-2]
        if word.endswith("s") and len(word) > 4 and not word.endswith(("ss", "us", "is", "news", "series", "analysis", "status", "business")):
            return word[:-1]
        if word.endswith("ly") and len(word) > 5:
            return word[:-2]
        return word

    @staticmethod
    def _damerau_distance_one(left: str, right: str) -> bool:
        """Return whether strings are at Damerau-Levenshtein distance <= 1."""
        if left == right:
            return True
        if abs(len(left) - len(right)) > 1:
            return False
        if len(left) == len(right):
            mismatches = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
            if len(mismatches) == 1:
                return True
            if len(mismatches) == 2:
                first, second = mismatches
                return second == first + 1 and left[first] == right[second] and left[second] == right[first]
            return False
        shorter, longer = (left, right) if len(left) < len(right) else (right, left)
        i = j = edits = 0
        while i < len(shorter) and j < len(longer):
            if shorter[i] == longer[j]:
                i += 1
                j += 1
            else:
                edits += 1
                j += 1
                if edits > 1:
                    return False
        return True

    def _correct(self, token: str, vocabulary: set[str]) -> str:
        direct = _KNOWN_CORRECTIONS.get(token)
        if direct:
            return direct
        if len(token) < 5 or not token.isalpha() or token in vocabulary:
            return token
        candidates = [word for word in vocabulary if abs(len(word) - len(token)) <= 1 and self._damerau_distance_one(token, word)]
        # Only correct when the candidate is unambiguous. Guessing is worse than
        # preserving an unfamiliar word and letting retrieval reject it safely.
        return candidates[0] if len(candidates) == 1 else token

    @staticmethod
    def _intent(normalized: str) -> tuple[str, str | None]:
        for intent, pattern in _INTENT_PATTERNS:
            if pattern.search(normalized):
                first = re.match(r"([a-z]+)", normalized)
                return intent, first.group(1) if first and first.group(1) in {"what", "who", "where", "when", "why", "how"} else None
        first = re.match(r"([a-z]+)", normalized)
        question_word = first.group(1) if first and first.group(1) in {"what", "who", "where", "when", "why", "how"} else None
        return ("general_question" if "?" in normalized or question_word else "statement"), question_word

    @staticmethod
    def _extract_entities(original: str, tokens: tuple[str, ...]) -> tuple[str, ...]:
        # Lightweight proper-name extraction: retain capitalized words/acronyms
        # before normalization. It is intentionally conservative.
        found = re.findall(r"(?<!\w)[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*", original)
        unique: list[str] = []
        for item in found:
            cleaned = item.strip()
            if cleaned.lower() in {"what", "who", "where", "when", "why", "how", "explain", "tell", "the", "a", "an"}:
                continue
            if cleaned and cleaned.lower() not in {x.lower() for x in unique}:
                unique.append(cleaned)
        return tuple(unique[:8])

    @staticmethod
    def extract_focus(text: str) -> str:
        """Remove common conversational question frames to expose the topic.

        Examples: ``Could you explain what a tensor is?`` -> ``tensor``;
        ``What does NLP do?`` -> ``nlp``. This is deliberately rule-based.
        """
        value = CustomNLP.normalize(text)
        value = re.sub(r"[?!.]+$", "", value).strip()
        prefixes = (
            r"^(?:please\s+)?(?:can|could|would)\s+you\s+(?:please\s+)?(?:explain|describe|define|tell\s+me|summarize|summary)\s+(?:to\s+me\s+)?",
            r"^(?:according to|based on|from)\s+(?:the\s+)?(?:uploaded\s+)?(?:document|file|pdf|report)\s*,?\s*",
            r"^(?:please\s+)?(?:summarize|summary)\s+",
            r"^(?:please\s+)?tell\s+me\s+about\s+",
            r"^(?:can|could|may)\s+i\s+ask\s+(?:you\s+)?",
            r"^(?:please\s+)?(?:what can you tell me about|give me (?:some )?(?:details|information) about|i need information about|i would like to know more about)\s+",
            r"^(?:i\s+am|i'm)\s+(?:wondering|curious)\s+(?:about\s+)?",
            r"^(?:i\s+)?(?:want|need|would\s+like)\s+to\s+know\s+(?:about\s+)?",
            r"^(?:i\s+have)\s+a\s+question\s+about\s+",
            r"^(?:please\s+)?(?:explain|describe|define)\s+",
            r"^(?:how\s+(?:many|much|long|often|far|early|frequently)(?:\s+should|\s+can|\s+do|\s+does)?)\s+",
            r"^(?:what|who|where|when|why|how)\s+(?:is|are|was|were|do|does|did|can|could|would|should)\s+",
            r"^what\s+",
            r"^(?:who)\s+(?:created|invented|made|built)\s+",
            r"^(?:where)\s+(?:can\s+i\s+)?(?:find|get)\s+",
        )
        for pattern in prefixes:
            value = re.sub(pattern, "", value, count=1, flags=re.IGNORECASE)
        # Remove question-form endings only when they are acting as grammar,
        # not when they are part of a named concept.
        value = re.sub(r"^(?:the\s+)?(?:purpose|role|use|function|meaning)\s+of\s+", "", value)
        value = re.sub(r"^(?:the\s+)?(?:uploaded\s+)?(?:document|file|pdf|report)\s+(?:say|says|contain|contains|include|includes)(?:\s+about)?\s+", "", value)
        value = re.sub(r"^(?:the\s+)?(?:uploaded\s+)?(?:document|file|pdf|report)\s+about\s+", "", value)
        value = re.sub(r"\s+(?:is|are|was|were|do|does|did|work|works|mean|means|called|used|used\s+for)\s*$", "", value)
        value = re.sub(r"\s+(?:(?:do|does|did|should|can|could|will|would)\s+)?(?:i|you|we|they|employees?|people|users?|customers?)\s+(?:receive|take|submit|send|use|need|make|do|have|work|mean|contain|include|store|support|provide)(?:\s+\w+){0,2}$", "", value)
        value = re.sub(r"\s+(?:be\s+)?(?:submitted|approved|sent|received|calculated|stored|created|generated)$", "", value)
        value = re.sub(r"^(?:a|an|the)\s+", "", value)
        value = re.sub(r"\s+in\s+simple\s+terms$", "", value)
        value = re.sub(r"\s+(?:in|from|inside|within|according to)\s+(?:the\s+)?(?:uploaded\s+)?(?:document|file|pdf|report|attachment)$", "", value)
        return re.sub(r"\s+", " ", value).strip(" ,:;-")

    def analyze(self, text: str, *, vocabulary: Iterable[str] = ()) -> NLPAnalysis:
        normalized = self.normalize(text)
        raw_tokens = self.tokenize(normalized)
        lexicon = self.vocabulary | {self._clean_vocab_word(item) for item in vocabulary if item}
        corrected_tokens: list[str] = []
        corrected_pairs: list[tuple[str, str]] = []
        for token in raw_tokens:
            corrected = self._correct(token, lexicon) if _WORD_RE.match(token) else token
            corrected_tokens.append(corrected)
            if corrected != token:
                corrected_pairs.append((token, corrected))

        lemmas = tuple(self.lemmatize(token) for token in corrected_tokens if _WORD_RE.match(token))
        keywords = tuple(token for token in lemmas if token not in _STOP_WORDS and not token.isdigit())
        expanded = list(keywords)
        for token in keywords:
            expanded.extend(_SYNONYM_EXPANSIONS.get(token, ()))
        # Expand phrase aliases only when the exact phrase appears.
        joined = " ".join(lemmas)
        if "back propagation" in joined:
            expanded.append("backpropagation")
        if "artificial intelligence" in joined:
            expanded.append("ai")
        expanded_keywords = tuple(dict.fromkeys(expanded))
        bigrams = tuple(f"{left} {right}" for left, right in zip(keywords, keywords[1:]))
        intent, question_word = self._intent(normalized)
        focus = self.extract_focus(normalized)
        focus_tokens_list: list[str] = []
        for token in self.tokenize(focus):
            corrected_focus = self._correct(token, lexicon)
            lemma = self.lemmatize(corrected_focus)
            if lemma not in _STOP_WORDS and not lemma.isdigit():
                focus_tokens_list.append(lemma)
        focus_tokens = tuple(focus_tokens_list)
        return NLPAnalysis(
            original=str(text), normalized=normalized,
            tokens=tuple(corrected_tokens), lemmas=lemmas,
            keywords=keywords, expanded_keywords=expanded_keywords,
            bigrams=bigrams, intent=intent, question_word=question_word,
            corrected=tuple(corrected_pairs),
            entities=self._extract_entities(str(text), tuple(corrected_tokens)),
            focus=focus, focus_tokens=focus_tokens,
        )

    @staticmethod
    def similarity(left: NLPAnalysis, right: NLPAnalysis) -> float:
        """Score lexical, phrase and intent overlap with transparent weights."""
        a = set(left.expanded_keywords)
        b = set(right.expanded_keywords)
        if not a or not b:
            return 0.0
        shared = a & b
        coverage = len(shared) / len(a)
        jaccard = len(shared) / len(a | b)
        left_phrases = set(left.bigrams)
        right_phrases = set(right.bigrams)
        phrase_score = len(left_phrases & right_phrases) / max(1, min(len(left_phrases), len(right_phrases)))
        left_focus = set(left.focus_tokens)
        right_focus = set(right.focus_tokens)
        focus_score = len(left_focus & right_focus) / max(1, min(len(left_focus), len(right_focus)))
        intent_score = 0.0
        if left.intent == right.intent and left.intent not in {"statement", "general_question"}:
            intent_score = 1.0
        elif {left.intent, right.intent} <= {"definition", "explanation", "general_question", "who", "where", "when", "why"}:
            intent_score = 0.5
        # Character trigrams add a small spelling/word-shape signal without
        # replacing meaning-bearing word and phrase overlap.
        def trigrams(value: str) -> set[str]:
            compact = re.sub(r"\s+", " ", value.lower()).strip()
            if len(compact) < 3:
                return {compact} if compact else set()
            return {compact[index:index + 3] for index in range(len(compact) - 2)}
        left_text = left.focus or " ".join(left.lemmas)
        right_text = right.focus or " ".join(right.lemmas)
        left_grams = trigrams(left_text)
        right_grams = trigrams(right_text)
        char_score = len(left_grams & right_grams) / max(1, len(left_grams | right_grams))
        # Query coverage remains the strongest signal; character overlap only
        # helps recover close wording and minor misspellings.
        return min(1.0, 0.34 * coverage + 0.16 * jaccard + 0.15 * phrase_score + 0.14 * intent_score + 0.13 * focus_score + 0.08 * char_score)

    def plan(self, text: str, *, vocabulary: Iterable[str] = ()) -> dict[str, object]:
        """Build a compact intent/topic plan for downstream custom tools.

        This is deterministic NLP, not a pretrained classifier. It exposes the
        interpretation so retrieval, document QA, and tools can share one plan.
        """
        analysis = self.analyze(text, vocabulary=vocabulary)
        normalized = analysis.normalized
        temporal = tuple(term for term in ("today", "yesterday", "tomorrow", "this week", "latest", "current", "recent") if term in normalized)
        if analysis.intent == "summary":
            action = "summarize"
        elif analysis.intent == "document_question":
            action = "retrieve_document_answer"
        elif analysis.intent == "procedure":
            action = "explain_steps"
        elif analysis.intent == "comparison":
            action = "compare"
        elif analysis.intent in {"definition", "explanation"}:
            action = "explain"
        elif analysis.intent == "list":
            action = "list"
        else:
            action = "answer"
        output_format = "bullets" if analysis.intent in {"summary", "list"} else "paragraph"
        if analysis.intent in {"who"}:
            answer_type = "person"
        elif analysis.intent in {"where"}:
            answer_type = "location"
        elif analysis.intent in {"when"} or temporal:
            answer_type = "date_or_time"
        elif re.search(r"\b(how many|how much|how long|what percentage|how far)\b", normalized):
            answer_type = "quantity"
        elif analysis.intent == "comparison":
            answer_type = "comparison"
        elif analysis.intent == "summary":
            answer_type = "summary"
        elif analysis.intent == "procedure":
            answer_type = "steps"
        elif analysis.intent == "definition":
            answer_type = "definition"
        else:
            answer_type = "explanation"
        relational = bool(re.search(
            r"\b(relationship|relate|related|connect|connection|affect|affects|impact|influence|help|helps|work together|combine|interact)\b",
            normalized,
        ))
        reasoning_style = (
            "compare" if analysis.intent == "comparison" else
            "summarize" if analysis.intent == "summary" else
            "stepwise" if analysis.intent == "procedure" else
            "multi_evidence" if relational else
            "direct"
        )
        return {
            "action": action,
            "intent": analysis.intent,
            "focus": analysis.focus,
            "focus_tokens": list(analysis.focus_tokens),
            "keywords": list(analysis.expanded_keywords),
            "entities": list(analysis.entities),
            "temporal_constraints": list(temporal),
            "negated": bool(re.search(r"\b(not|never|without|except)\b", normalized)),
            "answer_type": answer_type,
            "reasoning_style": reasoning_style,
            "requires_multiple_evidence": reasoning_style in {"compare", "summarize", "multi_evidence"},
            "output_format": output_format,
        }

    def inspect(self, text: str, *, vocabulary: Iterable[str] = ()) -> dict[str, object]:
        """Return a JSON-friendly diagnostic view useful for experiments/tests."""
        result = self.analyze(text, vocabulary=vocabulary)
        return {
            "normalized": result.normalized,
            "tokens": list(result.tokens),
            "lemmas": list(result.lemmas),
            "keywords": list(result.keywords),
            "expanded_keywords": list(result.expanded_keywords),
            "bigrams": list(result.bigrams),
            "intent": result.intent,
            "question_word": result.question_word,
            "corrections": [list(pair) for pair in result.corrected],
            "entities": list(result.entities),
            "focus": result.focus,
            "focus_tokens": list(result.focus_tokens),
            "plan": self.plan(text, vocabulary=vocabulary),
        }
