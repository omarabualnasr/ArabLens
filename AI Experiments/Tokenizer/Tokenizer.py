"""Character-level and Byte Pair Encoding tokenizers built from scratch.

This educational tokenizer starts from Unicode characters and learns
character-based BPE merges. It uses no third-party tokenizer package.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from numbers import Integral
from typing import Iterable, Sequence

DEFAULT_SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<BOS>", "<EOS>")


def _validate_special_tokens(special_tokens: Sequence[str]) -> tuple[str, ...]:
    tokens = tuple(special_tokens)
    if not tokens or any(not isinstance(t, str) or not t for t in tokens):
        raise ValueError("Special tokens must be non-empty strings.")
    if len(set(tokens)) != len(tokens):
        raise ValueError("Special tokens must be unique.")
    return tokens


def _split_special_tokens(text: str, special_tokens: Sequence[str]) -> list[tuple[bool, str]]:
    if not text:
        return []
    alternatives = sorted(special_tokens, key=len, reverse=True)
    pattern = "(" + "|".join(re.escape(token) for token in alternatives) + ")"
    known = set(special_tokens)
    return [(piece in known, piece) for piece in re.split(pattern, text) if piece]


class CharacterTokenizer:
    """Character-level tokenizer used as a baseline for BPE."""

    def __init__(self, special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS) -> None:
        self.special_tokens = _validate_special_tokens(special_tokens)
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: list[str] = []
        self.trained = False

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def train(self, texts: Iterable[str]) -> "CharacterTokenizer":
        corpus = list(texts)
        if not corpus or any(not isinstance(text, str) for text in corpus):
            raise ValueError("Training requires one or more text strings.")
        alphabet = sorted({char for text in corpus for char in text})
        self.id_to_token = list(self.special_tokens) + [
            char for char in alphabet if char not in self.special_tokens
        ]
        self.token_to_id = {token: index for index, token in enumerate(self.id_to_token)}
        self.trained = True
        return self

    def tokenize(self, text: str) -> list[str]:
        self._require_trained()
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        output: list[str] = []
        for is_special, piece in _split_special_tokens(text, self.special_tokens):
            if is_special:
                output.append(piece)
            else:
                output.extend(char if char in self.token_to_id else "<UNK>" for char in piece)
        return output

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.token_to_id[token] for token in self.tokenize(text)]
        if add_bos:
            ids.insert(0, self.token_to_id[self._special_by_role("BOS")])
        if add_eos:
            ids.append(self.token_to_id[self._special_by_role("EOS")])
        return ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        self._require_trained()
        output: list[str] = []
        specials = set(self.special_tokens)
        for token_id in ids:
            token = self._id_to_token(token_id)
            if not (skip_special_tokens and token in specials):
                output.append(token)
        return "".join(output)

    def batch_encode(
        self, texts: Iterable[str], *, add_bos: bool = False, add_eos: bool = False
    ) -> list[list[int]]:
        return [self.encode(text, add_bos=add_bos, add_eos=add_eos) for text in texts]

    def _special_by_role(self, role: str) -> str:
        candidate = f"<{role}>"
        if candidate not in self.special_tokens:
            raise ValueError(f"{candidate} is not configured as a special token")
        return candidate

    def _id_to_token(self, token_id: int) -> str:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("token IDs must be integers")
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError(f"token ID out of range: {token_id}")
        return self.id_to_token[token_id]

    def _require_trained(self) -> None:
        if not self.trained:
            raise RuntimeError("Train the tokenizer before using it.")


class BPETokenizer(CharacterTokenizer):
    """Deterministic character-based Byte Pair Encoding tokenizer."""
    FORMAT_VERSION = 1

    def __init__(self, special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS) -> None:
        super().__init__(special_tokens=special_tokens)
        self.merges: list[tuple[str, str]] = []
        self.base_tokens: set[str] = set()

    def train(
        self,
        texts: Iterable[str],
        *,
        vocab_size: int = 256,
        min_pair_frequency: int = 2,
    ) -> "BPETokenizer":
        corpus = list(texts)
        if not corpus or any(not isinstance(text, str) for text in corpus):
            raise ValueError("Training requires one or more text strings.")
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 1:
            raise ValueError("vocab_size must be a positive integer")
        if (
            isinstance(min_pair_frequency, bool)
            or not isinstance(min_pair_frequency, int)
            or min_pair_frequency < 1
        ):
            raise ValueError("min_pair_frequency must be a positive integer")

        alphabet = sorted({char for text in corpus for char in text})
        initial_vocab = list(self.special_tokens) + [
            char for char in alphabet if char not in self.special_tokens
        ]
        if vocab_size < len(initial_vocab):
            raise ValueError(
                f"vocab_size ({vocab_size}) is smaller than the initial vocabulary "
                f"({len(initial_vocab)})"
            )

        self.id_to_token = initial_vocab.copy()
        self.token_to_id = {token: index for index, token in enumerate(self.id_to_token)}
        self.base_tokens = set(initial_vocab)
        self.merges = []
        sequences = [list(text) for text in corpus]

        while len(self.id_to_token) < vocab_size:
            pair_counts: Counter[tuple[str, str]] = Counter()
            for sequence in sequences:
                pair_counts.update(zip(sequence, sequence[1:]))
            if not pair_counts:
                break
            highest_frequency = max(pair_counts.values())
            if highest_frequency < min_pair_frequency:
                break
            best_pair = min(
                pair for pair, count in pair_counts.items() if count == highest_frequency
            )
            merged_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            if merged_token not in self.token_to_id:
                self.token_to_id[merged_token] = len(self.id_to_token)
                self.id_to_token.append(merged_token)
            sequences = [self._merge_pair(sequence, best_pair) for sequence in sequences]

        self.trained = True
        return self

    @staticmethod
    def _merge_pair(sequence: list[str], pair: tuple[str, str]) -> list[str]:
        merged: list[str] = []
        index = 0
        while index < len(sequence):
            if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                merged.append(pair[0] + pair[1])
                index += 2
            else:
                merged.append(sequence[index])
                index += 1
        return merged

    def tokenize(self, text: str) -> list[str]:
        self._require_trained()
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        output: list[str] = []
        for is_special, piece in _split_special_tokens(text, self.special_tokens):
            if is_special:
                output.append(piece)
                continue
            sequence = [char if char in self.base_tokens else "<UNK>" for char in piece]
            for pair in self.merges:
                sequence = self._merge_pair(sequence, pair)
            output.extend(sequence)
        return output

    def save(self, path: str | Path) -> None:
        self._require_trained()
        payload = {
            "format_version": self.FORMAT_VERSION,
            "type": "character_bpe",
            "special_tokens": list(self.special_tokens),
            "vocabulary": self.id_to_token,
            "merges": [list(pair) for pair in self.merges],
            "base_tokens": sorted(self.base_tokens),
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("Unsupported tokenizer file version")
        if payload.get("type") != "character_bpe":
            raise ValueError("File does not contain a character BPE tokenizer")
        tokenizer = cls(payload["special_tokens"])
        vocabulary = payload["vocabulary"]
        if not isinstance(vocabulary, list) or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("Tokenizer vocabulary is invalid")
        if vocabulary[: len(tokenizer.special_tokens)] != list(tokenizer.special_tokens):
            raise ValueError("Special tokens must appear first in the vocabulary")
        merges = payload["merges"]
        if not isinstance(merges, list) or any(
            not isinstance(pair, list) or len(pair) != 2
            or not all(isinstance(part, str) for part in pair)
            for pair in merges
        ):
            raise ValueError("Tokenizer merge rules are invalid")
        tokenizer.id_to_token = vocabulary
        tokenizer.token_to_id = {token: index for index, token in enumerate(vocabulary)}
        tokenizer.merges = [(pair[0], pair[1]) for pair in merges]
        tokenizer.base_tokens = set(payload["base_tokens"])
        if not set(tokenizer.special_tokens).issubset(tokenizer.base_tokens):
            raise ValueError("Tokenizer base vocabulary is invalid")
        if not tokenizer.base_tokens.issubset(set(vocabulary)):
            raise ValueError("Tokenizer base vocabulary is not present in vocabulary")
        tokenizer.trained = True
        return tokenizer


class ByteBPETokenizer:
    """UTF-8 byte-level BPE tokenizer with complete Unicode coverage.

    Unlike the educational character BPE above, every possible input string can
    be encoded without an unknown-token fallback. The tokenizer and merge rules
    are learned from the supplied corpus and serialized as plain JSON.
    """

    FORMAT_VERSION = 1

    def __init__(self, special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS) -> None:
        self.special_tokens = _validate_special_tokens(special_tokens)
        self.special_to_id = {token: index for index, token in enumerate(self.special_tokens)}
        self.id_to_token: list[str] = list(self.special_tokens)
        self.token_to_id: dict[str, int] = dict(self.special_to_id)
        self.token_bytes: list[bytes | None] = [None] * len(self.special_tokens)
        self.merges: list[tuple[int, int, int]] = []
        self.trained = False

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def train(
        self,
        texts: Iterable[str],
        *,
        vocab_size: int = 512,
        min_pair_frequency: int = 2,
    ) -> "ByteBPETokenizer":
        corpus = list(texts)
        if not corpus or any(not isinstance(text, str) for text in corpus):
            raise ValueError("Training requires one or more text strings.")
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
            raise TypeError("vocab_size must be an integer")
        minimum = len(self.special_tokens) + 256
        if vocab_size < minimum:
            raise ValueError(f"vocab_size must be at least {minimum} for all UTF-8 bytes")
        if isinstance(min_pair_frequency, bool) or not isinstance(min_pair_frequency, int) or min_pair_frequency < 1:
            raise ValueError("min_pair_frequency must be a positive integer")

        self.id_to_token = list(self.special_tokens)
        self.token_to_id = dict(self.special_to_id)
        self.token_bytes = [None] * len(self.special_tokens)
        byte_offset = len(self.special_tokens)
        for value in range(256):
            token_id = byte_offset + value
            label = f"<0x{value:02X}>"
            self.id_to_token.append(label)
            self.token_to_id[label] = token_id
            self.token_bytes.append(bytes((value,)))

        sequences: list[list[int]] = []
        for text in corpus:
            for is_special, piece in _split_special_tokens(text, self.special_tokens):
                if is_special:
                    sequences.append([self.special_to_id[piece]])
                elif piece:
                    sequences.append([byte_offset + value for value in piece.encode("utf-8")])

        self.merges = []
        while len(self.id_to_token) < vocab_size:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                pair_counts.update(zip(sequence, sequence[1:]))
            if not pair_counts:
                break
            highest = max(pair_counts.values())
            if highest < min_pair_frequency:
                break
            best_pair = min(pair for pair, count in pair_counts.items() if count == highest)
            left, right = best_pair
            merged_id = len(self.id_to_token)
            merged_bytes = self.token_bytes[left] + self.token_bytes[right]
            self.token_bytes.append(merged_bytes)
            label = f"<BPE:{merged_id}:{merged_bytes.hex()}>"
            self.id_to_token.append(label)
            self.token_to_id[label] = merged_id
            self.merges.append((left, right, merged_id))
            sequences = [self._merge_ids(sequence, left, right, merged_id) for sequence in sequences]

        self.trained = True
        return self

    @staticmethod
    def _merge_ids(sequence: list[int], left: int, right: int, merged: int) -> list[int]:
        output: list[int] = []
        index = 0
        while index < len(sequence):
            if index + 1 < len(sequence) and sequence[index] == left and sequence[index + 1] == right:
                output.append(merged)
                index += 2
            else:
                output.append(sequence[index])
                index += 1
        return output

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        self._require_trained()
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        output: list[int] = []
        byte_offset = len(self.special_tokens)
        for is_special, piece in _split_special_tokens(text, self.special_tokens):
            if is_special:
                output.append(self.special_to_id[piece])
                continue
            sequence = [byte_offset + value for value in piece.encode("utf-8")]
            for left, right, merged in self.merges:
                sequence = self._merge_ids(sequence, left, right, merged)
            output.extend(sequence)
        if add_bos:
            output.insert(0, self.special_to_id["<BOS>"])
        if add_eos:
            output.append(self.special_to_id["<EOS>"])
        return output

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        self._require_trained()
        payload = bytearray()
        pieces: list[str] = []
        for raw_id in ids:
            if isinstance(raw_id, bool) or not isinstance(raw_id, Integral):
                raise TypeError("token IDs must be integers")
            token_id = int(raw_id)
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"token ID out of range: {token_id}")
            token_bytes = self.token_bytes[token_id]
            if token_bytes is None:
                if not skip_special_tokens:
                    pieces.append(payload.decode("utf-8", errors="replace"))
                    payload.clear()
                    pieces.append(self.id_to_token[token_id])
            else:
                payload.extend(token_bytes)
        pieces.append(payload.decode("utf-8", errors="replace"))
        return "".join(pieces)

    def save(self, path: str | Path) -> None:
        self._require_trained()
        payload = {
            "format_version": self.FORMAT_VERSION,
            "type": "utf8_byte_bpe",
            "special_tokens": list(self.special_tokens),
            "vocabulary": self.id_to_token,
            "token_bytes_hex": [None if value is None else value.hex() for value in self.token_bytes],
            "merges": [list(rule) for rule in self.merges],
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != cls.FORMAT_VERSION or payload.get("type") != "utf8_byte_bpe":
            raise ValueError("Unsupported or invalid byte BPE tokenizer file")
        tokenizer = cls(payload["special_tokens"])
        vocabulary = payload.get("vocabulary")
        encoded_bytes = payload.get("token_bytes_hex")
        merges = payload.get("merges")
        if (not isinstance(vocabulary, list) or any(not isinstance(token, str) for token in vocabulary)
                or len(set(vocabulary)) != len(vocabulary) or not isinstance(encoded_bytes, list)
                or len(vocabulary) != len(encoded_bytes)):
            raise ValueError("Tokenizer vocabulary data is invalid")
        if vocabulary[:len(tokenizer.special_tokens)] != list(tokenizer.special_tokens):
            raise ValueError("Special tokens must appear first")
        if len(vocabulary) < len(tokenizer.special_tokens) + 256:
            raise ValueError("Byte vocabulary is incomplete")
        token_bytes: list[bytes | None] = []
        for value in encoded_bytes:
            if value is None:
                token_bytes.append(None)
            elif isinstance(value, str):
                try:
                    token_bytes.append(bytes.fromhex(value))
                except ValueError as exc:
                    raise ValueError("Tokenizer contains invalid byte data") from exc
            else:
                raise ValueError("Tokenizer byte entries must be hex strings or null")
        if any(value is not None for value in token_bytes[:len(tokenizer.special_tokens)]):
            raise ValueError("Special tokens cannot contain byte payloads")
        if any(value is None or len(value) == 0 for value in token_bytes[len(tokenizer.special_tokens):]):
            raise ValueError("Regular byte-BPE tokens must contain one or more bytes")
        if any(token_bytes[len(tokenizer.special_tokens) + index] != bytes((index,)) for index in range(256)):
            raise ValueError("Base byte vocabulary is invalid")
        if not isinstance(merges, list):
            raise ValueError("Tokenizer merge rules are invalid")
        normalized_merges: list[tuple[int, int, int]] = []
        for expected_id, rule in enumerate(merges, start=len(tokenizer.special_tokens) + 256):
            if not isinstance(rule, list) or len(rule) != 3 or any(isinstance(item, bool) or not isinstance(item, int) for item in rule):
                raise ValueError("Tokenizer merge rule is invalid")
            left, right, merged = rule
            if left < 0 or right < 0 or left >= expected_id or right >= expected_id or merged != expected_id:
                raise ValueError("Tokenizer merge rule references invalid token IDs")
            if token_bytes[merged] != token_bytes[left] + token_bytes[right]:
                raise ValueError("Tokenizer merge payload does not match its inputs")
            normalized_merges.append((left, right, merged))
        if len(vocabulary) != len(tokenizer.special_tokens) + 256 + len(normalized_merges):
            raise ValueError("Tokenizer vocabulary and merge count do not match")
        tokenizer.id_to_token = vocabulary
        tokenizer.token_bytes = token_bytes
        tokenizer.token_to_id = {token: index for index, token in enumerate(vocabulary)}
        tokenizer.merges = normalized_merges
        tokenizer.trained = True
        return tokenizer

    def _require_trained(self) -> None:
        if not self.trained:
            raise RuntimeError("Train the tokenizer before using it.")


if __name__ == "__main__":
    sample_corpus = [
        "low low low lower lower lowest",
        "low lower lowest",
        "newer wider",
        "artificial intelligence learns patterns from data",
        "artificial intelligence learns from examples",
    ]
    character_tokenizer = CharacterTokenizer().train(sample_corpus)
    bpe_tokenizer = BPETokenizer().train(sample_corpus, vocab_size=80)
    sample = "low lower"
    print("AI Experiment — Experiment 006: Tokenizer")
    print("=" * 52)
    print(f"Training documents: {len(sample_corpus)}")
    print(f"Character vocabulary size: {character_tokenizer.vocab_size}")
    print(f"BPE vocabulary size: {bpe_tokenizer.vocab_size}")
    print(f"BPE merge operations learned: {len(bpe_tokenizer.merges)}")
    print(f"\nOriginal text: {sample!r}")
    print(f"Character tokens: {character_tokenizer.tokenize(sample)}")
    print(f"BPE tokens:       {bpe_tokenizer.tokenize(sample)}")
    encoded = bpe_tokenizer.encode(sample, add_bos=True, add_eos=True)
    print(f"Encoded IDs:      {encoded}")
    print(f"Decoded text:     {bpe_tokenizer.decode(encoded, skip_special_tokens=True)!r}")
