"""Experiment 023: a modern decoder-only language model built from scratch.

The architecture uses this repository's Tensor/autograd engine and NumPy only:
RMSNorm, rotary position embeddings, grouped-query causal attention, SwiGLU,
pre-normalized residual blocks, tied embeddings, and dropout.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for folder in (
    ROOT / "AI Experiments" / "Tensor Engine",
    ROOT / "AI Experiments" / "Transformer Architecture",
):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from tensor_engine import Tensor  # noqa: E402
from transformer_engine import Embedding, parameter_count  # noqa: E402


def _stable_softmax(values: Tensor, axis: int = -1) -> Tensor:
    shifted = values - np.max(values.data, axis=axis, keepdims=True)
    exponentials = shifted.exp()
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


def _dropout(values: Tensor, probability: float, rng: np.random.Generator, training: bool) -> Tensor:
    if not training or probability <= 0.0:
        return values
    keep_probability = 1.0 - probability
    mask = (rng.random(values.shape) < keep_probability).astype(np.float64) / keep_probability
    output = values.data * mask

    def backward(result: Tensor):
        def apply() -> None:
            if values.requires_grad:
                values.grad += result.grad * mask
        return apply

    return values._result(output, (values,), "dropout", backward)


def apply_rotary_embedding(values: Tensor, *, base: float = 10000.0, position_offset: int = 0) -> Tensor:
    """Apply RoPE to (batch, heads, sequence, head_dim), with custom backward."""
    if values.ndim != 4:
        raise ValueError("Rotary embeddings expect (batch, heads, sequence, head_dim)")
    batch, heads, sequence, head_dim = values.shape
    if head_dim % 2:
        raise ValueError("Rotary embedding head dimension must be even")
    if not math.isfinite(base) or base <= 1.0:
        raise ValueError("Rotary base must be finite and greater than one")
    if isinstance(position_offset, bool) or not isinstance(position_offset, int) or position_offset < 0:
        raise ValueError("position_offset must be a non-negative integer")

    pair_count = head_dim // 2
    inverse_frequencies = 1.0 / (base ** (np.arange(pair_count, dtype=np.float64) * 2.0 / head_dim))
    angles = (position_offset + np.arange(sequence, dtype=np.float64))[:, None] * inverse_frequencies[None, :]
    cosine = np.cos(angles)[None, None, :, :]
    sine = np.sin(angles)[None, None, :, :]
    pairs = values.data.reshape(batch, heads, sequence, pair_count, 2)
    first = pairs[..., 0]
    second = pairs[..., 1]
    output_pairs = np.stack((first * cosine - second * sine, first * sine + second * cosine), axis=-1)
    output = output_pairs.reshape(values.shape)

    def backward(result: Tensor):
        def apply() -> None:
            if values.requires_grad:
                gradient_pairs = result.grad.reshape(batch, heads, sequence, pair_count, 2)
                grad_first = gradient_pairs[..., 0] * cosine + gradient_pairs[..., 1] * sine
                grad_second = -gradient_pairs[..., 0] * sine + gradient_pairs[..., 1] * cosine
                values.grad += np.stack((grad_first, grad_second), axis=-1).reshape(values.shape)
        return apply

    return values._result(output, (values,), "rotary_embedding", backward)


def _repeat_key_value(values: Tensor, repeats: int) -> Tensor:
    """Repeat KV heads for grouped-query attention and sum their gradients back."""
    if repeats == 1:
        return values
    batch, kv_heads, sequence, head_dim = values.shape
    output = np.repeat(values.data, repeats, axis=1)

    def backward(result: Tensor):
        def apply() -> None:
            if values.requires_grad:
                grouped = result.grad.reshape(batch, kv_heads, repeats, sequence, head_dim)
                values.grad += grouped.sum(axis=2)
        return apply

    return values._result(output, (values,), "repeat_key_value", backward)


class RMSNorm:
    """Root-mean-square normalization with one learned scale per channel."""

    def __init__(self, width: int, epsilon: float = 1e-6):
        if width < 1 or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("RMSNorm width and epsilon must be positive")
        self.width = int(width)
        self.epsilon = float(epsilon)
        self.weight = Tensor(np.ones(width, dtype=np.float64))

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[-1] != self.width:
            raise ValueError(f"RMSNorm expected final dimension {self.width}")
        mean_square = (values ** 2.0).mean(axis=-1, keepdims=True)
        normalized = values * (mean_square + self.epsilon) ** -0.5
        return normalized * self.weight

    __call__ = forward


class GroupedQueryAttention:
    """Causal self-attention with RoPE and optional grouped-query KV heads."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        rng: np.random.Generator,
        *,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        residual_scale: float = 1.0,
    ):
        if d_model < 2 or num_heads < 1 or d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if num_heads % num_kv_heads:
            raise ValueError("num_kv_heads must divide num_heads")
        self.head_dim = d_model // num_heads
        if self.head_dim % 2:
            raise ValueError("d_model / num_heads must be even for rotary embeddings")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.kv_repeats = num_heads // num_kv_heads
        self.dropout = float(dropout)
        self.rope_base = float(rope_base)
        self.wq = Tensor(rng.normal(0.0, 0.02, (d_model, d_model)))
        kv_width = num_kv_heads * self.head_dim
        self.wk = Tensor(rng.normal(0.0, 0.02, (d_model, kv_width)))
        self.wv = Tensor(rng.normal(0.0, 0.02, (d_model, kv_width)))
        self.wo = Tensor(rng.normal(0.0, 0.02 * residual_scale, (d_model, d_model)))
        self.last_attention_weights: np.ndarray | None = None

    def forward(self, inputs: Tensor, rng: np.random.Generator, training: bool) -> Tensor:
        batch, sequence, _ = inputs.shape
        q = (inputs @ self.wq).reshape(batch, sequence, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        k = (inputs @ self.wk).reshape(batch, sequence, self.num_kv_heads, self.head_dim).transpose((0, 2, 1, 3))
        v = (inputs @ self.wv).reshape(batch, sequence, self.num_kv_heads, self.head_dim).transpose((0, 2, 1, 3))
        q = apply_rotary_embedding(q, base=self.rope_base)
        k = apply_rotary_embedding(k, base=self.rope_base)
        k = _repeat_key_value(k, self.kv_repeats)
        v = _repeat_key_value(v, self.kv_repeats)

        scores = (q @ k.transpose((0, 1, 3, 2))) / math.sqrt(self.head_dim)
        causal = np.triu(np.ones((sequence, sequence), dtype=bool), k=1)
        scores = scores + Tensor(np.where(causal[None, None, :, :], -1e9, 0.0), requires_grad=False)
        weights = _stable_softmax(scores)
        self.last_attention_weights = weights.data.copy()
        weights = _dropout(weights, self.dropout, rng, training)
        attended = weights @ v
        merged = attended.transpose((0, 2, 1, 3)).reshape(batch, sequence, self.d_model)
        return merged @ self.wo

    def forward_with_cache(
        self, inputs: Tensor, past_key_value: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[Tensor, tuple[np.ndarray, np.ndarray]]:
        """Inference-only attention path returning rotated-key and value caches."""
        if self.dropout:
            # Cache inference is deterministic regardless of configured train dropout.
            pass
        batch, sequence, _ = inputs.shape
        past_length = 0
        if past_key_value is not None:
            past_keys, past_values = past_key_value
            if past_keys.ndim != 4 or past_values.shape != past_keys.shape:
                raise ValueError("Cached keys and values must have matching 4-D shapes")
            if past_keys.shape[0] != batch or past_keys.shape[1] != self.num_kv_heads or past_keys.shape[3] != self.head_dim:
                raise ValueError("KV cache dimensions do not match this attention layer")
            past_length = past_keys.shape[2]

        q = (inputs @ self.wq).reshape(batch, sequence, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        new_keys = (inputs @ self.wk).reshape(batch, sequence, self.num_kv_heads, self.head_dim).transpose((0, 2, 1, 3))
        new_values = (inputs @ self.wv).reshape(batch, sequence, self.num_kv_heads, self.head_dim).transpose((0, 2, 1, 3))
        q = apply_rotary_embedding(q, base=self.rope_base, position_offset=past_length)
        new_keys = apply_rotary_embedding(new_keys, base=self.rope_base, position_offset=past_length)

        if past_key_value is None:
            all_keys, all_values = new_keys.data.copy(), new_values.data.copy()
        else:
            all_keys = np.concatenate((past_keys, new_keys.data), axis=2)
            all_values = np.concatenate((past_values, new_values.data), axis=2)
        key_tensor = Tensor(all_keys, requires_grad=False)
        value_tensor = Tensor(all_values, requires_grad=False)
        expanded_keys = _repeat_key_value(key_tensor, self.kv_repeats)
        expanded_values = _repeat_key_value(value_tensor, self.kv_repeats)
        scores = (q @ expanded_keys.transpose((0, 1, 3, 2))) / math.sqrt(self.head_dim)

        query_positions = past_length + np.arange(sequence)
        key_positions = np.arange(past_length + sequence)
        causal = key_positions[None, :] > query_positions[:, None]
        scores = scores + Tensor(np.where(causal[None, None, :, :], -1e9, 0.0), requires_grad=False)
        weights = _stable_softmax(scores)
        self.last_attention_weights = weights.data.copy()
        attended = weights @ expanded_values
        merged = attended.transpose((0, 2, 1, 3)).reshape(batch, sequence, self.d_model)
        return merged @ self.wo, (all_keys, all_values)


class SwiGLU:
    """Gated feed-forward network: SiLU(xW_gate) * (xW_value), then down-proj."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        rng: np.random.Generator,
        *,
        residual_scale: float = 1.0,
        dropout: float = 0.0,
    ):
        if d_model < 1 or d_ff < 1:
            raise ValueError("SwiGLU dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.dropout = float(dropout)
        self.w_gate = Tensor(rng.normal(0.0, 0.02, (d_model, d_ff)))
        self.w_value = Tensor(rng.normal(0.0, 0.02, (d_model, d_ff)))
        self.w_down = Tensor(rng.normal(0.0, 0.02 * residual_scale, (d_ff, d_model)))

    def forward(self, inputs: Tensor, rng: np.random.Generator, training: bool) -> Tensor:
        gate = inputs @ self.w_gate
        value = inputs @ self.w_value
        hidden = gate.sigmoid() * gate * value
        hidden = _dropout(hidden, self.dropout, rng, training)
        return hidden @ self.w_down


class ModernTransformerBlock:
    """Pre-norm Transformer block with grouped-query attention and SwiGLU."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        d_ff: int,
        rng: np.random.Generator,
        *,
        dropout: float,
        rope_base: float,
        residual_scale: float,
    ):
        self.attention_norm = RMSNorm(d_model)
        self.attention = GroupedQueryAttention(
            d_model, num_heads, num_kv_heads, rng,
            dropout=dropout, rope_base=rope_base, residual_scale=residual_scale,
        )
        self.feed_forward_norm = RMSNorm(d_model)
        self.feed_forward = SwiGLU(
            d_model, d_ff, rng, residual_scale=residual_scale, dropout=dropout,
        )
        self.dropout = float(dropout)

    def forward(self, inputs: Tensor, rng: np.random.Generator, training: bool) -> Tensor:
        attention_output = self.attention.forward(self.attention_norm(inputs), rng, training)
        hidden = inputs + _dropout(attention_output, self.dropout, rng, training)
        feed_forward_output = self.feed_forward.forward(self.feed_forward_norm(hidden), rng, training)
        return hidden + _dropout(feed_forward_output, self.dropout, rng, training)

    def forward_with_cache(
        self, inputs: Tensor, past_key_value: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[Tensor, tuple[np.ndarray, np.ndarray]]:
        normalized = self.attention_norm(inputs)
        attention_output, new_cache = self.attention.forward_with_cache(normalized, past_key_value)
        hidden = inputs + attention_output
        feed_forward_output = self.feed_forward.forward(self.feed_forward_norm(hidden), np.random.default_rng(0), False)
        return hidden + feed_forward_output, new_cache


class ModernTransformer:
    """Configurable decoder-only language model built on custom NumPy autograd."""

    ARCHITECTURE = "rope-rmsnorm-gqa-swiglu-decoder-v1"

    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 192,
        num_kv_heads: int | None = None,
        dropout: float = 0.05,
        rope_base: float = 10000.0,
        seed: int = 42,
    ):
        dimensions = (vocab_size, max_sequence_length, d_model, num_heads, num_layers, d_ff)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in dimensions):
            raise TypeError("Model dimensions must be integers")
        if vocab_size < 2 or max_sequence_length < 1 or d_model < 2 or num_layers < 1 or d_ff < 1:
            raise ValueError("Invalid model dimensions")
        if num_heads < 1 or d_model % num_heads or (d_model // num_heads) % 2:
            raise ValueError("d_model must be divisible by num_heads with an even head dimension")
        kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if isinstance(kv_heads, bool) or not isinstance(kv_heads, int) or kv_heads < 1 or num_heads % kv_heads:
            raise ValueError("num_kv_heads must be a positive divisor of num_heads")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(rope_base) or rope_base <= 1:
            raise ValueError("rope_base must be greater than one")

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = kv_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = float(dropout)
        self.rope_base = float(rope_base)
        self.training = True
        self.rng = np.random.default_rng(seed + 1009)
        rng = np.random.default_rng(seed)
        self.token_embedding = Embedding(vocab_size, d_model, rng, scale=0.02)
        residual_scale = 1.0 / math.sqrt(2.0 * num_layers)
        self.blocks = [
            ModernTransformerBlock(
                d_model, num_heads, kv_heads, d_ff, rng,
                dropout=dropout, rope_base=rope_base, residual_scale=residual_scale,
            )
            for _ in range(num_layers)
        ]
        self.final_norm = RMSNorm(d_model)

    def train(self) -> "ModernTransformer":
        self.training = True
        return self

    def eval(self) -> "ModernTransformer":
        self.training = False
        return self

    def forward(self, token_ids: object) -> Tensor:
        ids = np.asarray(token_ids)
        if ids.dtype.kind not in "iu":
            raise TypeError("token_ids must be an integer array")
        if ids.ndim != 2 or ids.shape[0] < 1 or ids.shape[1] < 1:
            raise ValueError("token_ids must have shape (batch, sequence) with non-empty dimensions")
        if ids.shape[1] > self.max_sequence_length:
            raise ValueError("Input sequence exceeds max_sequence_length")
        if np.any(ids < 0) or np.any(ids >= self.vocab_size):
            raise ValueError("Token ID is outside the model vocabulary")

        hidden = self.token_embedding.forward(ids)
        hidden = _dropout(hidden, self.dropout, self.rng, self.training)
        for block in self.blocks:
            hidden = block.forward(hidden, self.rng, self.training)
        hidden = self.final_norm(hidden)
        return hidden @ self.token_embedding.weight.transpose((1, 0))

    __call__ = forward

    def forward_with_cache(
        self,
        token_ids: object,
        past_key_values: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None,
    ) -> tuple[Tensor, tuple[tuple[np.ndarray, np.ndarray], ...]]:
        """Run inference with a per-layer KV cache for fast autoregressive decoding."""
        if self.training:
            raise RuntimeError("Call model.eval() before using the KV-cache inference path")
        ids = np.asarray(token_ids)
        if ids.dtype.kind not in "iu" or ids.ndim != 2 or ids.shape[0] < 1 or ids.shape[1] < 1:
            raise ValueError("token_ids must be a non-empty integer array with shape (batch, sequence)")
        if ids.shape[1] > self.max_sequence_length:
            raise ValueError("Input sequence exceeds model context length")
        if np.any(ids < 0) or np.any(ids >= self.vocab_size):
            raise ValueError("Token ID is outside the model vocabulary")
        if past_key_values is not None:
            if len(past_key_values) != self.num_layers:
                raise ValueError("KV cache must contain one entry per Transformer layer")
            first_keys = past_key_values[0][0]
            if first_keys.ndim != 4:
                raise ValueError("KV cache tensors must have four dimensions")
            past_length = first_keys.shape[2]
            if past_length + ids.shape[1] > self.max_sequence_length:
                raise ValueError("KV cache plus new tokens would exceed the model context length")
            if any(cache[0].shape[2] != past_length for cache in past_key_values):
                raise ValueError("All Transformer layers must have equally sized KV caches")

        hidden = self.token_embedding.forward(ids)
        caches: list[tuple[np.ndarray, np.ndarray]] = []
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[index]
            hidden, cache = block.forward_with_cache(hidden, past)
            caches.append(cache)
        logits = self.final_norm(hidden) @ self.token_embedding.weight.transpose((1, 0))
        return logits, tuple(caches)

    def parameters(self) -> list[Tensor]:
        parameters: list[Tensor] = [self.token_embedding.weight]
        for block in self.blocks:
            parameters.extend((block.attention_norm.weight,))
            parameters.extend((
                block.attention.wq, block.attention.wk,
                block.attention.wv, block.attention.wo,
            ))
            parameters.append(block.feed_forward_norm.weight)
            parameters.extend((
                block.feed_forward.w_gate,
                block.feed_forward.w_value,
                block.feed_forward.w_down,
            ))
        parameters.append(self.final_norm.weight)
        return parameters

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def config(self) -> dict[str, int | float]:
        return {
            "vocab_size": self.vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "d_ff": self.d_ff,
            "num_kv_heads": self.num_kv_heads,
            "dropout": self.dropout,
            "rope_base": self.rope_base,
        }


def count_parameters(model: ModernTransformer) -> int:
    return parameter_count(model.parameters())


__all__ = [
    "RMSNorm", "GroupedQueryAttention", "SwiGLU", "ModernTransformer",
    "apply_rotary_embedding", "count_parameters",
]
