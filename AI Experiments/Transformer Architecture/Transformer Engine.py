"""Experiment 008: a tiny decoder-only Transformer built with NumPy autograd.

The implementation reuses the project's own Tensor engine (Experiment 005).
It does not use PyTorch, TensorFlow, pretrained weights, or hosted AI APIs.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

# Import the project's own Tensor implementation from the numbered experiment folder.
from tensor_engine import Tensor


def _stable_softmax(values: Tensor, axis: int = -1) -> Tensor:
    """Softmax expressed with Tensor operations so gradients remain connected.

    The row maximum is a numerical-stability shift. Treating this shift as a
    constant does not change the softmax derivative because softmax is shift
    invariant and its gradient sums to zero along the normalized axis.
    """
    shifted = values - np.max(values.data, axis=axis, keepdims=True)
    exponentials = shifted.exp()
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


def _gelu(values: Tensor) -> Tensor:
    """GELU using the tanh approximation common in Transformer models."""
    coefficient = math.sqrt(2.0 / math.pi)
    inner = coefficient * (values + 0.044715 * (values ** 3.0))
    return 0.5 * values * (1.0 + inner.tanh())


class Embedding:
    """Trainable token or position lookup table with scatter-add gradients."""

    def __init__(self, count: int, width: int, rng: np.random.Generator, scale: float = 0.02):
        if count <= 0 or width <= 0:
            raise ValueError("Embedding count and width must be positive")
        self.weight = Tensor(rng.normal(0.0, scale, size=(count, width)))

    def forward(self, indices: object) -> Tensor:
        ids = np.asarray(indices)
        if ids.dtype.kind not in "iu" or ids.ndim < 1:
            raise TypeError("Embedding indices must be a one-or-more-dimensional integer array")
        if np.any(ids < 0) or np.any(ids >= self.weight.shape[0]):
            raise ValueError("Embedding index is outside the vocabulary/table range")
        ids = ids.astype(np.int64, copy=False)
        output = self.weight.data[ids]

        def backward(result: Tensor):
            def apply() -> None:
                if self.weight.requires_grad:
                    # Repeated IDs must accumulate gradients, not overwrite them.
                    np.add.at(self.weight.grad, ids.reshape(-1), result.grad.reshape(-1, self.weight.shape[1]))
            return apply

        return self.weight._result(output, (self.weight,), "embedding_lookup", backward)


class LayerNorm:
    """Layer normalization over the final feature dimension."""

    def __init__(self, width: int, epsilon: float = 1e-5):
        if width <= 0 or epsilon <= 0:
            raise ValueError("width and epsilon must be positive")
        self.width = width
        self.epsilon = float(epsilon)
        self.gamma = Tensor(np.ones(width))
        self.beta = Tensor(np.zeros(width))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.width:
            raise ValueError(f"LayerNorm expected final dimension {self.width}")
        mean = inputs.mean(axis=-1, keepdims=True)
        centered = inputs - mean
        variance = (centered ** 2.0).mean(axis=-1, keepdims=True)
        normalized = centered / (variance + self.epsilon) ** 0.5
        return normalized * self.gamma + self.beta


class CausalMultiHeadSelfAttention:
    """Multi-head self-attention with a causal mask and Tensor autograd."""

    def __init__(self, d_model: int, num_heads: int, rng: np.random.Generator):
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        scale = 1.0 / math.sqrt(d_model)
        self.wq = Tensor(rng.normal(0.0, scale, (d_model, d_model)))
        self.wk = Tensor(rng.normal(0.0, scale, (d_model, d_model)))
        self.wv = Tensor(rng.normal(0.0, scale, (d_model, d_model)))
        self.wo = Tensor(rng.normal(0.0, scale, (d_model, d_model)))
        self.last_attention_weights: np.ndarray | None = None

    def forward(self, inputs: Tensor) -> Tensor:
        batch, sequence, _ = inputs.shape
        q = (inputs @ self.wq).reshape(batch, sequence, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        k = (inputs @ self.wk).reshape(batch, sequence, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        v = (inputs @ self.wv).reshape(batch, sequence, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))

        scores = (q @ k.transpose((0, 1, 3, 2))) / math.sqrt(self.head_dim)
        causal_mask = np.triu(np.ones((sequence, sequence), dtype=bool), k=1)
        mask_bias = Tensor(
            np.where(causal_mask[None, None, :, :], -1e9, 0.0),
            requires_grad=False,
        )
        masked_scores = scores + mask_bias
        weights = _stable_softmax(masked_scores)
        self.last_attention_weights = weights.data.copy()
        attended = weights @ v
        merged = attended.transpose((0, 2, 1, 3)).reshape(batch, sequence, self.d_model)
        return merged @ self.wo


class FeedForward:
    """Position-wise two-layer MLP with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, rng: np.random.Generator):
        if d_ff <= 0:
            raise ValueError("d_ff must be positive")
        self.w1 = Tensor(rng.normal(0.0, 1.0 / math.sqrt(d_model), (d_model, d_ff)))
        self.b1 = Tensor(np.zeros(d_ff))
        self.w2 = Tensor(rng.normal(0.0, 1.0 / math.sqrt(d_ff), (d_ff, d_model)))
        self.b2 = Tensor(np.zeros(d_model))

    def forward(self, inputs: Tensor) -> Tensor:
        return _gelu(inputs @ self.w1 + self.b1) @ self.w2 + self.b2


class TransformerBlock:
    """Pre-normalized decoder block: masked attention + residual + MLP + residual."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, rng: np.random.Generator):
        self.norm1 = LayerNorm(d_model)
        self.attention = CausalMultiHeadSelfAttention(d_model, num_heads, rng)
        self.norm2 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, rng)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs + self.attention.forward(self.norm1.forward(inputs))
        return hidden + self.feed_forward.forward(self.norm2.forward(hidden))

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)


class TinyTransformer:
    """Small decoder-only Transformer language-model backbone.

    Input token IDs: (batch, sequence). Output logits: (batch, sequence, vocab).
    This phase builds the architecture and differentiable forward pass. The
    next experiment adds next-token loss, batching, and a training loop.
    """

    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        d_model: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 64,
        seed: int = 42,
    ):
        values = (vocab_size, max_sequence_length, d_model, num_heads, num_layers, d_ff)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("Model dimensions must be integers")
        if vocab_size < 2 or max_sequence_length < 1 or d_model < 2 or num_layers < 1 or d_ff < 1:
            raise ValueError("Model dimensions must be positive (vocab_size >= 2, d_model >= 2, num_layers >= 1)")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        rng = np.random.default_rng(seed)
        self.token_embedding = Embedding(vocab_size, d_model, rng)
        self.position_embedding = Embedding(max_sequence_length, d_model, rng)
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff, rng) for _ in range(num_layers)]
        self.final_norm = LayerNorm(d_model)

    def forward(self, token_ids: object) -> Tensor:
        ids = np.asarray(token_ids)
        if ids.dtype.kind not in "iu":
            raise TypeError("token_ids must be an integer array")
        if ids.ndim != 2 or ids.shape[0] < 1 or ids.shape[1] < 1:
            raise ValueError("token_ids must have shape (batch, sequence) with non-empty dimensions")
        if ids.shape[1] > self.max_sequence_length:
            raise ValueError("Input sequence exceeds max_sequence_length")
        if np.any(ids < 0) or np.any(ids >= self.vocab_size):
            raise ValueError("token ID is outside the configured vocabulary")

        positions = np.broadcast_to(np.arange(ids.shape[1], dtype=np.int64), ids.shape)
        hidden = self.token_embedding.forward(ids) + self.position_embedding.forward(positions)
        for block in self.blocks:
            hidden = block.forward(hidden)
        hidden = self.final_norm.forward(hidden)
        # Tie the output projection to the token embedding matrix, as in many GPT models.
        return hidden @ self.token_embedding.weight.transpose((1, 0))

    __call__ = forward

    def parameters(self) -> list[Tensor]:
        """Return every unique trainable tensor in this model."""
        params: list[Tensor] = [self.token_embedding.weight, self.position_embedding.weight]
        for block in self.blocks:
            params.extend((block.norm1.gamma, block.norm1.beta))
            params.extend((block.attention.wq, block.attention.wk, block.attention.wv, block.attention.wo))
            params.extend((block.norm2.gamma, block.norm2.beta))
            params.extend((block.feed_forward.w1, block.feed_forward.b1,
                           block.feed_forward.w2, block.feed_forward.b2))
        params.extend((self.final_norm.gamma, self.final_norm.beta))
        # Output weights are tied to token_embedding.weight, so they appear once.
        return params

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def config(self) -> dict[str, int]:
        return {
            "vocab_size": self.vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "d_ff": self.d_ff,
        }


def parameter_count(parameters: Iterable[Tensor]) -> int:
    """Count scalar trainable parameters."""
    return int(sum(parameter.data.size for parameter in parameters))
