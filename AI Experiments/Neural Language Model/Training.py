"""Training utilities for Experiment 023's from-scratch language model."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from neural_core import ModernTransformer, count_parameters
from tensor_engine import Tensor


def cross_entropy_loss(logits: Tensor, targets: object, *, label_smoothing: float = 0.0) -> Tensor:
    """Stable next-token cross entropy with an explicit gradient implementation."""
    target_ids = np.asarray(targets)
    if target_ids.dtype.kind not in "iu":
        raise TypeError("targets must be an integer array")
    if logits.ndim != 3 or target_ids.shape != logits.shape[:2]:
        raise ValueError("logits must be (batch, sequence, vocab) and targets (batch, sequence)")
    if target_ids.size == 0 or np.any(target_ids < 0) or np.any(target_ids >= logits.shape[-1]):
        raise ValueError("targets are empty or contain an out-of-range token ID")
    if not math.isfinite(label_smoothing) or not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")

    shifted = logits.data - np.max(logits.data, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
    log_probabilities = shifted - np.log(np.sum(exponentials, axis=-1, keepdims=True))
    rows = np.arange(target_ids.shape[0])[:, None]
    positions = np.arange(target_ids.shape[1])[None, :]
    nll = -log_probabilities[rows, positions, target_ids]
    if label_smoothing:
        smooth = -log_probabilities.mean(axis=-1)
        loss_value = float(((1.0 - label_smoothing) * nll + label_smoothing * smooth).mean())
    else:
        loss_value = float(nll.mean())

    def backward(result: Tensor):
        def apply() -> None:
            if logits.requires_grad:
                gradient = probabilities.copy()
                if label_smoothing:
                    gradient -= label_smoothing / probabilities.shape[-1]
                gradient[rows, positions, target_ids] -= 1.0 - label_smoothing
                gradient /= target_ids.size
                logits.grad += gradient * float(result.grad)
        return apply

    return logits._result(np.asarray(loss_value), (logits,), "language_model_cross_entropy", backward)


class AdamW:
    """AdamW optimizer with decoupled weight decay and global norm clipping."""

    def __init__(
        self,
        parameters: list[Tensor],
        *,
        learning_rate: float = 1.5e-3,
        beta1: float = 0.9,
        beta2: float = 0.95,
        epsilon: float = 1e-8,
        weight_decay: float = 0.1,
        clip_norm: float = 1.0,
    ):
        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("AdamW requires trainable parameters")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("AdamW beta values must be in [0, 1)")
        if epsilon <= 0 or not math.isfinite(epsilon):
            raise ValueError("epsilon must be positive and finite")
        if weight_decay < 0 or not math.isfinite(weight_decay):
            raise ValueError("weight_decay must be finite and non-negative")
        if clip_norm <= 0 or not math.isfinite(clip_norm):
            raise ValueError("clip_norm must be positive and finite")
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.weight_decay = float(weight_decay)
        self.clip_norm = float(clip_norm)
        self.step_count = 0
        self.first_moments = [np.zeros_like(parameter.data) for parameter in self.parameters]
        self.second_moments = [np.zeros_like(parameter.data) for parameter in self.parameters]

    def step(self) -> None:
        gradients = [parameter.grad for parameter in self.parameters]
        if any(not np.all(np.isfinite(gradient)) for gradient in gradients):
            raise FloatingPointError("Non-finite gradient; AdamW update aborted")
        norm = math.sqrt(sum(float(np.sum(gradient * gradient)) for gradient in gradients))
        scale = min(1.0, self.clip_norm / (norm + 1e-12))
        self.step_count += 1
        correction1 = 1.0 - self.beta1 ** self.step_count
        correction2 = 1.0 - self.beta2 ** self.step_count
        for index, (parameter, raw_gradient) in enumerate(zip(self.parameters, gradients)):
            gradient = raw_gradient * scale
            self.first_moments[index] = self.beta1 * self.first_moments[index] + (1.0 - self.beta1) * gradient
            self.second_moments[index] = self.beta2 * self.second_moments[index] + (1.0 - self.beta2) * (gradient * gradient)
            first_hat = self.first_moments[index] / correction1
            second_hat = self.second_moments[index] / correction2
            update = first_hat / (np.sqrt(second_hat) + self.epsilon)
            # Standard AdamW decay: decay matrix-like weights, not 1-D norm scales.
            if parameter.data.ndim >= 2 and self.weight_decay:
                parameter.data *= 1.0 - self.learning_rate * self.weight_decay
            parameter.data -= self.learning_rate * update
            if not np.all(np.isfinite(parameter.data)):
                raise FloatingPointError("AdamW produced non-finite model parameters")

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()


def make_next_token_batch(
    token_ids: np.ndarray,
    batch_size: int,
    sequence_length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(token_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError("token_ids must be a one-dimensional integer array")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer")
    if len(ids) <= sequence_length:
        raise ValueError("token stream must contain more tokens than sequence_length")
    starts = rng.integers(0, len(ids) - sequence_length, size=batch_size)
    offsets = np.arange(sequence_length)
    inputs = np.stack([ids[start + offsets] for start in starts]).astype(np.int64)
    targets = np.stack([ids[start + offsets + 1] for start in starts]).astype(np.int64)
    return inputs, targets


def split_token_stream(
    token_ids: np.ndarray,
    *,
    validation_fraction: float = 0.05,
    sequence_length: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Chronologically split a token stream to prevent train/validation leakage."""
    ids = np.asarray(token_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError("token_ids must be a one-dimensional integer array")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between zero and 0.5")
    if len(ids) < (sequence_length + 2) * 2:
        raise ValueError("Corpus is too short for a separate validation stream")
    validation_size = max(sequence_length + 1, int(len(ids) * validation_fraction))
    train_size = len(ids) - validation_size
    if train_size <= sequence_length:
        raise ValueError("Training split is too short for the configured sequence length")
    return ids[:train_size].astype(np.int64), ids[train_size:].astype(np.int64)


def cosine_learning_rate(
    step: int,
    total_steps: int,
    peak_learning_rate: float,
    *,
    warmup_steps: int = 0,
    minimum_ratio: float = 0.1,
) -> float:
    if total_steps < 1 or step < 1 or step > total_steps:
        raise ValueError("step must be in [1, total_steps]")
    if peak_learning_rate <= 0 or not math.isfinite(peak_learning_rate):
        raise ValueError("peak_learning_rate must be positive and finite")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio must be in [0, 1]")
    if warmup_steps > 0 and step <= warmup_steps:
        return peak_learning_rate * step / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_learning_rate * (minimum_ratio + (1.0 - minimum_ratio) * cosine)


def evaluate_model(
    model: ModernTransformer,
    token_ids: np.ndarray,
    *,
    sequence_length: int,
    batch_size: int = 8,
    max_batches: int = 32,
) -> float:
    ids = np.asarray(token_ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) <= sequence_length:
        raise ValueError("Evaluation stream must be one-dimensional and longer than sequence_length")
    was_training = model.training
    model.eval()
    losses: list[float] = []
    stride = sequence_length
    max_start = len(ids) - sequence_length - 1
    starts = list(range(0, max_start + 1, stride))[:max_batches * batch_size]
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset:offset + batch_size]
        x = np.stack([ids[start:start + sequence_length] for start in batch_starts])
        y = np.stack([ids[start + 1:start + sequence_length + 1] for start in batch_starts])
        loss = cross_entropy_loss(model(x), y)
        losses.append(loss.item())
    if was_training:
        model.train()
    if not losses:
        raise ValueError("No complete evaluation batches are available")
    return float(np.mean(losses))


def train_language_model(
    model: ModernTransformer,
    train_tokens: np.ndarray,
    *,
    steps: int = 1500,
    batch_size: int = 8,
    sequence_length: int = 96,
    learning_rate: float = 1.5e-3,
    warmup_steps: int = 50,
    minimum_lr_ratio: float = 0.1,
    weight_decay: float = 0.1,
    gradient_accumulation_steps: int = 1,
    validation_tokens: np.ndarray | None = None,
    eval_interval: int = 100,
    seed: int = 123,
    log_every: int = 100,
) -> dict[str, Any]:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if sequence_length > model.max_sequence_length:
        raise ValueError("sequence_length exceeds model context length")
    if isinstance(gradient_accumulation_steps, bool) or not isinstance(gradient_accumulation_steps, int) or gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError("warmup_steps must be a non-negative integer")
    if eval_interval < 1:
        raise ValueError("eval_interval must be positive")
    rng = np.random.default_rng(seed)
    optimizer = AdamW(model.parameters(), learning_rate=learning_rate, weight_decay=weight_decay)
    history: dict[str, Any] = {"train_loss": [], "validation": [], "learning_rate": []}

    for step in range(1, steps + 1):
        current_lr = cosine_learning_rate(
            step, steps, learning_rate, warmup_steps=min(warmup_steps, max(0, steps - 1)),
            minimum_ratio=minimum_lr_ratio,
        )
        optimizer.learning_rate = current_lr
        optimizer.zero_grad()
        model.train()
        accumulated_loss = 0.0
        for _ in range(gradient_accumulation_steps):
            inputs, targets = make_next_token_batch(train_tokens, batch_size, sequence_length, rng)
            loss = cross_entropy_loss(model(inputs), targets, label_smoothing=0.01)
            accumulated_loss += loss.item()
            (loss / gradient_accumulation_steps).backward()
        optimizer.step()
        mean_loss = accumulated_loss / gradient_accumulation_steps
        history["train_loss"].append(mean_loss)
        history["learning_rate"].append(current_lr)

        should_evaluate = validation_tokens is not None and (
            step == 1 or step % eval_interval == 0 or step == steps
        )
        if should_evaluate:
            validation_loss = evaluate_model(
                model, validation_tokens,
                sequence_length=sequence_length,
                batch_size=batch_size,
            )
            history["validation"].append({"step": step, "loss": validation_loss})
        if log_every and (step == 1 or step % log_every == 0 or step == steps):
            suffix = ""
            if history["validation"] and history["validation"][-1]["step"] == step:
                suffix = f" | val {history['validation'][-1]['loss']:.4f}"
            print(f"step {step:5d}/{steps} | train {mean_loss:.4f} | lr {current_lr:.2e}{suffix}")

    model.eval()
    history["steps"] = steps
    history["optimizer_steps"] = optimizer.step_count
    history["parameter_count"] = count_parameters(model)
    return {"history": history, "optimizer": optimizer}


def save_checkpoint(
    path: str | Path,
    model: ModernTransformer,
    *,
    training_steps: int = 0,
    history: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 1,
        "architecture": model.ARCHITECTURE,
        "model_config": model.config(),
        "training_steps": int(training_steps),
        "parameter_count": count_parameters(model),
        "history": history or {},
    }
    arrays = {f"parameter_{index:04d}": parameter.data for index, parameter in enumerate(model.parameters())}
    np.savez_compressed(destination, metadata=np.asarray(json.dumps(metadata)), **arrays)
    return destination


def load_checkpoint(path: str | Path) -> tuple[ModernTransformer, dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata"].item()))
        if metadata.get("format_version") != 1 or metadata.get("architecture") != ModernTransformer.ARCHITECTURE:
            raise ValueError("Unsupported neural-model checkpoint format or architecture")
        model = ModernTransformer(**metadata["model_config"])
        parameters = model.parameters()
        if count_parameters(model) != int(metadata.get("parameter_count", -1)):
            raise ValueError("Checkpoint parameter count does not match model configuration")
        for index, parameter in enumerate(parameters):
            key = f"parameter_{index:04d}"
            if key not in checkpoint:
                raise ValueError(f"Checkpoint is missing {key}")
            saved = checkpoint[key]
            if saved.shape != parameter.shape or not np.all(np.isfinite(saved)):
                raise ValueError(f"Checkpoint parameter {key} has an invalid shape or value")
            parameter.data[...] = saved
    model.eval()
    return model, metadata


def generate_text(
    model: ModernTransformer,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float = 0.92,
    repetition_penalty: float = 1.08,
    seed: int = 7,
) -> str:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
        raise ValueError("top_k must be a positive integer or None")
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if not math.isfinite(repetition_penalty) or repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be at least one")

    generated = tokenizer.encode(prompt, add_bos=True)
    if not generated:
        raise ValueError("prompt produced no tokens")
    rng = np.random.default_rng(seed)
    eos_id = tokenizer.token_to_id.get("<EOS>")
    was_training = model.training
    model.eval()
    context = generated[-model.max_sequence_length:]
    logits_tensor, key_value_cache = model.forward_with_cache(np.asarray([context], dtype=np.int64))
    next_logits = logits_tensor.data[0, -1].copy()

    for index in range(max_new_tokens):
        logits = next_logits.copy() / temperature
        if repetition_penalty > 1.0:
            for token_id in set(generated[-model.max_sequence_length:]):
                if logits[token_id] >= 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty
        if top_k is not None and top_k < logits.size:
            keep = np.argpartition(logits, -top_k)[-top_k:]
            filtered = np.full_like(logits, -np.inf)
            filtered[keep] = logits[keep]
            logits = filtered
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        if top_p < 1.0:
            sorted_ids = np.argsort(probabilities)[::-1]
            sorted_probabilities = probabilities[sorted_ids]
            cumulative = np.cumsum(sorted_probabilities)
            remove = cumulative - sorted_probabilities >= top_p
            probabilities[sorted_ids[remove]] = 0.0
            probabilities /= probabilities.sum()
        next_id = int(rng.choice(len(probabilities), p=probabilities))
        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
        if index + 1 < max_new_tokens:
            cached_length = key_value_cache[0][0].shape[2]
            if cached_length < model.max_sequence_length:
                next_logits_tensor, key_value_cache = model.forward_with_cache(
                    np.asarray([[next_id]], dtype=np.int64), key_value_cache,
                )
            else:
                # Slide the context window and re-prefill when the cache is full.
                context = generated[-model.max_sequence_length:]
                next_logits_tensor, key_value_cache = model.forward_with_cache(
                    np.asarray([context], dtype=np.int64), None,
                )
            next_logits = next_logits_tensor.data[0, -1].copy()
    if was_training:
        model.train()
    return tokenizer.decode(generated, skip_special_tokens=True)


__all__ = [
    "AdamW", "cross_entropy_loss", "make_next_token_batch", "split_token_stream",
    "cosine_learning_rate", "evaluate_model", "train_language_model",
    "save_checkpoint", "load_checkpoint", "generate_text",
]
