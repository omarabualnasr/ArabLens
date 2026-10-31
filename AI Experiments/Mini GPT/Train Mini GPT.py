"""Train the project's modern, from-scratch decoder model (Experiment 023)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import numpy as np
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT_023 = ROOT / "AI Experiments" / "Neural Language Model"
TOKENIZER_DIR = ROOT / "AI Experiments" / "Tokenizer"
TENSOR_DIR = ROOT / "AI Experiments" / "Tensor Engine"
TRANSFORMER_DIR = ROOT / "AI Experiments" / "Transformer Architecture"
for folder in (EXPERIMENT_023, TOKENIZER_DIR, TENSOR_DIR, TRANSFORMER_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def _load_module(module_name: str, path: Path):
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load model component: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


_load_module("tensor_engine", TENSOR_DIR / "Tensor Engine.py")
_load_module("transformer_engine", TRANSFORMER_DIR / "Transformer Engine.py")
_tokenizer_module = _load_module("tokenizer", TOKENIZER_DIR / "Tokenizer.py")
_neural_module = _load_module("neural_core", EXPERIMENT_023 / "Neural Core.py")
_training_module = _load_module("training", EXPERIMENT_023 / "Training.py")
ModernTransformer = _neural_module.ModernTransformer
count_parameters = _neural_module.count_parameters
ByteBPETokenizer = _tokenizer_module.ByteBPETokenizer
generate_text = _training_module.generate_text
save_checkpoint = _training_module.save_checkpoint
split_token_stream = _training_module.split_token_stream
train_language_model = _training_module.train_language_model

DEFAULT_CORPUS = HERE / "Training Data" / "Sample Corpus.txt"
KNOWLEDGE_PATH = ROOT / "AI Experiments" / "Tiny AI Engine" / "Knowledge.json"
PRESETS = {
    "tiny": {"d_model": 32, "num_heads": 4, "num_kv_heads": 2, "num_layers": 2, "d_ff": 96, "context": 64},
    "small": {"d_model": 64, "num_heads": 4, "num_kv_heads": 2, "num_layers": 4, "d_ff": 192, "context": 128},
    "medium": {"d_model": 128, "num_heads": 8, "num_kv_heads": 4, "num_layers": 6, "d_ff": 384, "context": 256},
}


def build_training_text(corpus_path: Path, *, include_knowledge: bool = True) -> str:
    text = corpus_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("The supplied training corpus is empty")
    if include_knowledge and KNOWLEDGE_PATH.exists():
        entries = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
        sections = [text]
        for entry in entries:
            question = str(entry.get("question", "")).strip()
            answer = str(entry.get("answer", "")).strip()
            if question and answer:
                sections.append(f"User: {question}\nAssistant: {answer}")
                for alternate in entry.get("alternate_questions", []):
                    if isinstance(alternate, str) and alternate.strip():
                        sections.append(f"User: {alternate.strip()}\nAssistant: {answer}")
        text = "\n\n".join(sections)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train AI Experiment's custom decoder-only language model from random initialization"
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="small")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--kv-heads", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.0015)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="User: What is artificial intelligence?\nAssistant:")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--checkpoint-dir", type=Path, default=HERE / "Checkpoints")
    parser.add_argument("--no-include-knowledge", action="store_true")
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.sequence_length < 2:
        raise SystemExit("steps, batch-size and sequence-length must be positive")
    preset = PRESETS[args.preset]
    config = {
        "d_model": args.d_model or preset["d_model"],
        "num_heads": args.heads or preset["num_heads"],
        "num_kv_heads": args.kv_heads or preset["num_kv_heads"],
        "num_layers": args.layers or preset["num_layers"],
        "d_ff": args.d_ff or preset["d_ff"],
        "max_sequence_length": args.context_length or preset["context"],
    }
    if args.sequence_length > config["max_sequence_length"]:
        raise SystemExit("sequence-length cannot exceed the model context-length")

    try:
        text = build_training_text(args.corpus, include_knowledge=not args.no_include_knowledge)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not prepare training text: {exc}") from exc

    tokenizer = ByteBPETokenizer().train([text], vocab_size=max(args.vocab_size, 260), min_pair_frequency=2)
    token_ids = np.asarray(tokenizer.encode(text, add_bos=True, add_eos=True), dtype=np.int64)
    try:
        train_tokens, validation_tokens = split_token_stream(
            token_ids, validation_fraction=0.08, sequence_length=args.sequence_length,
        )
    except ValueError as exc:
        raise SystemExit(f"Training corpus is too small: {exc}. Add more text or lower --sequence-length.") from exc

    model = ModernTransformer(
        vocab_size=tokenizer.vocab_size,
        seed=args.seed,
        dropout=args.dropout,
        **config,
    )
    print("AI Experiment — Experiment 023: From-Scratch Neural Language Model")
    print("=" * 72)
    print(f"Training characters : {len(text):,}")
    print(f"Training tokens     : {len(train_tokens):,}")
    print(f"Validation tokens   : {len(validation_tokens):,}")
    print(f"Byte-BPE vocabulary : {tokenizer.vocab_size:,}")
    print(f"Architecture        : {model.ARCHITECTURE}")
    print(f"Model configuration : {config}")
    print(f"Trainable params    : {count_parameters(model):,}")
    print("\nTraining from random initialization — no pretrained weights...\n")

    started = time.perf_counter()
    result = train_language_model(
        model,
        train_tokens,
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation,
        validation_tokens=validation_tokens,
        eval_interval=max(1, args.steps // 10),
        seed=args.seed + 1,
        log_every=max(1, args.steps // 10),
    )
    elapsed = time.perf_counter() - started
    history = result["history"]
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / "Neural Model.npz"
    tokenizer_path = args.checkpoint_dir / "Byte Tokenizer.json"
    metrics_path = args.checkpoint_dir / "Training Metrics.json"
    save_checkpoint(checkpoint_path, model, training_steps=args.steps, history=history)
    tokenizer.save(tokenizer_path)
    metrics_path.write_text(json.dumps({
        "architecture": model.ARCHITECTURE,
        "model_config": model.config(),
        "training_characters": len(text),
        "training_tokens": len(train_tokens),
        "validation_tokens": len(validation_tokens),
        "elapsed_seconds": round(elapsed, 3),
        "history": history,
    }, indent=2), encoding="utf-8")

    generated = generate_text(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=0.8, top_k=40, top_p=0.92,
        repetition_penalty=1.08, seed=args.seed + 2,
    )
    print("\nTraining complete")
    print(f"Final training loss : {history['train_loss'][-1]:.4f}")
    if history["validation"]:
        best = min(history["validation"], key=lambda item: item["loss"])
        print(f"Best validation loss: {best['loss']:.4f} at step {best['step']}")
    print(f"Elapsed time        : {elapsed:.1f}s")
    print(f"Checkpoint          : {checkpoint_path}")
    print(f"Tokenizer           : {tokenizer_path}")
    print(f"Metrics             : {metrics_path}")
    print(f"\nPrompt   : {args.prompt!r}")
    print(f"Generated: {generated!r}")
    print("\nNote: capability depends on training-data volume and quality; this small local model is not a frontier-scale assistant.")


if __name__ == "__main__":
    main()
