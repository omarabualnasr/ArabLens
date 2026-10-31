"""Generate text using an Experiment 023 neural-model checkpoint."""
from __future__ import annotations

import argparse
import importlib.util
import sys
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
_load_module("neural_core", EXPERIMENT_023 / "Neural Core.py")
_training_module = _load_module("training", EXPERIMENT_023 / "Training.py")
ByteBPETokenizer = _tokenizer_module.ByteBPETokenizer
generate_text = _training_module.generate_text
load_checkpoint = _training_module.load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with the custom Experiment 023 model")
    parser.add_argument("prompt", nargs="?", default="Artificial intelligence")
    parser.add_argument("--tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-dir", type=Path, default=HERE / "Checkpoints")
    args = parser.parse_args()

    model, metadata = load_checkpoint(args.checkpoint_dir / "Neural Model.npz")
    tokenizer = ByteBPETokenizer.load(args.checkpoint_dir / "Byte Tokenizer.json")
    if tokenizer.vocab_size != model.vocab_size:
        raise SystemExit("Tokenizer vocabulary does not match the model checkpoint")
    print(f"Model: {metadata['architecture']} | {metadata['parameter_count']:,} parameters")
    print(generate_text(
        model, tokenizer, args.prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
