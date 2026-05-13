"""Reclaim disk space by removing unused weight formats from the local model cache.

The Hugging Face snapshot for each model ships PyTorch, TensorFlow, Flax, ONNX,
OpenVINO and Rust weight variants. EarningsLens only loads the PyTorch path
(via `transformers` and `sentence-transformers`), so the rest is dead weight.

Running this script after a fresh install typically reclaims ~12 GB.
"""

from __future__ import annotations

import fnmatch
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from earningslens import config

# Patterns matched against each file's relative path inside a model directory.
PURGE_PATTERNS = [
    "onnx/*",
    "openvino/*",
    "runs/*",
    ".cache/*",
    "*.h5",
    "*.msgpack",
    "*.ot",
    "rust_model*",
    "*flax*",
    "*tf_model*",
    "train_script.py",
    "data_config.json",
    "trainer_state.json",
    "training_args.bin",
    "all_results.json",
    "eval_results.json",
    "train_results.json",
    "instructions_function_calling.md",
]

PURGE_DIRS = ["onnx", "openvino", "runs", ".cache"]

MODEL_SUBDIRS = ["local-llm", "finbert", "sentence-embedder"]


def human_bytes(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} PB"


def purge_model_dir(model_dir: Path) -> int:
    if not model_dir.exists():
        print(f"Skip {model_dir} (not present)")
        return 0

    freed = 0

    for sub in PURGE_DIRS:
        target = model_dir / sub
        if target.exists() and target.is_dir():
            size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
            shutil.rmtree(target, ignore_errors=True)
            freed += size
            print(f"  removed {target.relative_to(model_dir.parent)} ({human_bytes(size)})")

    for path in model_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(model_dir).as_posix()
        if any(fnmatch.fnmatch(rel, pattern) for pattern in PURGE_PATTERNS):
            size = path.stat().st_size
            try:
                path.unlink()
                freed += size
                print(f"  removed {path.relative_to(model_dir.parent)} ({human_bytes(size)})")
            except OSError as exc:
                print(f"  could not remove {path}: {exc}")

    return freed


def main() -> None:
    cache_dir = Path(config.LOCAL_MODEL_CACHE_DIR)
    if not cache_dir.exists():
        print(f"No model cache at {cache_dir}; nothing to do.")
        return

    total_freed = 0
    for sub in MODEL_SUBDIRS:
        model_dir = cache_dir / sub
        print(f"Cleaning {model_dir} ...")
        total_freed += purge_model_dir(model_dir)

    print(f"\nReclaimed {human_bytes(total_freed)}.")


if __name__ == "__main__":
    main()
