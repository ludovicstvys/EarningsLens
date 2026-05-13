from __future__ import annotations

from pathlib import Path
import shutil
import sys

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from earningslens import config


COMMON_IGNORE = [
    "onnx/*",
    "openvino/*",
    "runs/*",
    "*.h5",
    "*.msgpack",
    "*.ot",
    "*flax*",
    "*tf_model*",
    "rust_model*",
    "train_script.py",
    "data_config.json",
    "trainer_state.json",
    "training_args.bin",
    "all_results.json",
    "eval_results.json",
    "train_results.json",
]

MODEL_TARGETS = [
    ("local-llm", config.LOCAL_LLM_MODEL_ID, COMMON_IGNORE),
    ("finbert", config.FINBERT_MODEL_ID, COMMON_IGNORE),
    ("sentence-embedder", config.SENTENCE_EMBEDDER_MODEL_ID, COMMON_IGNORE),
]


def main() -> None:
    cache_dir = Path(config.LOCAL_MODEL_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for subdir, repo_id, ignore in MODEL_TARGETS:
        destination = cache_dir / subdir
        print(f"Downloading {repo_id} -> {destination}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(destination),
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=ignore,
        )
        staging = destination / ".cache"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
