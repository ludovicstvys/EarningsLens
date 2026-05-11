from __future__ import annotations

from pathlib import Path
import sys

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from earningslens import config


MODEL_TARGETS = [
    ("local-llm", config.LOCAL_LLM_MODEL_ID),
    ("finbert", config.FINBERT_MODEL_ID),
    ("sentence-embedder", config.SENTENCE_EMBEDDER_MODEL_ID),
]


def main() -> None:
    cache_dir = Path(config.LOCAL_MODEL_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for subdir, repo_id in MODEL_TARGETS:
        destination = cache_dir / subdir
        print(f"Downloading {repo_id} -> {destination}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(destination),
            local_dir_use_symlinks=False,
            resume_download=True,
        )


if __name__ == "__main__":
    main()
