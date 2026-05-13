from __future__ import annotations

import gc
import logging

LOGGER = logging.getLogger(__name__)


def clear_model_memory() -> None:
    gc.collect()
    try:
        import torch
    except Exception as exc:  # pragma: no cover - only relevant when torch import is broken.
        LOGGER.debug("Skipping accelerator cache cleanup because torch is unavailable: %s", exc)
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available() and hasattr(torch, "mps"):
        torch.mps.empty_cache()
