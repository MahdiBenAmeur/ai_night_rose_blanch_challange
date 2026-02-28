from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
DEFAULT_CACHE_DIR = Path("models_cache")


def embed_texts(texts: list[str]) -> np.ndarray:
    """Placeholder batch embedding function.

    TODO:
    - Load the imposed sentence-transformers model from ``MODEL_NAME``.
    - Use ``DEFAULT_CACHE_DIR`` or configured cache directory.
    - Return a float32 matrix shaped ``(len(texts), 384)``.

    The real model load is intentionally deferred in this scaffold.
    """

    raise NotImplementedError("embed_texts is a placeholder and must be implemented later.")


def embed_query(query: str) -> np.ndarray:
    """Placeholder query embedding function.

    TODO:
    - Encode a single query with the imposed model.
    - Return a float32 vector shaped ``(384,)``.

    The real model load is intentionally deferred in this scaffold.
    """

    raise NotImplementedError("embed_query is a placeholder and must be implemented later.")
