"""Embedding helpers for chunk questions and user search queries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import REQUIRED_EMBED_DIM, get_settings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
DEFAULT_CACHE_DIR = Path("models_cache")

_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        settings = get_settings()
        _MODEL = SentenceTransformer(
            MODEL_NAME,
            cache_folder=str(settings.model_cache_dir),
        )
    return _MODEL


def _encode(texts: list[str]) -> np.ndarray:
    model = _get_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[1] != REQUIRED_EMBED_DIM:
        raise ValueError(f"Expected embedding dimension {REQUIRED_EMBED_DIM}, got {array.shape[1]}.")
    return array


def embed_texts(chunks: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Embed chunk questions and prepare retrieval records from chunk answers.

    Args:
        chunks: Chunk dictionaries containing ``question``, ``answer``, and
            ``metadata``.

    Returns:
        A tuple containing:
        - the normalized embedding matrix for all questions
        - the retrieval records whose ``texte_fragment`` values come from the
          chunk answers
    """

    if not chunks:
        return np.empty((0, REQUIRED_EMBED_DIM), dtype=np.float32), []

    questions: list[str] = []
    records: list[dict[str, Any]] = []

    for index, chunk in enumerate(chunks):
        question = str(chunk["question"]).strip()
        answer = str(chunk["answer"]).strip()
        metadata = dict(chunk.get("metadata", {}))

        if not question:
            continue
            #raise ValueError(f"Chunk at index {index} has an empty question.")
        if not answer:
            continue
            #raise ValueError(f"Chunk at index {index} has an empty answer.")

        questions.append(question)
        records.append(
            {
                "id": index,
                "id_document": int(metadata.get("id_document", index + 1)),
                "texte_fragment": answer,
                "metadata": metadata,
            }
        )

    return _encode(questions), records


def embed_query(query: str) -> np.ndarray:
    """Embed a user query with the imposed sentence-transformers model.

    Args:
        query: User question to encode.

    Returns:
        A normalized embedding vector of dimension 384.
    """

    query_text = query.strip()
    if not query_text:
        raise ValueError("query must not be empty.")
    return _encode([query_text])[0]
