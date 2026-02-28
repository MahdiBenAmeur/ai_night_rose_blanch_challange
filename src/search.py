"""Query-time retrieval helpers for local FAISS and PostgreSQL search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np
try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

from src.config import PROJECT_ROOT, REQUIRED_EMBED_DIM, REQUIRED_TOP_K, get_settings
from src.embed import embed_query

FAISS_INDEX_PATH = PROJECT_ROOT / "data" / "faiss.index"
FAISS_MAPPING_PATH = PROJECT_ROOT / "data" / "faiss_mapping.jsonl"
FAISS_VECTORS_PATH = PROJECT_ROOT / "data" / "faiss_vectors.npy"


def _normalize_query(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    if array.shape[1] != REQUIRED_EMBED_DIM:
        raise ValueError(f"query vector must have dimension {REQUIRED_EMBED_DIM}.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


def load_faiss_index(index_path: Path = FAISS_INDEX_PATH) -> faiss.Index:
    """Load the local FAISS index used for similarity search."""

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    return faiss.read_index(str(index_path))


def load_mapping(mapping_path: Path = FAISS_MAPPING_PATH) -> list[dict[str, Any]]:
    """Load the text mapping associated with the local FAISS index."""

    if not mapping_path.exists():
        raise FileNotFoundError(f"FAISS mapping not found: {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _vector_literal(vector: Sequence[float] | np.ndarray) -> str:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    return "[" + ",".join(f"{float(value):.8f}" for value in array) + "]"


def _query_postgres(question: str, k: int ) -> list[dict[str, Any]]:
    if psycopg is None:
        raise ModuleNotFoundError("psycopg is required for PostgreSQL search.")
    settings = get_settings(backend="postgres")
    query_vector = embed_query(question)
    vector_literal = _vector_literal(query_vector)
    sql = """
        SELECT texte_fragment, 1 - (vecteur <=> %s::vector) AS score
        FROM embeddings
        ORDER BY vecteur <=> %s::vector
        LIMIT %s
    """

    with psycopg.connect(settings.database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, (vector_literal, vector_literal, k))
            rows = cursor.fetchall()

    return [{"texte_fragment": str(text), "score": float(score)} for text, score in rows]


def _search_local(question: str, k: int) -> list[dict[str, Any]]:
    if not FAISS_INDEX_PATH.exists() or not FAISS_MAPPING_PATH.exists():
        raise FileNotFoundError("Local VS not found. Build it first.")

    query_vector = embed_query(question)
    index = load_faiss_index()
    mapping = load_mapping()
    scores, indices = index.search(_normalize_query(query_vector), k)

    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0], strict=False):
        if idx < 0 or idx >= len(mapping):
            continue
        results.append(
            {
                "texte_fragment": str(mapping[idx]["texte_fragment"]),
                "score": float(score),
            }
        )
    return results


def search_topk(question: str, backend: str) -> list[dict[str, Any]]:
    """Search the selected backend and return the fixed top-k results.

    Args:
        question: User query to search for.
        backend: Either ``local`` for FAISS or ``postgres`` for pgvector.

    Returns:
        A list of search results containing only ``texte_fragment`` and
        ``score``.
    """

    settings = get_settings(backend=backend if backend == "postgres" else None)
    k = settings.top_k

    if backend == "local":
        results =  _search_local(question, k)

    elif backend == "postgres":
        results = _query_postgres(question, k)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    for index , res in enumerate(results , start=1):
        print(f"Résultat {index}")
        print(f"Texte : {res['texte_fragment']}")
        print(f"Score : {res['score']}")
    return results
