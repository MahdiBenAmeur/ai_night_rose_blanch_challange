from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np

from config import PROJECT_ROOT, REQUIRED_EMBED_DIM, REQUIRED_TOP_K, get_settings
from embed import embed_query

FAISS_INDEX_PATH = PROJECT_ROOT / "data" / "faiss.index"
FAISS_MAPPING_PATH = PROJECT_ROOT / "data" / "faiss_mapping.jsonl"
FAISS_VECTORS_PATH = PROJECT_ROOT / "data" / "faiss_vectors.npy"


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != REQUIRED_EMBED_DIM:
        raise ValueError(f"vectors must have shape (n, {REQUIRED_EMBED_DIM}).")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


def _normalize_query(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    if array.shape[1] != REQUIRED_EMBED_DIM:
        raise ValueError(f"query vector must have dimension {REQUIRED_EMBED_DIM}.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    normalized = _normalize_rows(vectors)
    index = faiss.IndexFlatIP(REQUIRED_EMBED_DIM)
    index.add(normalized)
    return index


def save_faiss_index(index: faiss.Index, index_path: Path = FAISS_INDEX_PATH) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def load_faiss_index(index_path: Path = FAISS_INDEX_PATH) -> faiss.Index:
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    return faiss.read_index(str(index_path))


def save_mapping(records: list[dict[str, Any]], mapping_path: Path = FAISS_MAPPING_PATH) -> None:
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_mapping(mapping_path: Path = FAISS_MAPPING_PATH) -> list[dict[str, Any]]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"FAISS mapping not found: {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_vectors(vectors: np.ndarray, vectors_path: Path = FAISS_VECTORS_PATH) -> None:
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, np.asarray(vectors, dtype=np.float32))


def load_vectors(vectors_path: Path = FAISS_VECTORS_PATH) -> np.ndarray:
    if not vectors_path.exists():
        raise FileNotFoundError(f"Saved vectors not found: {vectors_path}")
    return np.load(vectors_path)


def query_faiss(
    index: faiss.Index,
    query_vec: Sequence[float] | np.ndarray,
    mapping: list[dict[str, Any]],
    k: int = REQUIRED_TOP_K,
) -> list[dict[str, Any]]:
    if k != REQUIRED_TOP_K:
        raise ValueError(f"k is fixed to {REQUIRED_TOP_K}.")

    normalized_query = _normalize_query(query_vec)
    scores, indices = index.search(normalized_query, k)

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


def search_topk(question: str, backend: str, k: int = REQUIRED_TOP_K, **kwargs: Any) -> list[dict[str, Any]]:
    settings = get_settings(backend=backend if backend == "postgres" else None)
    if k != settings.top_k:
        raise ValueError(f"k is fixed to {settings.top_k}.")

    query_vector = kwargs.get("query_vector")
    if query_vector is None:
        query_vector = embed_query(question)

    if backend == "postgres":
        from db import query_topk_pg

        return query_topk_pg(query_vector=query_vector, k=k)

    if backend == "faiss":
        index_path = Path(kwargs.get("index_path", FAISS_INDEX_PATH))
        mapping_path = Path(kwargs.get("mapping_path", FAISS_MAPPING_PATH))
        index = load_faiss_index(index_path=index_path)
        mapping = load_mapping(mapping_path=mapping_path)
        return query_faiss(index=index, query_vec=query_vector, mapping=mapping, k=k)

    raise ValueError("backend must be either 'faiss' or 'postgres'.")
