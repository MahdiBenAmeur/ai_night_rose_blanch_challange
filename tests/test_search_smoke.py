from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import REQUIRED_EMBED_DIM, REQUIRED_TOP_K
from search import build_faiss_index, save_faiss_index, save_mapping, search_topk


def _normalized_random(size: int, dim: int = REQUIRED_EMBED_DIM, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    vectors = rng.normal(size=(size, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def test_faiss_backend_returns_three_items(tmp_path: Path) -> None:
    mapping = [
        {"id": 0, "id_document": 1, "texte_fragment": "fragment alpha"},
        {"id": 1, "id_document": 2, "texte_fragment": "fragment beta"},
        {"id": 2, "id_document": 3, "texte_fragment": "fragment gamma"},
        {"id": 3, "id_document": 4, "texte_fragment": "fragment delta"},
    ]
    vectors = _normalized_random(size=len(mapping))
    index = build_faiss_index(vectors)

    index_path = tmp_path / "faiss.index"
    mapping_path = tmp_path / "faiss_mapping.jsonl"
    save_faiss_index(index, index_path=index_path)
    save_mapping(mapping, mapping_path=mapping_path)

    query_vector = vectors[0]
    results = search_topk(
        question="unused in smoke test",
        backend="faiss",
        k=REQUIRED_TOP_K,
        query_vector=query_vector,
        index_path=index_path,
        mapping_path=mapping_path,
    )

    assert len(results) == REQUIRED_TOP_K
    assert all(set(item.keys()) == {"texte_fragment", "score"} for item in results)


@pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not configured")
def test_postgres_functions_are_importable() -> None:
    from db import ensure_schema, query_topk_pg, upsert_embeddings

    assert ensure_schema is not None
    assert upsert_embeddings is not None
    assert query_topk_pg is not None
