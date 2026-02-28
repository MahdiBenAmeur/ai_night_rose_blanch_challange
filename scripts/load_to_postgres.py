from __future__ import annotations

import sys
from pathlib import Path

import numpy as np



from src.db import ensure_schema, upsert_embeddings
from src.search import FAISS_MAPPING_PATH, FAISS_VECTORS_PATH, load_mapping, load_vectors


def _mock_fragments() -> list[dict[str, object]]:
    texts = [
        "Les enzymes soutiennent la structure de la pate pendant la fermentation.",
        "Le pgvector permet une recherche de similarite basee sur le cosine.",
        "FAISS fournit une recherche locale rapide sur des vecteurs normalises.",
    ]
    return [
        {"id": idx, "id_document": idx + 1, "texte_fragment": text}
        for idx, text in enumerate(texts)
    ]


def _mock_vectors(size: int, dim: int = 384) -> np.ndarray:
    rng = np.random.default_rng(seed=11)
    vectors = rng.normal(size=(size, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def main() -> None:
    try:
        mapping = load_mapping(FAISS_MAPPING_PATH)
        vectors = load_vectors(FAISS_VECTORS_PATH)
    except FileNotFoundError:
        mapping = _mock_fragments()
        vectors = _mock_vectors(size=len(mapping))

    rows = [
        (int(record["id_document"]), str(record["texte_fragment"]), vectors[idx])
        for idx, record in enumerate(mapping)
    ]

    ensure_schema()
    inserted = upsert_embeddings(rows)
    print(f"Inserted {inserted} rows into PostgreSQL.")


if __name__ == "__main__":
    main()
