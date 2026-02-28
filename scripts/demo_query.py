from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


from src.config import REQUIRED_EMBED_DIM, REQUIRED_TOP_K
from src.search import search_topk


def _mock_query_vector(question: str, dim: int = REQUIRED_EMBED_DIM) -> np.ndarray:
    seed = sum(ord(char) for char in question) % (2**32)
    rng = np.random.default_rng(seed=seed)
    vector = rng.normal(size=dim).astype(np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector
    return vector / norm


def _print_results( results: list[dict[str, float | str]]) -> None:
    for index ,item in enumerate(results, start=1):
        print(f"Résultat {index}:")
        print(f'Texte :"{item['texte_fragment']}"')
        print(f"Score :{item['score']}")


def main() -> None:
    question = "Quels fragments parlent de recherche semantique ou de similarite vectorielle ?"
    query_vector = _mock_query_vector(question)

    faiss_results = search_topk(question=question, backend="faiss", k=REQUIRED_TOP_K, query_vector=query_vector)
    _print_results( faiss_results)

    if os.getenv("DATABASE_URL"):
        postgres_results = search_topk(
            question=question,
            backend="postgres",
            k=REQUIRED_TOP_K,
            query_vector=query_vector,
        )
        _print_results("POSTGRES", postgres_results)
    else:
        print("POSTGRES")
        print("DATABASE_URL not set; skipping postgres demo.")


if __name__ == "__main__":
    main()
