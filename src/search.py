"""Query-time retrieval helpers for local FAISS and PostgreSQL search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np
from pydantic import BaseModel, Field
try:
    from mistralai import Mistral
except ModuleNotFoundError:
    Mistral = None
try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

from src.config import PROJECT_ROOT, REQUIRED_EMBED_DIM, REQUIRED_TOP_K, get_settings
from src.embed import embed_query

FAISS_INDEX_PATH = PROJECT_ROOT / "data" / "faiss.index"
FAISS_MAPPING_PATH = PROJECT_ROOT / "data" / "faiss_mapping.jsonl"
FAISS_VECTORS_PATH = PROJECT_ROOT / "data" / "faiss_vectors.npy"
MAX_AGENT_SEARCH_ATTEMPTS = 3


class SearchIterationDecision(BaseModel):
    """Structured decision for a single search iteration."""

    satisfied: bool = Field(..., description="Whether the current search results satisfy the user query.")
    reformulated_query: str = Field(..., description="A rewritten query to improve retrieval when needed.")


class CollectionSelectionDecision(BaseModel):
    """Structured decision used to choose the best attempt collection."""

    selected_collection_number: int = Field(..., description="One-based collection number: 1, 2, or 3.")


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


def _query_postgres(question: str, k: int) -> list[dict[str, Any]]:
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


def _run_search_backend(question: str, backend: str, k: int) -> list[dict[str, Any]]:
    if backend == "local":
        return _search_local(question, k)
    if backend == "postgres":
        return _query_postgres(question, k)
    raise ValueError(f"Unknown backend: {backend}")


def _print_results(results: list[dict[str, Any]]) -> None:
    for index, res in enumerate(results, start=1):
        print(f"Resultat {index}")
        print(f"Texte : {res['texte_fragment']}")
        print(f"Score : {res['score']}")


def _build_iteration_prompt(question: str, results: list[dict[str, Any]]) -> str:
    serialized_results = [
        {
            "result_number": index + 1,
            "texte_fragment": result["texte_fragment"],
            "score": result["score"],
        }
        for index, result in enumerate(results)
    ]
    return (
        "Evaluate these semantic search results for the user question.\n"
        "If the results are satisfactory, mark satisfied as true.\n"
        "If they are not satisfactory, provide a better reformulated query.\n"
        "Do not rewrite, quote, or reproduce any chunk text in your decision.\n\n"
        f"Question:\n{question}\n\n"
        f"Results:\n{json.dumps(serialized_results, ensure_ascii=False, indent=2)}"
    )


def _build_collection_selection_prompt(
    question: str,
    collections: list[list[dict[str, Any]]],
) -> str:
    serialized_collections = []
    for index, collection in enumerate(collections, start=1):
        serialized_collections.append(
            {
                "collection_number": index,
                "results": collection,
            }
        )
    return (
        "Choose the best semantic-search collection for the user question.\n"
        "Each collection is one full search attempt.\n"
        "Prefer the collection where the most logical and relevant chunk is ranked first.\n"
        "Do not rewrite or reproduce chunk text in your decision.\n"
        "Return only the best collection number.\n\n"
        f"Question:\n{question}\n\n"
        f"Collections:\n{json.dumps(serialized_collections, ensure_ascii=False, indent=2)}"
    )


def _agent_decide_iteration(
    question: str,
    results: list[dict[str, Any]],
) -> SearchIterationDecision:
    if Mistral is None:
        raise ModuleNotFoundError("mistralai is required for agent-assisted search.")

    settings = get_settings(require_mistral=True)
    client = Mistral(api_key=settings.mistral_api_key)
    response = client.chat.parse(
        model=settings.mistral_model,
        temperature=0.0,
        response_format=SearchIterationDecision,
        messages=[
            {
                "role": "system",
                "content": (
                    "You evaluate retrieval results. "
                    "If the results are insufficient, rewrite the query to improve retrieval. "
                    "Never reproduce chunk text in your answer."
                ),
            },
            {
                "role": "user",
                "content": _build_iteration_prompt(question, results),
            },
        ],
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("The search agent returned no structured decision.")
    return parsed


def _agent_select_collection(
    question: str,
    collections: list[list[dict[str, Any]]],
) -> CollectionSelectionDecision:
    if Mistral is None:
        raise ModuleNotFoundError("mistralai is required for agent-assisted search.")

    settings = get_settings(require_mistral=True)
    client = Mistral(api_key=settings.mistral_api_key)
    response = client.chat.parse(
        model=settings.mistral_model,
        temperature=0.0,
        response_format=CollectionSelectionDecision,
        messages=[
            {
                "role": "system",
                "content": (
                    "You choose the best retrieval collection among multiple search attempts. "
                    "Prefer the collection where the strongest and most logical chunk is ranked first. "
                    "Only return the best collection number."
                ),
            },
            {
                "role": "user",
                "content": _build_collection_selection_prompt(question, collections),
            },
        ],
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("The search agent returned no collection-selection decision.")
    return parsed


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
    results = _run_search_backend(question, backend, settings.top_k)
    _print_results(results)
    return results


def search_topk_with_agent(question: str, backend: str) -> list[dict[str, Any]]:
    """Search with an LLM evaluator that can reformulate the query up to three times.

    The function first runs a normal search. The retrieved results are then
    evaluated by the Mistral model. If the model is not satisfied, it rewrites
    the query and the search is repeated. If no satisfactory result set is
    found, the function returns the full result set from the last evaluated
    search iteration.

    The returned items keep the exact same shape as ``search_topk``.
    """

    settings = get_settings(backend=backend if backend == "postgres" else None, require_mistral=True)
    current_query = question
    attempted_collections: list[list[dict[str, Any]]] = []

    for _ in range(MAX_AGENT_SEARCH_ATTEMPTS):
        results = _run_search_backend(current_query, backend, settings.top_k)
        if not results:
            break

        attempted_collections.append(results)
        decision = _agent_decide_iteration(current_query, results)

        if decision.satisfied:
            _print_results(results)
            return results

        next_query = decision.reformulated_query.strip()
        if not next_query or next_query == current_query:
            break
        current_query = next_query

    if not attempted_collections:
        _print_results([])
        return []

    selection = _agent_select_collection(question, attempted_collections)
    collection_index = max(1, min(selection.selected_collection_number, len(attempted_collections))) - 1
    final_results = attempted_collections[collection_index]
    _print_results(final_results)
    return final_results
