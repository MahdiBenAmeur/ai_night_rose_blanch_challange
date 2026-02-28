"""Build-time persistence helpers for local FAISS artifacts and PostgreSQL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import faiss
import numpy as np
try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

from src.config import DATA_DIR, PROJECT_ROOT, REQUIRED_EMBED_DIM, get_settings
try:
    from src.chunking import chunk_text
except ModuleNotFoundError:
    chunk_text = None
try:
    from src.embed import embed_texts
except ModuleNotFoundError:
    embed_texts = None
try:
    from src.parser import parse_pdfs
except ModuleNotFoundError:
    parse_pdfs = None

INIT_SQL_PATH = PROJECT_ROOT / "scripts" / "init_db.sql"
FAISS_INDEX_PATH = PROJECT_ROOT / "data" / "faiss.index"
FAISS_MAPPING_PATH = PROJECT_ROOT / "data" / "faiss_mapping.jsonl"
FAISS_VECTORS_PATH = PROJECT_ROOT / "data" / "faiss_vectors.npy"


def _vector_literal(vector: Sequence[float] | np.ndarray) -> str:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    return "[" + ",".join(f"{float(value):.8f}" for value in array) + "]"


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != REQUIRED_EMBED_DIM:
        raise ValueError(f"vectors must have shape (n, {REQUIRED_EMBED_DIM}).")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """Build a FAISS inner-product index from normalized embeddings."""

    normalized = _normalize_rows(vectors)
    index = faiss.IndexFlatIP(REQUIRED_EMBED_DIM)
    index.add(normalized)
    return index


def save_faiss_index(index: faiss.Index, index_path: Path = FAISS_INDEX_PATH) -> None:
    """Save a FAISS index to disk."""

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def save_mapping(records: list[dict[str, Any]], mapping_path: Path = FAISS_MAPPING_PATH) -> None:
    """Save retrieval records as JSONL alongside the local vector store."""

    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_vectors(vectors: np.ndarray, vectors_path: Path = FAISS_VECTORS_PATH) -> None:
    """Save raw embedding vectors to disk for later PostgreSQL loading."""

    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, np.asarray(vectors, dtype=np.float32))


def load_mapping(mapping_path: Path = FAISS_MAPPING_PATH) -> list[dict[str, Any]]:
    """Load saved JSONL retrieval records from the local vector store."""

    if not mapping_path.exists():
        raise FileNotFoundError(f"FAISS mapping not found: {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_vectors(vectors_path: Path = FAISS_VECTORS_PATH) -> np.ndarray:
    """Load saved embedding vectors from the local vector store."""

    if not vectors_path.exists():
        raise FileNotFoundError(f"Saved vectors not found: {vectors_path}")
    return np.load(vectors_path)


def _get_connection() -> Any:
    if psycopg is None:
        raise ModuleNotFoundError("psycopg is required for PostgreSQL operations.")
    settings = get_settings(backend="postgres")
    return psycopg.connect(settings.database_url)


def _ensure_schema(sql_path: Path = INIT_SQL_PATH) -> None:
    if not sql_path.exists():
        raise FileNotFoundError(f"Schema SQL file not found: {sql_path}")

    sql_text = sql_path.read_text(encoding="utf-8")
    with _get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql_text)
        connection.commit()


def _insert_rows(rows: Iterable[tuple[int, str, Sequence[float] | np.ndarray]]) -> int:
    payload: list[tuple[int, str, str]] = []
    for id_document, texte_fragment, vecteur in rows:
        payload.append((id_document, texte_fragment, _vector_literal(vecteur)))

    if not payload:
        return 0

    sql = """
        INSERT INTO embeddings (id_document, texte_fragment, vecteur)
        VALUES (%s, %s, %s::vector)
    """
    with _get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.executemany(sql, payload)
        connection.commit()
    return len(payload)


def build_vs(input_dir: str | Path = DATA_DIR) -> bool:
    """Build the local vector store if it does not already exist.

    The build pipeline is:
    parse PDFs -> generate chunks -> embed questions -> save FAISS artifacts.

    Args:
        input_dir: Directory containing the source PDF documents.

    Returns:
        ``True`` if a new vector store was built, otherwise ``False``.
    """

    if parse_pdfs is None or chunk_text is None or embed_texts is None:
        raise ModuleNotFoundError("Build dependencies are missing. Install parser, chunking, and embedding dependencies.")
    if FAISS_INDEX_PATH.exists() and FAISS_MAPPING_PATH.exists() and FAISS_VECTORS_PATH.exists():
        print("VS exists.")
        return False

    documents = parse_pdfs(input_dir)
    chunks = chunk_text(documents)
    vectors, records = embed_texts(chunks)

    index = build_faiss_index(vectors)
    save_faiss_index(index)
    save_mapping(records)
    save_vectors(vectors)

    print(f"VS built with {len(records)} chunks.")
    return True


def push_vs_to_postgres() -> int:
    """Push the locally built vector store into PostgreSQL.

    Returns:
        The number of inserted rows.
    """

    if not FAISS_MAPPING_PATH.exists() or not FAISS_VECTORS_PATH.exists():
        raise FileNotFoundError("Local VS not found. Build it first.")

    mapping = load_mapping(FAISS_MAPPING_PATH)
    vectors = load_vectors(FAISS_VECTORS_PATH)
    rows = [
        (int(record["id_document"]), str(record["texte_fragment"]), vectors[idx])
        for idx, record in enumerate(mapping)
    ]

    _ensure_schema()
    inserted = _insert_rows(rows)
    print(f"Inserted {inserted} rows into PostgreSQL.")
    return inserted
