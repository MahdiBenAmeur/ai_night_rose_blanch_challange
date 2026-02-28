from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import psycopg

from config import PROJECT_ROOT, REQUIRED_TOP_K, get_settings

INIT_SQL_PATH = PROJECT_ROOT / "scripts" / "init_db.sql"


def _vector_literal(vector: Sequence[float] | np.ndarray) -> str:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    return "[" + ",".join(f"{float(value):.8f}" for value in array) + "]"


def get_connection() -> psycopg.Connection:
    settings = get_settings(backend="postgres")
    return psycopg.connect(settings.database_url)


def ensure_schema(sql_path: Path = INIT_SQL_PATH) -> None:
    if not sql_path.exists():
        raise FileNotFoundError(f"Schema SQL file not found: {sql_path}")

    sql_text = sql_path.read_text(encoding="utf-8")
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql_text)
        connection.commit()


def upsert_embeddings(rows: Iterable[tuple[int, str, Sequence[float] | np.ndarray]]) -> int:
    """Insert embedding rows into PostgreSQL.

    The schema only guarantees a synthetic primary key on ``id``.
    Because no natural unique constraint was imposed for fragments, this
    function behaves as an append insert API while keeping the requested name.
    """

    payload: list[tuple[int, str, str]] = []
    for id_document, texte_fragment, vecteur in rows:
        payload.append((id_document, texte_fragment, _vector_literal(vecteur)))

    if not payload:
        return 0

    sql = """
        INSERT INTO embeddings (id_document, texte_fragment, vecteur)
        VALUES (%s, %s, %s::vector)
    """
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.executemany(sql, payload)
        connection.commit()
    return len(payload)


def query_topk_pg(query_vector: Sequence[float] | np.ndarray, k: int = REQUIRED_TOP_K) -> list[dict[str, Any]]:
    if k != REQUIRED_TOP_K:
        raise ValueError(f"k is fixed to {REQUIRED_TOP_K}.")

    vector_literal = _vector_literal(query_vector)
    sql = """
        SELECT texte_fragment, 1 - (vecteur <=> %s::vector) AS score
        FROM embeddings
        ORDER BY vecteur <=> %s::vector
        LIMIT %s
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, (vector_literal, vector_literal, k))
            rows = cursor.fetchall()

    return [{"texte_fragment": str(text), "score": float(score)} for text, score in rows]
