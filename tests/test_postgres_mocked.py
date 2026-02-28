from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import db, search


class FakeCursor:
    def __init__(self, fetch_rows: list[tuple[str, float]] | None = None) -> None:
        self.fetch_rows = fetch_rows or []
        self.executed: list[tuple[str, tuple | None]] = []
        self.executemany_calls: list[tuple[str, list[tuple[str, str, str]]]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, sql: str, params: tuple | None = None) -> None:
        self.executed.append((sql, params))

    def executemany(self, sql: str, seq_of_params: list[tuple[str, str, str]]) -> None:
        self.executemany_calls.append((sql, list(seq_of_params)))

    def fetchall(self) -> list[tuple[str, float]]:
        return self.fetch_rows


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor
        self.commit_count = 0

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.commit_count += 1


class FakePsycopg:
    def __init__(self, connections: list[FakeConnection]) -> None:
        self.connections = connections
        self.calls: list[str] = []

    def connect(self, dsn: str) -> FakeConnection:
        self.calls.append(dsn)
        if not self.connections:
            raise AssertionError("No fake connections left.")
        return self.connections.pop(0)


def test_push_vs_to_postgres_uses_expected_schema_and_insert(monkeypatch, tmp_path: Path) -> None:
    mapping_path = tmp_path / "faiss_mapping.jsonl"
    vectors_path = tmp_path / "faiss_vectors.npy"
    sql_path = tmp_path / "init_db.sql"

    mapping = [
        {"id_document": 10, "texte_fragment": "fragment one"},
        {"id_document": 11, "texte_fragment": "fragment two"},
    ]
    db.save_mapping(mapping, mapping_path=mapping_path)
    db.save_vectors(
        np.asarray(
            [
                np.full(384, 0.1, dtype=np.float32),
                np.full(384, 0.2, dtype=np.float32),
            ]
        ),
        vectors_path=vectors_path,
    )
    sql_path.write_text(
        "CREATE EXTENSION IF NOT EXISTS vector;\nCREATE TABLE IF NOT EXISTS embeddings (id BIGSERIAL PRIMARY KEY, id_document INTEGER NOT NULL, texte_fragment TEXT NOT NULL, vecteur VECTOR(384) NOT NULL);\n",
        encoding="utf-8",
    )

    insert_cursor = FakeCursor()
    fake_psycopg = FakePsycopg([FakeConnection(insert_cursor)])
    schema_calls: list[Path] = []

    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/testdb")
    monkeypatch.setattr(db, "FAISS_MAPPING_PATH", mapping_path)
    monkeypatch.setattr(db, "FAISS_VECTORS_PATH", vectors_path)
    monkeypatch.setattr(db, "psycopg", fake_psycopg)
    monkeypatch.setattr(db, "_ensure_schema", lambda: schema_calls.append(sql_path))

    inserted = db.push_vs_to_postgres()

    assert inserted == 2
    assert schema_calls == [sql_path]
    assert fake_psycopg.calls == ["postgresql://user:pass@localhost:5432/testdb"]
    assert insert_cursor.executemany_calls
    insert_sql, payload = insert_cursor.executemany_calls[0]
    assert "INSERT INTO embeddings (id_document, texte_fragment, vecteur)" in insert_sql
    assert payload[0][0] == 10
    assert payload[0][1] == "fragment one"
    assert payload[0][2].startswith("[")


def test_search_topk_postgres_returns_expected_shape_without_real_db(monkeypatch) -> None:
    fetch_rows = [
        ("fragment alpha", 0.91),
        ("fragment beta", 0.87),
        ("fragment gamma", 0.83),
    ]
    search_cursor = FakeCursor(fetch_rows=fetch_rows)
    fake_psycopg = FakePsycopg([FakeConnection(search_cursor)])

    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/testdb")
    monkeypatch.setattr(search, "psycopg", fake_psycopg)
    monkeypatch.setattr(search, "embed_query", lambda _: np.ones(384, dtype=np.float32))

    results = search.search_topk("test postgres query", backend="postgres")

    assert len(results) == 3
    assert results[0] == {"texte_fragment": "fragment alpha", "score": 0.91}
    assert all(set(item.keys()) == {"texte_fragment", "score"} for item in results)
    assert search_cursor.executed
    sql, params = search_cursor.executed[0]
    assert "FROM embeddings" in sql
    assert "1 - (vecteur <=> %s::vector) AS score" in sql
    assert params is not None
    assert params[2] == 3
