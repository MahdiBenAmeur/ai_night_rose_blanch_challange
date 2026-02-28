# Semantic Search Prototype

This project provides a minimal RAG-oriented semantic search prototype with two backends:

- Local vector search with FAISS
- PostgreSQL with pgvector

The implementation enforces these fixed constraints in code:

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: `384`
- Similarity method: cosine similarity
- Top K returned: `3`
- Result shape: `texte_fragment` and similarity `score` only

## Layout

```text
  README.md
  requirements.txt
  .env.example
  models_cache/
  data/
    raw_pdfs/
  src/
    config.py
    parser.py
    chunking.py
    embed.py
    db.py
    search.py
  scripts/
    init_db.sql
    build_local.py
    load_to_postgres.py
    demo_query.py
  tests/
    test_search_smoke.py
```

## Notes

- The embedding model is fixed to `all-MiniLM-L6-v2`.
- The local build pipeline is: parse PDFs -> generate QA chunks -> embed chunk questions -> save FAISS artifacts.
- The PostgreSQL load pipeline reads the local artifacts and inserts them into the `embeddings` table.

## Setup

```bash
pip install -r requirements.txt
```

Optionally create a `.env` file from `.env.example`.

## PostgreSQL / pgvector

Schema initialization is defined in `scripts/init_db.sql` and is executed automatically by `src.db.push_vs_to_postgres()`.

You can still run the SQL manually:

```bash
psql "$DATABASE_URL" -f scripts/init_db.sql
```

The SQL uses an `ivfflat` index with `vector_cosine_ops` for cosine search on `embeddings.vecteur`.

## Local FAISS demo

```bash
python scripts/build_local.py
python scripts/demo_query.py
```

## PostgreSQL demo

Set `DATABASE_URL`, then:

```bash
python scripts/load_to_postgres.py
python scripts/demo_query.py
```

## Testing

```bash
pytest -q
```
