# Semantic Search Prototype

This scaffold provides a minimal RAG-oriented semantic search prototype with two backends:

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
project/
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

- `src/parser.py`, `src/chunking.py`, and `src/embed.py` are placeholders by design.
- The model is fixed to `all-MiniLM-L6-v2`, but the placeholder embedding functions intentionally do not download or execute it yet.
- The runnable scripts fall back to deterministic mock fragments and normalized mock vectors until real parsing/chunking/embedding is added.

## Setup

```bash
pip install -r requirements.txt
```

Optionally create a `.env` file from `.env.example`.

## PostgreSQL / pgvector

Run the schema script:

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

Set `DATABASE_URL`, initialize the schema, then:

```bash
python scripts/load_to_postgres.py
python scripts/demo_query.py
```

## Testing

```bash
pytest -q
```
