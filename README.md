# Semantic Search Prototype

This project is a small semantic search / RAG prototype built around a fixed embedding setup and two retrieval backends:

- local vector search with FAISS
- PostgreSQL with pgvector

The pipeline is:

1. parse PDF files
2. generate question / answer chunks from the parsed text
3. embed the chunk questions
4. save a local vector store
5. optionally push that vector store into PostgreSQL
6. search either the local store or PostgreSQL

## Fixed Constraints

These constraints are enforced in code:

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: `384`
- Similarity method: cosine similarity
- Top K returned: `3`
- Search result shape:
  - `texte_fragment`
  - `score`

## PostgreSQL Schema

The PostgreSQL side is designed to match this exact structure:

- Table: `embeddings`
- Columns:
  - `id` primary key
  - `id_document` integer
  - `texte_fragment` text
  - `vecteur` `VECTOR(384)`

Defined in [scripts/init_db.sql](/c:/disque%20d/ai_stuff/projects/challanges/ai_night_rag/scripts/init_db.sql).

Actual SQL:

```sql
CREATE TABLE IF NOT EXISTS embeddings (
    id BIGSERIAL PRIMARY KEY,
    id_document INTEGER NOT NULL,
    texte_fragment TEXT NOT NULL,
    vecteur VECTOR(384) NOT NULL
);
```

PostgreSQL search uses cosine distance through pgvector and converts it into a similarity-like score with:

```sql
1 - (vecteur <=> query_vector)
```

The index uses:

```sql
USING ivfflat (vecteur vector_cosine_ops)
```

## Project Structure

```text
README.md
main.py
requirements.txt
.env.example
data/
  raw_pdfs/
models_cache/
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

## What Each Module Does

### `src/config.py`

Loads environment variables and validates runtime settings.

Important settings:

- `DATABASE_URL`
- `MODEL_CACHE_DIR`
- `TOP_K`
- `EMBED_DIM`
- `MISTRAL_API_KEY`
- `MISTRAL_MODEL`

### `src/parser.py`

Parses PDF files with PyMuPDF.

What it does:

- walks through `data/raw_pdfs`
- extracts page text
- keeps rough structure using Markdown-like sections
- converts detected tables into Markdown tables
- returns one item per document with:
  - `text`
  - `metadata`
  - `id_document`
  - `source_path`

### `src/chunking.py`

Generates chunks from parsed documents using Mistral structured output.

Input:

- list of parsed documents

Output:

- list of chunks shaped like:

```python
{
    "question": "...",
    "answer": "...",
    "metadata": {...}
}
```

Behavior:

- sends one full document at a time to Mistral
- asks for many short, grounded question / answer pairs
- retries up to 3 times
- waits 2 seconds between retries
- raises if `MISTRAL_API_KEY` or `MISTRAL_MODEL` is missing

### `src/embed.py`

Embeds:

- chunk `question` fields for indexing
- user queries for retrieval

Important detail:

- the embedded text is the chunk `question`
- the stored retrieval text is the chunk `answer`

So the vector store retrieves short answer fragments, not the embedded questions.

### `src/db.py`

Contains build-time and persistence logic only.

Main public functions:

- `build_vs(...)`
- `push_vs_to_postgres()`

What it does:

- builds the local FAISS vector store if it does not already exist
- saves:
  - FAISS index
  - JSONL mapping
  - raw vectors
- pushes the local vector store into PostgreSQL

It does not perform search queries.

### `src/search.py`

Contains query-time logic only.

Main public function:

- `search_topk(question, backend)`

Supported backends:

- `local`
- `postgres`

Behavior:

- local:
  - loads the saved FAISS index and mapping
  - embeds the query
  - searches with `IndexFlatIP` on normalized vectors
  - returns top 3
- postgres:
  - embeds the query
  - searches the `embeddings` table with pgvector cosine distance
  - returns top 3

## Environment Variables

Copy [.env.example](/c:/disque%20d/ai_stuff/projects/challanges/ai_night_rag/.env.example) into a local `.env` file if needed.

Example:

```env
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
TOP_K=3
MODEL_CACHE_DIR=models_cache
MISTRAL_API_KEY=your_mistral_api_key
MISTRAL_MODEL=mistral-small-latest
```

### `DATABASE_URL`

This is the full PostgreSQL connection string, including:

- username
- password
- host
- port
- database name

Example:

```env
DATABASE_URL=postgresql://postgres:secret123@localhost:5432/rag_db
```

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Current dependencies are:

- `faiss-cpu`
- `mistralai`
- `numpy`
- `pydantic`
- `psycopg[binary]`
- `python-dotenv`
- `pgvector`
- `sentence-transformers`
- `pymupdf`
- `pytest`

## Main Usage

The preferred user-facing entry file is [main.py](/c:/disque%20d/ai_stuff/projects/challanges/ai_night_rag/main.py).

It contains four clearly separated blocks. Uncomment one block at a time:

1. build local vector store
2. search local vector store
3. push local vector store to PostgreSQL
4. search PostgreSQL

This is the simplest way to use the project manually.

## Programmatic Usage

### 1. Build the local vector store

```python
from src.config import DATA_DIR
from src.db import build_vs

build_vs(DATA_DIR)
```

What this does:

- parses PDFs from `data/raw_pdfs`
- chunks them with Mistral
- embeds the chunk questions
- saves local FAISS artifacts

If the vector store already exists, it prints:

```text
VS exists.
```

### 2. Search the local vector store

```python
from src.search import search_topk

results = search_topk(
    question="What is the package of BVZyme GOX 110?",
    backend="local",
)

print(results)
```

If the local vector store does not exist yet, search raises an error.

### 3. Push the local vector store to PostgreSQL

```python
from src.db import push_vs_to_postgres

push_vs_to_postgres()
```

What this does:

- ensures the schema exists by executing `scripts/init_db.sql`
- loads the local mapping and vectors
- inserts rows into the `embeddings` table

### 4. Search PostgreSQL

```python
from src.search import search_topk

results = search_topk(
    question="What is the package of BVZyme GOX 110?",
    backend="postgres",
)

print(results)
```

## Script Wrappers

The scripts are thin wrappers around the `src` functions:

- [build_local.py](/c:/disque%20d/ai_stuff/projects/challanges/ai_night_rag/scripts/build_local.py)
- [load_to_postgres.py](/c:/disque%20d/ai_stuff/projects/challanges/ai_night_rag/scripts/load_to_postgres.py)
- [demo_query.py](/c:/disque%20d/ai_stuff/projects/challanges/ai_night_rag/scripts/demo_query.py)

Examples:

```bash
python scripts/build_local.py
python scripts/load_to_postgres.py
python scripts/demo_query.py
```

## Local Artifacts

The local vector store consists of:

- `data/faiss.index`
- `data/faiss_mapping.jsonl`
- `data/faiss_vectors.npy`

Purpose of each:

- `faiss.index`: the FAISS similarity index
- `faiss_mapping.jsonl`: the stored retrieval text and metadata
- `faiss_vectors.npy`: raw vectors used for PostgreSQL insertion

## Search Behavior

### Local search

Local search uses:

- FAISS `IndexFlatIP`
- normalized vectors
- normalized query vector

This makes inner product behave like cosine similarity.

### PostgreSQL search

PostgreSQL search uses:

- pgvector cosine distance operator `<=>`
- score computed as:

```sql
1 - (vecteur <=> query_vector)
```

This means:

- lower distance = closer vectors
- higher score = more similar result

## Returned Search Format

Both backends return exactly:

```python
[
    {
        "texte_fragment": "...",
        "score": 0.0,
    }
]
```

No extra fields are returned by the search API.

## Notes and Assumptions

- the project currently assumes source PDFs are in `data/raw_pdfs`
- the local build must happen before pushing to PostgreSQL
- the local build must happen before local search
- pushing to PostgreSQL must happen before PostgreSQL search
- Mistral access is required for chunk generation
- PostgreSQL access is required only for PostgreSQL insert/search operations

## Testing

Run:

```bash
python -m pytest -q
```

Current smoke tests cover:

- local FAISS retrieval result shape
- importability of the main DB/search entry points

## Recommended User Flow

For the simplest manual usage:

1. set your environment variables
2. uncomment the build block in `main.py` and run it
3. uncomment the local search block in `main.py` and test retrieval
4. if needed, uncomment the PostgreSQL push block
5. uncomment the PostgreSQL search block and test database retrieval
