# Semantic Search / RAG Prototype (AI Night Challenge)

This repository is my attempt at the **AI Night challenge** proposed by
**STE AGRO MELANGE TECHNOLOGIE -- ROSE BLANCHE Group**, where the goal
is to build a semantic search module (RAG-style) over a vector database
of technical ingredient/additive documents.

I implemented a full RAG pipeline (PDF → chunks → embeddings → vector
store → search), with two retrieval backends:

-   Local vector store with FAISS
-   PostgreSQL + pgvector (same schema as described in the statement)

## Important note about the provided PostgreSQL database

The instructions mention that embeddings are already stored in a
PostgreSQL database. However, in the provided Drive content, I could not
find any database credentials (even read-only) or a DB dump to test
against.

Because of that, I chose to: 
- build my own vector store from the
provided PDF dataset, and
- keep full compatibility with the expected PostgreSQL schema so the
same search module can query Postgres once credentials are available.

------------------------------------------------------------------------

## Why chunking matters for the expected output format

The expected output is very short and specific answer fragments, for
example:

**Question** \> Améliorant de panification : quelles sont les quantités
recommandées d'alpha-amylase, xylanase et d'Acide ascorbique ?

**Expected outputs (indicative)** - "Dosage recommandé : 0.005% à 0.02%
du poids de farine."
- "Alpha-amylase : utilisation entre 5 et 20 ppm selon la farine."
- "Xylanase : améliore l'extensibilité de la pâte..."

A basic approach that embeds document text chunks as-is tends to
struggle here:

-   A very short answer like "Dosage recommandé : 0.005% à 0.02%..."
    contains little context.
-   Embedding such a short fragment alone can drift away from the user
    question embedding.
-   In practice, stronger embeddings often need more context to align
    reliably with rich questions.

So the retrieval must return short answers while still being searchable
by question-like meaning.

------------------------------------------------------------------------

## My chunking approach (agentic Q/A chunk generation)

To produce answer fragments that match the expected format while still
retrieving well:

1.  Each document is passed to an LLM (Mistral).
2.  The LLM generates many question/answer pairs that collectively cover
    the document content.
3.  During indexing:
    -   I embed the generated question.
    -   I store the generated answer.
4.  During retrieval:
    -   The user query is embedded.
    -   Similarity is computed against the embeddings of the generated
        questions.
    -   The system returns the stored answer text (short, specific).

This way, the indexed vectors are "question-shaped", and the returned
text is "answer-shaped".

------------------------------------------------------------------------


## Fixed constraints 

-   Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
-   Embedding dimension: `384`
-   Similarity: cosine similarity
-   Returned results: Top K = 3
-   Returned fields (for each result):
    -   `texte_fragment`
    -   `score`

------------------------------------------------------------------------

## Backends supported

### 1) Local FAISS backend

-   Uses FAISS `IndexFlatIP`
-   Vectors are L2-normalized, so inner product behaves like cosine
    similarity
-   Stores:
    -   `data/faiss.index`
    -   `data/faiss_mapping.jsonl`
    -   `data/faiss_vectors.npy`

### 2) PostgreSQL + pgvector backend

Matches the expected schema:

-   Table: `embeddings`
-   Columns:
    -   `id` (primary key)
    -   `id_document` (int)
    -   `texte_fragment` (text)
    -   `vecteur` `VECTOR(384)`

Cosine scoring is computed as:

1 - (vecteur \<=\> query_vector)

------------------------------------------------------------------------

## Project structure

README.md\
main.py\
requirements.txt\
.env.example

data/\
raw_pdfs/

models_cache/

src/\
config.py\
parser.py\
chunking.py\
embed.py\
db.py\
search.py

scripts/\
init_db.sql\
build_local.py\
load_to_postgres.py\
demo_query.py

tests/\
test_search_smoke.py

------------------------------------------------------------------------

## Installation

Install dependencies:

`pip install -r requirements.txt`

## Environment Configuration (.env)

This project includes a `.env.example` file.\
You must create a `.env` file and fill the required variables for
correct functionality.

### Required depending on usage

If you plan to:

-   Rebuild the vector store (LLM-based chunking)
-   Use the agent-assisted search (`search_topk_with_agent`)

You must define:

    MISTRAL_API_KEY=your_mistral_api_key

If you plan to:

-   Push vectors to PostgreSQL
-   Search using the PostgreSQL backend

You must define:

    DATABASE_URL=postgresql://user:password@host:port/dbname

if basic approach , aka use of prebuild vs and normal retrieval (search_topk)
theres no need to define anything 

### Optional (default values provided)

The following variables are optional and already have default values in
the code:

    TOP_K=3
    MODEL_CACHE_DIR=models_cache

If not provided, the system will still work using the default
configuration.

### how to use

Then run the main entrypoint:

`python main.py`

The `main.py` file contains the high-level APIs to: 
- build the vector store
- query the local vector store
- push the vector store to PostgreSQL (after providing a full
DATABASE_URL in config)
- query PostgreSQL using cosine similarity

you can chose what ever you need by simply uncommenting the funciton you want to use 
whats set :
build_vs() (vs already build locally and will exist when cloning this repo so it will say vs exists)
and search query , which will return and print the chunks as needed  

------------------------------------------------------------------------

## Returned Search Format

Both backends return exactly:

\[ {"texte_fragment": "...", "score": 0.87}, {"texte_fragment": "...",
"score": 0.82}, {"texte_fragment": "...", "score": 0.79}\]

Only `texte_fragment` and `score` are returned.
a print like that was asked is printed too

## Agentic Search Extension 

In addition to the standard `search_topk` function, this project
includes an optional agent-assisted retrieval method:

`search_topk_with_agent`

This function extends the standard cosine similarity search with a
lightweight iterative refinement loop.

### Environment Requirement

To use the agent-assisted search, you must define the following
environment variables in your `.env` file:

-   `MISTRAL_API_KEY`
-   `MISTRAL_MODEL`

Example:

    MISTRAL_API_KEY=your_mistral_api_key
    MISTRAL_MODEL=mistral-small-latest

Without these variables, the agentic search functionality will not run.

### How it works

1.  The query is searched normally using the selected backend (`local`
    or `postgres`).
2.  The Top 3 results are passed to an LLM-based evaluator.
3.  The agent decides:
    -   If at least one result is satisfactory → return the results
        immediately.
    -   If not satisfactory → reformulate the query.
4.  The reformulated query is searched again.
5.  This loop runs:
    -   Up to 3 attempts, or
    -   Until the agent marks the results as satisfactory.
6.  If no satisfactory result is found after 3 attempts:
    -   The system returns the highest-ranked result selected by the
        agent.

**Important:**

-   The returned format remains exactly the same as `search_topk`.
-   The system still returns only:
    -   `texte_fragment`
    -   `score`
-   The refinement logic does not alter the output schema.


The agentic refinement is an additional enhancement layer, not a
replacement.

Fully optional and separated from the base retrieval logic

The default search behavior remains the standard vector similarity
search.


## Future suggestion (model choice)

The imposed embedding model is `all-MiniLM-L6-v2`. For a bilingual
(French/English) setting, a multilingual embedding model like
`multilingual-e5-base` or `multilingual-e5-large` would likely be more
suitable in real deployment.

(This is only a recommendation; the prototype follows the imposed model
constraint.)

------------------------------------------------------------------------