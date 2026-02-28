"""Simple entry file for the main project actions.

Uncomment one block at a time depending on what you want to do:
- build the local vector store
- search the local vector store
- push the local vector store to PostgreSQL
- search PostgreSQL
"""

from __future__ import annotations

from src.config import DATA_DIR
from src.db import build_vs, push_vs_to_postgres
from src.search import search_topk


def main() -> None:
    """Run one manually selected workflow from the blocks below."""
    # highly recommend building the vs , as im using a special technique of chunking that will work well for this intended use case
    # Build the local vector store from the PDF directory.
    # If already build , which it is cause i pushed it to github it wouldnt build and just use the local vector store
    # This uses: parser -> chunking -> embeddings -> FAISS save.
    build_vs(DATA_DIR)

    # Search in the local vector store.
    # Make sure the local VS already exists before using this block.
    # you could search locally by passing "local" to the backend paramater or "postgres" for searching in postgres
    query = "What is the package of BVZyme GOX 110?"
    local_results = search_topk(
         question=query,
         backend="local",
    )

    # Push the local vector store into PostgreSQL.
    # This initializes the embeddings table if needed, then inserts rows.
    # push_vs_to_postgres()

    # Search directly in PostgreSQL.
    # Make sure DATABASE_URL is set and the local VS has already been pushed.
    # postgres_results = search_topk(
    #     question="What is the package of BVZyme GOX 110?",
    #     backend="postgres",
    # )
    # print(postgres_results)

    pass


if __name__ == "__main__":
    main()
