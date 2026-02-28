from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from src.chunking import chunk_text
from src.embed import embed_texts
from src.parser import parse_pdfs
from src.search import build_faiss_index, save_faiss_index, save_mapping, save_vectors

RAW_PDF_DIR =  "data" / "raw_pdfs"


def _mock_fragments() -> list[dict[str, object]]:
    texts = [
        "Les enzymes ameliorent la tenue de pate et la regularite du process.",
        "Un traitement enzymatique adapte peut renforcer le volume final du pain.",
        "La recherche semantique retrouve des fragments proches d'une question metier.",
        "Le stockage vectoriel permet un rappel rapide de textes similaires.",
        "La qualite des chunks influence directement la precision du moteur de recherche.",
    ]
    return [
        {"id": idx, "id_document": idx + 1, "texte_fragment": text}
        for idx, text in enumerate(texts)
    ]


def _mock_vectors(size: int, dim: int = 384) -> np.ndarray:
    rng = np.random.default_rng(seed=7)
    vectors = rng.normal(size=(size, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def _try_placeholder_pipeline() -> tuple[list[dict[str, object]], np.ndarray]:
    documents = parse_pdfs(RAW_PDF_DIR)
    fragments: list[dict[str, object]] = []
    for document in documents:
        doc_id = int(document["id_document"])
        for fragment in chunk_text(str(document["text"])):
            fragments.append(
                {
                    "id": len(fragments),
                    "id_document": doc_id,
                    "texte_fragment": fragment,
                }
            )

    vectors = embed_texts([str(item["texte_fragment"]) for item in fragments])
    return fragments, vectors


def main() -> None:
    pdf_files = sorted(RAW_PDF_DIR.glob("*.pdf"))

    try:
        if pdf_files:
            records, vectors = _try_placeholder_pipeline()
        else:
            raise NotImplementedError("No PDFs found in raw_pdfs; using mock data.")
    except NotImplementedError:
        records = _mock_fragments()
        vectors = _mock_vectors(size=len(records))

    index = build_faiss_index(vectors)
    save_faiss_index(index)
    save_mapping(records)
    save_vectors(vectors)

    print(f"Saved {len(records)} fragments to local FAISS artifacts.")


if __name__ == "__main__":
    main()
