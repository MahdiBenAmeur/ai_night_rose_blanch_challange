"""Chunk generation utilities that turn parsed documents into QA pairs."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from src.config import get_settings
import tqdm
from mistralai import Mistral

RETRY_DELAY_SECONDS = 2
MAX_RETRIES = 3


class ChunkPair(BaseModel):
    question: str = Field(..., description="A specific retrieval question grounded in the document.")
    answer: str = Field(..., description="A short exact answer grounded in the document.")


class ChunkExtraction(BaseModel):
    pairs: list[ChunkPair] = Field(default_factory=list)


def _build_prompt(text: str) -> str:
    return f"""
You are extracting exhaustive retrieval chunks from a document.
the document already have titles and answer under titles , you could use that as insperation but add some other questions too
Your task:
- Read the full document carefully from beginning to end.
- Cover all factual information in the document.
- Generate many small question and answer pairs.
- Do not miss information.
- Keep each answer short, simple, and exact.
- Do not invent or infer facts not explicitly present.
- If the document contains tables, include their values as short answers.
- Prefer many precise pairs over a few broad ones.

Rules:
- Each question must be answerable only from the document text.
- Each answer must be as short as possible while preserving the fact.
- Questions should be useful for semantic retrieval.
- Return structured output only.

Document text:
{text}
""".strip()


def _request_pairs(text: str) -> list[dict[str, str]]:
    settings = get_settings(require_mistral=True)

    client = Mistral(api_key=settings.mistral_api_key)
    prompt = _build_prompt(text)

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.parse(
                model=settings.mistral_model,
                temperature=0.0,
                response_format=ChunkExtraction,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract exhaustive question-answer pairs from the document. "
                            "Answers must stay short and strictly grounded in the text."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise ValueError("Mistral returned no parsed structured output.")

            pairs: list[dict[str, str]] = []
            for item in parsed.pairs:
                question = item.question.strip()
                answer = item.answer.strip()
                if question and answer:
                    pairs.append({"question": question, "answer": answer})

            return pairs
        except Exception as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            print(f"Mistral chunk generation failed. Retrying in {RETRY_DELAY_SECONDS}s ({attempt}/{MAX_RETRIES})...")
            time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError("Failed to generate chunks with Mistral.") from last_error


def chunk_text(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate question-answer chunks for a batch of parsed documents.

    Each document is sent to Mistral as a full text block. The model returns
    structured question-answer pairs, and the function attaches document
    metadata to every generated chunk.

    Args:
        documents: Parsed documents produced by ``src.parser.parse_pdfs``.

    Returns:
        A flat list of chunk dictionaries with ``question``, ``answer``, and
        ``metadata`` keys.
    """

    chunks: list[dict[str, Any]] = []

    for document in tqdm.tqdm(documents):
        normalized_text = str(document.get("text", "")).strip()
        if not normalized_text:
            continue

        base_metadata = dict(document.get("metadata", {}))
        if "id_document" in document:
            base_metadata["id_document"] = int(document["id_document"])
        if "source_path" in document:
            base_metadata["source_path"] = str(document["source_path"])

        pairs = _request_pairs(normalized_text)
        for pair_index, pair in enumerate(pairs, start=1):
            chunk_metadata = dict(base_metadata)
            chunk_metadata["pair_index"] = pair_index
            chunks.append(
                {
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "metadata": chunk_metadata,
                }
            )

    return chunks
