"""PDF parsing utilities that convert source documents into structured text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz
from src.config import DATA_DIR

def _normalize_whitespace(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.splitlines()).strip()


def _bboxes_overlap(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> bool:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    return x1 > x0 and y1 > y0


def _format_markdown_table(rows: list[list[Any]]) -> str:
    cleaned_rows: list[list[str]] = []
    for row in rows:
        values = [str(cell).strip() if cell is not None else "" for cell in row]
        if any(values):
            cleaned_rows.append(values)

    if not cleaned_rows:
        return ""

    column_count = max(len(row) for row in cleaned_rows)
    normalized_rows = [row + [""] * (column_count - len(row)) for row in cleaned_rows]

    header = [cell if cell else f"col_{idx + 1}" for idx, cell in enumerate(normalized_rows[0])]
    body = normalized_rows[1:] if len(normalized_rows) > 1 else []

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * column_count) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _extract_page_markdown(page: fitz.Page) -> str:
    table_finder = page.find_tables()
    table_bboxes = [table.bbox for table in table_finder.tables]

    text_blocks: list[str] = []
    for block in sorted(page.get_text("blocks"), key=lambda item: (item[1], item[0])):
        bbox = (float(block[0]), float(block[1]), float(block[2]), float(block[3]))
        raw_text = _normalize_whitespace(str(block[4]))
        if not raw_text:
            continue
        if any(_bboxes_overlap(bbox, table_bbox) for table_bbox in table_bboxes):
            continue
        text_blocks.append(raw_text)

    sections: list[str] = []
    if text_blocks:
        sections.append("\n\n".join(text_blocks))

    for table_index, table in enumerate(table_finder.tables, start=1):
        markdown_table = _format_markdown_table(table.extract())
        if not markdown_table:
            continue
        sections.append(f"### Table {table_index}\n{markdown_table}")

    return "\n\n".join(section for section in sections if section).strip()


def parse_pdfs(input_dir: str | Path = DATA_DIR) -> list[dict]:
    """Parse PDF files into Markdown-like text plus metadata.

    The parser walks through all PDF files in the input directory, extracts the
    text page by page, preserves rough document structure, and converts detected
    tables to Markdown.

    Args:
        input_dir: Directory containing PDF files to parse.

    Returns:
        A list of document dictionaries with ``text`` and ``metadata`` fields.
    """

    base_dir = Path(input_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {base_dir}")

    pdf_files = sorted(base_dir.rglob("*.pdf"))
    documents: list[dict[str, Any]] = []

    for index, pdf_path in enumerate(pdf_files, start=1):
        with fitz.open(pdf_path) as document:
            page_sections: list[str] = []
            for page_number, page in enumerate(document, start=1):
                page_markdown = _extract_page_markdown(page)
                if not page_markdown:
                    continue
                page_sections.append(f"## Page {page_number}\n\n{page_markdown}")

            metadata = {
                "id_document": index,
                "source_path": str(pdf_path.resolve()),
                "file_name": pdf_path.name,
                "page_count": len(document),
                "parser": "pymupdf",
            }
            text = f"# {pdf_path.stem}\n\n" + "\n\n".join(page_sections)
            documents.append(
                {
                    "id_document": index,
                    "source_path": str(pdf_path.resolve()),
                    "text": text.strip(),
                    "metadata": metadata,
                }
            )

    return documents
