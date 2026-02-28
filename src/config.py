from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CACHE_DIR = "models_cache"
REQUIRED_TOP_K = 3
REQUIRED_EMBED_DIM = 384
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_DIR = "data/raw_pdfs"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    database_url: str | None
    model_cache_dir: Path
    top_k: int
    embed_dim: int


def _read_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def get_settings(backend: str | None = None) -> Settings:
    top_k = _read_int_env("TOP_K", REQUIRED_TOP_K)
    embed_dim = _read_int_env("EMBED_DIM", REQUIRED_EMBED_DIM)
    database_url = os.getenv("DATABASE_URL")
    model_cache_dir = Path(os.getenv("MODEL_CACHE_DIR", DEFAULT_MODEL_CACHE_DIR))

    if top_k != REQUIRED_TOP_K:
        raise ValueError(f"TOP_K is fixed to {REQUIRED_TOP_K}.")
    if embed_dim != REQUIRED_EMBED_DIM:
        raise ValueError(f"EMBED_DIM is fixed to {REQUIRED_EMBED_DIM}.")
    if backend == "postgres" and not database_url:
        raise ValueError("DATABASE_URL is required when backend='postgres'.")

    return Settings(
        database_url=database_url,
        model_cache_dir=model_cache_dir,
        top_k=top_k,
        embed_dim=embed_dim,
    )
