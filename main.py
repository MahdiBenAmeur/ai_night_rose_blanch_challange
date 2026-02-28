"""from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = "models_cache"

os.makedirs(CACHE_DIR, exist_ok=True)

model = SentenceTransformer(MODEL_NAME, cache_folder=CACHE_DIR)

print("Model downloaded and cached in:", CACHE_DIR)"""

from src.parser import parse_pdfs

ret = parse_pdfs()

print(ret)