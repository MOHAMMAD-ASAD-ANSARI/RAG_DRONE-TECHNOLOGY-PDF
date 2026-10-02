"""All runtime settings, read from environment / .env."""
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL") or None
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_PATH = os.getenv("QDRANT_PATH") or None
COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_chunks")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_MAX_SEQ_LEN = int(os.getenv("EMBED_MAX_SEQ_LEN", "384"))
# ------added------
EMBED_QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX", "")
EMBED_DOC_PREFIX = os.getenv("EMBED_DOC_PREFIX", "")

SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-flash")
