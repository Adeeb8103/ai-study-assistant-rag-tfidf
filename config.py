"""
Central configuration for the Personal AI Study Assistant.

Everything that a demonstrator might want to tweak during a viva
(model name, chunk size, top-k, prices) lives here in one place.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

# --------------------------------------------------------------------------
# Storage layout
# --------------------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"          # original files the student uploads
INDEX_DIR = DATA_DIR / "indexes"           # FAISS + TF-IDF artefacts per document
EXPORT_DIR = DATA_DIR / "exports"          # CSV exports of evaluation runs
DB_PATH = DATA_DIR / "study_assistant.db"  # SQLite database

for _directory in (DATA_DIR, UPLOAD_DIR, INDEX_DIR, EXPORT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Approach 1 - Modern: OpenRouter + RAG + vector store
# --------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models exposed in the sidebar. Prices are USD per 1 million tokens and are
# only used to *estimate* the running cost shown in the comparison tables.
AVAILABLE_MODELS = {
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "anthropic/claude-3.5-haiku": {"input": 0.80, "output": 4.00},
    "meta-llama/llama-3.1-8b-instruct": {"input": 0.02, "output": 0.03},
    "google/gemini-flash-1.5": {"input": 0.075, "output": 0.30},
}
DEFAULT_MODEL = "openai/gpt-4o-mini"

LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 700
LLM_TIMEOUT_SECONDS = 60

# Sentence-transformer used to turn chunks into vectors.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --------------------------------------------------------------------------
# Shared retrieval settings (used by BOTH approaches so the comparison is fair)
# --------------------------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4

# --------------------------------------------------------------------------
# Approach 2 - Traditional: TF-IDF retrieval tuning
# --------------------------------------------------------------------------
# Stemming collapses "evaluate"/"evaluating"/"evaluation" into one term, so a
# question worded differently from the notes still overlaps them.
TFIDF_USE_STEMMING = True

# Passage (MaxP) scoring. A whole 1000-character chunk holds far more words
# than a one-line question, which dilutes the cosine between them. Each chunk
# is also indexed as short overlapping sentence windows and scored as
#   PASSAGE_WEIGHT * best window + (1 - PASSAGE_WEIGHT) * whole chunk
# floored at the plain whole-chunk cosine. 0.0 restores the old behaviour.
TFIDF_PASSAGE_WEIGHT = 0.85
TFIDF_PASSAGE_SENTENCES = 2   # sentences per window
TFIDF_PASSAGE_STRIDE = 1      # window step, so windows overlap

# Below this cosine the traditional approach reports "no match found".
TFIDF_MIN_SCORE = 0.01

# --------------------------------------------------------------------------
# Conversational memory
# --------------------------------------------------------------------------
MEMORY_WINDOW_TURNS = 6      # how many recent Q/A pairs are replayed to the LLM
MEMORY_SUMMARY_AFTER = 6     # older turns beyond this are compressed to a summary

# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = ["pdf", "txt", "md", "docx"]
APP_TITLE = "Personal AI Study Assistant"
APP_TAGLINE = "Chat with your own notes - Modern RAG vs Traditional TF-IDF"


def has_api_key() -> bool:
    """True when an OpenRouter key is configured (Approach 1 needs it)."""
    return bool(OPENROUTER_API_KEY and OPENROUTER_API_KEY.strip())
