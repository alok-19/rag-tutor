import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Directories
DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "chroma_db"))
STUDY_DIR = Path(os.getenv("STUDY_MATERIALS_DIR", "study_materials"))

# Chroma DB Settings
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "study_buddy_notes")
REGISTRY_FILE = DB_PATH / "ingestion_registry.json"

# Chunking strategy: "recursive" (default, langchain) | "llama_index"
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "recursive").lower().strip()

# Local hybrid features (opt-in; require the [hybrid] extras to actually run)
# ENABLE_RERANKER:  two-stage retrieval with BGE cross-encoder.
# RERANKER_MODEL:   HF model id (default BAAI/bge-reranker-base).
# RERANKER_INITIAL_K: how many candidates to fetch before reranking.
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower().strip() in {"1", "true", "yes", "on"}
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base").strip()
RERANKER_INITIAL_K = int(os.getenv("RERANKER_INITIAL_K", "20"))

# BGE-M3 local embeddings (used when EMBEDDING_PROVIDER=bge)
BGE_M3_MODEL_NAME = os.getenv("BGE_M3_MODEL_NAME", "BAAI/bge-m3").strip()
BGE_M3_DIM = int(os.getenv("BGE_M3_DIM", "1024"))
BGE_M3_DEVICE = os.getenv("BGE_M3_DEVICE", "cpu").strip()

# LLM Provider (gemini | openai | deepseek)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", LLM_PROVIDER).lower().strip()

# Provider-specific model configs
MODEL_CONFIG = {
    "gemini": {
        "embedding": os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"),
        "primary": os.getenv("GEMINI_PRIMARY_MODEL", "gemini-2.5-flash"),
        "fallback": os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite"),
    },
    "openai": {
        "embedding": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        "primary": os.getenv("OPENAI_PRIMARY_MODEL", "gpt-4o-mini"),
        "fallback": os.getenv("OPENAI_FALLBACK_MODEL", "gpt-3.5-turbo"),
    },
    "deepseek": {
        "embedding": None,  # DeepSeek has no embedding API
        "primary": os.getenv("DEEPSEEK_PRIMARY_MODEL", "deepseek-v4-flash"),
        "fallback": os.getenv("DEEPSEEK_FALLBACK_MODEL", "deepseek-v4-pro"),
    },
    "bge": {
        # BGE is embeddings-only; primary/fallback are placeholders.
        "embedding": os.getenv("BGE_M3_MODEL_NAME", "BAAI/bge-m3"),
        "primary": None,
        "fallback": None,
    },
}

# Backwards-compatible aliases (used by legacy code paths)
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
PRIMARY_GENERATION_MODEL = os.getenv("GEMINI_PRIMARY_MODEL", "gemini-2.5-flash")
FALLBACK_GENERATION_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite")

# Ensure dirs exist
STUDY_DIR.mkdir(exist_ok=True)
DB_PATH.mkdir(exist_ok=True)