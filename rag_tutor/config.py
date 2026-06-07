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

# Models
EMBEDDING_MODEL = "gemini-embedding-2"
PRIMARY_GENERATION_MODEL = "gemini-2.5-flash"
FALLBACK_GENERATION_MODEL = "gemini-1.5-flash"

# Ensure dirs exist
STUDY_DIR.mkdir(exist_ok=True)
DB_PATH.mkdir(exist_ok=True)
