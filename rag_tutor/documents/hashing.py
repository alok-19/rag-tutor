import hashlib
from pathlib import Path

def calculate_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of a file to detect changes."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()
