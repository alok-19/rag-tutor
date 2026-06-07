import json
from pathlib import Path
from study_rag.config import REGISTRY_FILE

def load_registry(registry_path: Path = REGISTRY_FILE) -> dict:
    """Load the registry of already ingested files.
    Returns an empty dict if the file is missing, empty, or malformed.
    """
    if registry_path.exists():
        try:
            with open(registry_path, "r") as f:
                content = f.read().strip()
                if not content:
                    return {}
                data = json.loads(content)
                if isinstance(data, dict):
                    return data
                return {}
        except Exception:
            return {}
    return {}

def save_registry(registry: dict, registry_path: Path = REGISTRY_FILE):
    """Save the registry of ingested files."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=4)

def migrate_legacy_registry(registry: dict, subject_name: str = "Operating System") -> dict:
    """Migrate legacy top-level registry records to a subject-grouped record.
    Returns the updated registry.
    """
    if subject_name == "Operating System":
        # Support subject grouping in registry
        subject_registry = registry.get(subject_name, {})
        
        # Check if there are legacy top-level registry records and migrate them to 'Operating System'
        legacy_keys = [k for k in registry.keys() if k.endswith(".pdf")]
        if legacy_keys:
            if not isinstance(subject_registry, dict):
                subject_registry = {}
            for k in legacy_keys:
                if k not in subject_registry:
                    subject_registry[k] = registry[k]
                del registry[k]
            registry[subject_name] = subject_registry
            
    return registry
