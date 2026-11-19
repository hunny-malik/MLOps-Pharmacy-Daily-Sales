import hashlib
from pathlib import Path
from typing import Any, Dict
import yaml


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Load configuration parameters using yaml.safe_load."""
    path = Path(params_path)
    if not path.exists():
        # Look one level up if called from subdirectory
        parent_path = Path("..") / params_path
        if parent_path.exists():
            path = parent_path
        else:
            raise FileNotFoundError(f"Config file not found at: {params_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def compute_file_sha256(file_path: str) -> str:
    """Compute the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()
