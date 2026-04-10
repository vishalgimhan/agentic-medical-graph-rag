import os
import yaml
from pathlib import Path
from typing import Any, Dict

# Project Paths
_PROJECT_ROOT = Path(__file__).parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"

# YAML Config loading
def _load_yaml(filename: str) -> Dict[str, Any]:
    """
    Load a YAML config file and return its contents as a dictionary.
    """   
    filepath = _CONFIG_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Config file '{filename}' not found in '{_CONFIG_DIR}'")

    with open(filepath, "r") as f:
        return yaml.safe_load(f)

def _get_nested(d: Dict[str, Any], *keys: str, default=None) -> Any:
    """
    Get nested dictionary values safely"""
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d

# Load config
_PARAMS = _load_yaml("params.yaml")

# Neo4j Graph Database
NEO4J_URI = os.getenv(
    _get_nested(_PARAMS, "graph", "uri_env", default="NEO4J_URI"))
NEO4J_USERNAME = os.getenv(
    _get_nested(_PARAMS, "graph", "user_env", default="NEO4J_USERNAME"))
NEO4J_PASSWORD = os.getenv(
    _get_nested(_PARAMS, "graph", "password_env", default="NEO4J_PASSWORD"))
NEO4J_DATABASE = os.getenv(
    _get_nested(_PARAMS, "graph", "database", default="neo4j"))