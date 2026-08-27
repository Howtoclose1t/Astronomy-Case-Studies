from pathlib import Path
import yaml

from .paths import resolve_project_path


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file using a project-relative path."""
    resolved = resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
