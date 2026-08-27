from pathlib import Path


def project_root() -> Path:
    """Return the task directory from any module inside src."""
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a project-relative path without storing machine-specific paths."""
    path = Path(path)
    if path.is_absolute():
        raise ValueError(f"Expected a relative path, got: {path}")
    return (project_root() / path).resolve()
