"""Resolve deployment paths against the checkout, independent of caller cwd."""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configured_path(name: str, default: str | Path) -> Path:
    path = Path(os.getenv(name) or default).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
