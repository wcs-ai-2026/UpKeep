"""Project paths.

The repository currently stores data in ``Data/`` while the plan and contracts
use ``data/`` (T-12, pending). This works on macOS only because its file system
ignores case, so :func:`data_dir` looks for both spellings.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def data_dir(root: Path = PROJECT_ROOT) -> Path:
    for name in ("data", "Data"):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return root / "data"


def models_dir(root: Path = PROJECT_ROOT) -> Path:
    """Saved models live in ``models_saved/`` (D-05; gitignored)."""
    return root / "models_saved"


def event_info_path(root: Path = PROJECT_ROOT) -> Path:
    return data_dir(root) / "event_info.csv"
