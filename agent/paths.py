"""Dataset and template path helpers for the GUI agent sidecar."""

from __future__ import annotations

import os
from pathlib import Path


def default_dataset_dir() -> Path:
    override = os.environ.get("LOCALTEXT3D_AGENT_DATASET")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / "LocalText3D" / "agent_dataset").resolve()


def ensure_dataset(root: Path | None = None) -> Path:
    base = root or default_dataset_dir()
    (base / "screenshots").mkdir(parents=True, exist_ok=True)
    (base / "videos").mkdir(parents=True, exist_ok=True)
    return base


def episodes_path(root: Path | None = None) -> Path:
    return ensure_dataset(root) / "episodes.jsonl"


def memory_path(root: Path | None = None) -> Path:
    return ensure_dataset(root) / "memory.jsonl"


def screenshots_dir(root: Path | None = None) -> Path:
    return ensure_dataset(root) / "screenshots"


def videos_dir(root: Path | None = None) -> Path:
    return ensure_dataset(root) / "videos"


def last_session_path(root: Path | None = None) -> Path:
    return ensure_dataset(root) / "last_session.json"


def default_templates_dir() -> Path:
    return Path(__file__).resolve().parent / "templates"


def prune_old_files(folder: Path, glob_pattern: str, keep: int) -> None:
    """Delete older matches, keeping the newest `keep` files. Best-effort."""
    keep = max(0, int(keep))
    try:
        files = [path for path in folder.glob(glob_pattern) if path.is_file()]
        files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    except OSError:
        return
    for path in files[keep:]:
        try:
            path.unlink()
        except OSError:
            pass
