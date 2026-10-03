#!/usr/bin/env python3
"""Zip the Blender 5.1 extension from addon/."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADDON = ROOT / "addon"
DIST = ROOT / "dist"
MANIFEST = ADDON / "blender_manifest.toml"
# Only ship what Blender needs; stray editor/backup files in addon/ are skipped.
INCLUDE_SUFFIXES = {".py", ".toml"}


def addon_version() -> str:
    """Single source of truth: the version in blender_manifest.toml."""
    text = MANIFEST.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    if not match:
        raise SystemExit(f"No version field in {MANIFEST}")
    return match.group(1)


def zip_path() -> Path:
    return DIST / f"localtext3d-{addon_version()}.zip"


def build() -> Path:
    DIST.mkdir(exist_ok=True)
    dest = zip_path()
    if dest.exists():
        dest.unlink()
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ADDON.rglob("*")):
            if path.is_dir():
                continue
            if "__pycache__" in path.parts or path.suffix not in INCLUDE_SUFFIXES:
                continue
            archive.write(path, path.relative_to(ADDON).as_posix())
    return dest


def main() -> int:
    dest = build()
    print(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
