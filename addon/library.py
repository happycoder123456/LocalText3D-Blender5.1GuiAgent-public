"""Discover local Quixel Bridge / Megascans / Fab downloads. No remote marketplaces."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

IMAGE_EXT = {".png", ".jpg", ".jpeg"}
HINTS = ("preview", "thumb", "albedo", "diffuse", "diff")
SKIP_DIRS = {".git", "__pycache__", "node_modules", "uassets", ".vs"}
MAX_CATALOG_BYTES = 32_000_000
MAX_EMBEDDED_URI = 4_000_000
STOP = {
    "a", "an", "the", "of", "with", "and", "or", "for", "to", "in", "on", "at",
    "from", "by", "is", "it", "this", "that", "3d", "asset", "model", "object",
    "mesh", "scan", "quixel", "megascan", "megascans", "fab", "bridge", "local",
}
MIN_SCORE = 5
TOKEN_RE = re.compile(r"[a-z0-9]+")
FLAT_TYPES = ("decal", "surface", "atlas", "displacement", "imperfection", "metahuman")
MATERIAL_WORDS = {
    "asphalt", "concrete", "wood", "wooden", "metal", "stone", "brick", "dirt",
    "sand", "rust", "paint", "texture", "material", "surface", "albedo",
    "gravel", "mud", "plaster", "marble", "leather", "fabric", "soil", "tarmac",
    "bitumen", "pothole", "alpha",
}


def default_library_roots() -> list[Path]:
    home = Path.home()
    docs = home / "Documents"
    local = Path(os.environ.get("LOCALAPPDATA") or (home / "AppData" / "Local"))
    return [
        docs / "Megascans Library",
        home / "Megascans Library",
        docs / "Fab",
        home / "Fab",
        local / "Fab",
        docs / "Quixel",
        home / "Quixel",
        docs / "Quixel Megascans",
        local / "Quixel",
        local / "Bridge",
    ]


def resolve_library_root(explicit: str = "") -> Path | None:
    roots = all_library_roots(explicit)
    return roots[0] if roots else None


def all_library_roots(explicit: str = "") -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    candidates = []
    if explicit.strip():
        candidates.append(Path(explicit.strip()).expanduser())
    candidates.extend(default_library_roots())
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not resolved.is_dir() or resolved in seen:
            continue
        if any(resolved == parent or parent in resolved.parents for parent in seen):
            continue
        seen.add(resolved)
        found.append(resolved)
    return found


def _uassets_catalog(root: Path) -> Path | None:
    for candidate in (
        root / "Downloaded" / "UAssets" / "uassetsData.json",
        root / "UAssets" / "uassetsData.json",
        root / "uassetsData.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def _best_preview_file(folder: Path) -> Path | None:
    if not folder.is_dir():
        return None
    ranked: list[tuple[int, Path]] = []
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXT:
            continue
        name = path.name.lower()
        score = 0
        if path.stem.lower() == folder.name.lower():
            score += 4
        if "preview" in name or "retina" in name:
            score += 3
        if "thumb" in name:
            score += 2
        if "albedo" in name:
            score += 1
        ranked.append((score, path))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1].stat().st_size), reverse=True)
    return ranked[0][1]


def _embedded_preview(asset_json: Path) -> str:
    try:
        data = json.loads(asset_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return ""
    images = ((data.get("previews") or {}).get("images") or [])
    best = ""
    for item in images:
        if isinstance(item, str):
            uri = item.strip()
        elif isinstance(item, dict):
            uri = str(item.get("uri") or item.get("url") or "").strip()
        else:
            continue
        if len(uri) < 200:
            continue
        if len(uri) > MAX_EMBEDDED_URI:
            continue
        if uri.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".exr")):
            continue
        if len(uri) > len(best):
            best = uri
    return best


def _asset_dict(raw: dict, folder: Path) -> dict:
    asset_id = str(raw.get("id") or "").strip()
    name = str(raw.get("name") or asset_id)
    asset_json = folder / f"{asset_id}.json"
    preview = _best_preview_file(folder)
    search = f"{name} {raw.get('searchStr') or ''} {raw.get('assetType') or ''}"
    return {
        "id": asset_id,
        "name": name,
        "type": str(raw.get("assetType") or ""),
        "search": search,
        "folder": str(folder),
        "preview_file": str(preview) if preview else "",
        "preview_url": str(raw.get("previewUrl") or ""),
        "embedded": _embedded_preview(asset_json) if asset_json.is_file() else "",
    }


def assets_from_uassets(catalog: Path, query: str = "", limit: int = 40) -> list[dict]:
    try:
        if catalog.stat().st_size > MAX_CATALOG_BYTES:
            return []
        payload = json.loads(catalog.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    items = []
    for raw in payload.get("assets") or []:
        if not isinstance(raw, dict):
            continue
        asset_id = str(raw.get("id") or "").strip()
        folder = catalog.parent / asset_id
        items.append(_asset_dict(raw, folder))
    return _filter_rank(items, query, limit)


def _is_asset_folder(filenames: list[str]) -> bool:
    lower = [name.lower() for name in filenames]
    return any(any(hint in name for hint in HINTS) for name in lower)


def assets_from_walk(root: Path, query: str = "", limit: int = 40) -> list[dict]:
    items: list[dict] = []
    root = root.resolve()
    for dirpath, dirnames, filenames in root.walk() if hasattr(root, "walk") else _walk(root):
        rel = dirpath.relative_to(root)
        if len(rel.parts) > 6:
            dirnames[:] = []
            continue
        dirnames[:] = [name for name in dirnames if name.lower() not in SKIP_DIRS]
        if not _is_asset_folder(filenames):
            continue
        preview = _best_preview_file(dirpath)
        if preview is None:
            continue
        name = dirpath.name
        items.append(
            {
                "id": name,
                "name": name,
                "type": "file",
                "search": f"{name} {preview.name}",
                "folder": str(dirpath),
                "preview_file": str(preview),
                "preview_url": "",
                "embedded": "",
            }
        )
        if len(items) >= max(limit, 400):
            break
    return _filter_rank(items, query, limit)


def _walk(root: Path):
    import os

    for dirpath, dirnames, filenames in os.walk(root):
        yield Path(dirpath), dirnames, filenames


def tokens(text: str) -> list[str]:
    return [word for word in TOKEN_RE.findall(text.lower()) if len(word) >= 3 and word not in STOP]


def score_asset(prompt: str, asset: dict) -> int:
    needles = tokens(prompt)
    if not needles:
        return 0
    name = (asset.get("name") or "").lower()
    blob = f"{name} {asset.get('search') or ''} {asset.get('type') or ''}".lower()
    name_words = set(tokens(name))
    blob_words = set(tokens(blob))
    score = 0
    hits = 0
    name_hits = 0
    for needle in needles:
        if needle in name_words:
            score += 6
            hits += 1
            name_hits += 1
        elif needle in name:
            score += 4
            hits += 1
            name_hits += 1
        elif needle in blob_words:
            score += 3
            hits += 1
        elif any(len(word) >= 4 and (word.startswith(needle) or needle.startswith(word)) for word in blob_words):
            score += 1
    atype = (asset.get("type") or "").lower()
    if "3d" in atype:
        score += 2
    elif not name_hits and ("decal" in atype or "metahuman" in atype):
        score -= 3
    if asset.get("preview_file"):
        score += 1
    return score if hits else 0


def rank_assets(assets: list[dict], prompt: str) -> list[tuple[int, dict]]:
    ranked = [(score_asset(prompt, asset), asset) for asset in assets]
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def best_match(assets: list[dict], prompt: str, min_score: int = MIN_SCORE) -> dict | None:
    ranked = rank_assets(assets, prompt)
    if not ranked or ranked[0][0] < min_score:
        return None
    return ranked[0][1]


def is_flat_asset(asset: dict) -> bool:
    kind = (asset.get("type") or "").lower()
    return any(key in kind for key in FLAT_TYPES)


def is_shape_asset(asset: dict) -> bool:
    kind = (asset.get("type") or "").lower()
    return "3d" in kind and not is_flat_asset(asset)


def object_overlap(prompt: str, asset: dict) -> set[str]:
    return set(tokens(asset.get("name") or "")) & set(tokens(prompt)) - MATERIAL_WORDS


def pick_auto_scan(assets: list[dict], prompt: str) -> tuple[str, dict] | None:
    ranked = rank_assets(assets, prompt)
    for score, asset in ranked:
        if score < MIN_SCORE:
            break
        if is_shape_asset(asset) and object_overlap(prompt, asset):
            return "shape", asset
    for score, asset in ranked:
        if score < MIN_SCORE:
            break
        return "look", asset
    return None


def _filter_rank(items: list[dict], query: str, limit: int) -> list[dict]:
    if not query.strip():
        return items[:limit]
    ranked = rank_assets(items, query)
    return [asset for score, asset in ranked if score > 0][:limit]


def list_scan_assets(root: Path, query: str = "", limit: int = 40) -> list[dict]:
    catalog = _uassets_catalog(root)
    if catalog is not None:
        found = assets_from_uassets(catalog, query="", limit=400)
        if found:
            return _filter_rank(found, query, limit)
    return assets_from_walk(root, query=query, limit=limit)


def list_all_scan_assets(roots: list[Path], query: str = "", limit: int = 400) -> list[dict]:
    items: list[dict] = []
    seen: set[str] = set()
    for root in roots:
        for asset in list_scan_assets(root, query="", limit=limit):
            key = str(asset.get("id") or asset.get("folder") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            items.append(asset)
    if query.strip():
        return _filter_rank(items, query, limit)
    return items[:limit]
