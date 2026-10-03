"""Write and open Geometry Nodes instruction text files."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def instructions_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    folder = Path(base) / "localtext3d" / "instructions"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _safe_filename(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
    return (cleaned[:56] or "GN").strip("_")


MAX_INSTRUCTION_INDEX = 999


def build_instructions_document(
    *,
    object_name: str,
    prompt: str,
    model: str,
    seed: int,
    spec: dict[str, Any],
    instructions_text: str,
) -> str:
    title = str(spec.get("name") or object_name)
    description = str(spec.get("description") or "").strip()
    lines = [
        f"{title} — How to use",
        "=" * max(len(title) + 14, 24),
        "",
        f"Prompt: {prompt}",
        f"Model: {model}",
        f"Seed: {seed}",
        f"Object name: {object_name}",
        "",
    ]
    if description:
        lines.extend([description, ""])
    lines.append("Steps")
    lines.append("-----")
    lines.append(instructions_text)
    lines.extend(
        [
            "",
            "Tips",
            "----",
            "- Select the generated object, then use the Geometry Nodes modifier in the Properties panel.",
            "- Exposed inputs appear on the modifier once the node group defines them.",
            "- Click the node-group button on the modifier to edit the procedural tree.",
            "",
        ]
    )
    inputs = spec.get("inputs") or []
    if isinstance(inputs, list) and inputs:
        lines.append("Exposed inputs")
        lines.append("----------------")
        for item in inputs:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            default = item.get("default")
            hint = f" (default: {default})" if default is not None else ""
            lines.append(f"- {name}{hint}")
        lines.append("")
    return "\n".join(lines)


def write_instructions_file(
    *,
    object_name: str,
    prompt: str,
    model: str,
    seed: int,
    spec: dict[str, Any],
    instructions_text: str,
) -> str:
    folder = instructions_dir()
    path = folder / f"{_safe_filename(object_name)}_instructions.txt"
    if path.exists():
        index = 1
        while (folder / f"{_safe_filename(object_name)}_instructions_{index:03d}.txt").exists():
            index += 1
            if index > MAX_INSTRUCTION_INDEX:
                raise RuntimeError("Too many instruction files for this object name")
        path = folder / f"{_safe_filename(object_name)}_instructions_{index:03d}.txt"
    body = build_instructions_document(
        object_name=object_name,
        prompt=prompt,
        model=model,
        seed=seed,
        spec=spec,
        instructions_text=instructions_text,
    )
    path.write_text(body, encoding="utf-8")
    return str(path)


def open_text_file(path: str) -> None:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Instructions file not found: {path}")
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", path])  # noqa: S603
        return
    subprocess.Popen(["xdg-open", path])  # noqa: S603


def reveal_in_folder(path: str) -> None:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Instructions file not found: {path}")
    normalized = os.path.normpath(path)
    if sys.platform == "win32":
        subprocess.Popen(["explorer", "/select,", normalized])  # noqa: S603
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", normalized])  # noqa: S603
        return
    open_text_file(path)
