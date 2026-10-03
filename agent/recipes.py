"""Common Blender goal recipes — reliable hotkey sequences when the VLM is vague."""

from __future__ import annotations

import re
from typing import Any


def _norm(goal: str) -> str:
    return re.sub(r"\s+", " ", (goal or "").lower()).strip()


def _step(
    action: str,
    *,
    keys: list[str] | None = None,
    text: str = "",
    reason: str = "",
    seconds: float = 0.5,
) -> dict[str, Any]:
    return {
        "action": action,
        "keys": keys or [],
        "text": text,
        "reason": reason,
        "target": "",
        "x": 0,
        "y": 0,
        "x2": 0,
        "y2": 0,
        "seconds": seconds,
    }


def recipe_for_goal(goal: str) -> list[dict[str, Any]] | None:
    """
    Return a short scripted action list for well-known goals, else None.
    Uses Blender operator search (F3) — avoids Shift+A and bare 'A' (select/deselect).
    """
    g = _norm(goal)
    if not g:
        return None

    wants_cube = bool(re.search(r"\bcube\b", g)) and bool(
        re.search(r"\b(add|place|create|spawn|insert|make|new|another|one|\d+)\b", g)
    )
    if not wants_cube:
        return None

    return [
        _step("wait", reason="Focus Blender", seconds=0.35),
        _step("key", keys=["f3"], reason="Open operator search (F3)"),
        _step("wait", reason="Wait for search box", seconds=0.6),
        _step("type", text="Add Cube", reason="Find Add Cube operator"),
        _step("wait", reason="Wait for results", seconds=0.45),
        _step("key", keys=["enter"], reason="Run Add Cube"),
        _step("wait", reason="Wait for new mesh", seconds=0.5),
        _step("stop", reason="Add Cube finished"),
    ]
