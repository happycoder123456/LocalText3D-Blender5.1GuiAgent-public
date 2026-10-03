"""Ollama vision policy — refine local vision models with prompts, demos, and memory."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from agent.window import downsample_for_vlm, encode_jpeg_b64
from worker.loopback import read_http_body

OLLAMA_URL = "http://127.0.0.1:11434"
_MAX_OLLAMA_BODY = 8_000_000
MAX_WAIT_SECONDS = 8.0


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError("Refusing HTTP redirect")


_OPENER = urllib.request.build_opener(_NoRedirect)
# Vision-capable families only — plain "llama3.2" is text-only and cannot see screenshots.
VISION_PREFER = (
    "llama3.2-vision",
    "qwen2.5vl",
    "qwen2.5-vl",
    "qwen3-vl",
    "gemma3",
    "llava",
    "bakllava",
    "moondream",
    "minicpm-v",
)
_VISION_MARKERS = ("vision", "vl", "llava", "moondream", "minicpm-v")

SYSTEM_PROMPT = """You control Blender 5.x with mouse/keyboard like a careful human. Output ONE small JSON object only.
Never invent long key lists. keys must have at most 3 items.

Schema:
{"action":"click|drag|key|hotkey|type|wait|stop","target":"","x":0,"y":0,"x2":0,"y2":0,"keys":[],"text":"","reason":""}

Coordinates x,y are pixels in the Blender window client area (origin top-left). Click the actual UI control or viewport target you see.

Blender facts:
- Prefer operator search: F3, type the operator name, Enter. Never press bare A (select/deselect all).
- Object mode vs Edit mode: Tab toggles. Extrude (E), loop cut (Ctrl+R), inset (I) need Edit mode.
- Grab/move G, Scale S, Rotate R. Confirm transforms with Enter or left-click; cancel with Esc.
- Delete: X then Enter, or Delete. Undo: Ctrl+Z.
- Add mesh: F3 then "Add Cube" / "Add Cylinder" / etc. Avoid Shift+A menus unless visible.
- One purposeful action per step. Never spam the same letter. Never more than 3 keys.
- Learn from MEMORY: copy SUCCESS / teacher patterns; never repeat AVOID or BANNED actions.
- If last reward was negative, change keys AND click target — do not nudge the same miss.
- stop only when the goal is visibly done in the viewport.
"""

_JSON_RE = re.compile(r"\{[\s\S]*\}")


class PolicyError(Exception):
    pass


_OLLAMA_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


def safe_ollama_name(name: str, *, field: str = "model") -> str:
    """Reject Modelfile/API injection via newlines or shell-metacharacter names."""
    text = (name or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    if not _OLLAMA_NAME_RE.fullmatch(text) or ".." in text or text.startswith("/"):
        raise ValueError(f"Invalid {field} name")
    return text


def list_ollama_models(timeout: float = 5.0) -> list[str]:
    url = OLLAMA_URL.rstrip("/") + "/api/tags"
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            payload = json.loads(read_http_body(resp, _MAX_OLLAMA_BODY).decode("utf-8"))
    except Exception as exc:
        raise PolicyError(f"Cannot reach Ollama at {OLLAMA_URL}: {exc}") from exc
    names: list[str] = []
    for item in payload.get("models") or []:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            if name:
                names.append(name)
    return names


def prefer_vision_model(models: list[str]) -> str:
    lower_map = {m.lower(): m for m in models}
    for key in VISION_PREFER:
        for name_l, name in lower_map.items():
            if key in name_l:
                return name
    return models[0] if models else "llava"


TEXT_PREFER = (
    "llama3.1",
    "qwen2.5",
    "qwen3",
    "llama3.2-vision",
    "llama3.2",
    "gemma",
    "mistral",
    "phi",
)


def _is_vision_name(name_l: str) -> bool:
    return any(marker in name_l for marker in _VISION_MARKERS)


def prefer_text_model(models: list[str], fallback: str = "") -> str:
    """Pick a fast local text model for planning (vision weights are slower and worse at JSON)."""
    lower_map = {m.lower(): m for m in models}
    # Pass 1: text-only models. Pass 2: anything (a vision model still plans, just slower).
    for allow_vision in (False, True):
        for key in TEXT_PREFER:
            for name_l, name in lower_map.items():
                if key not in name_l or "coder" in name_l or "embed" in name_l:
                    continue
                if not allow_vision and _is_vision_name(name_l):
                    continue
                return name
    return fallback or (models[0] if models else "")


def resolve_planner_model(requested: str, vision_model: str = "") -> str:
    """'auto'/'' → best installed text model; 'none' → no LLM planner; else the requested name."""
    req = (requested or "").strip()
    if req.lower() == "none":
        return ""
    if req and req.lower() != "auto":
        return req
    try:
        models = list_ollama_models(timeout=3.0)
    except PolicyError:
        return vision_model or ""
    return prefer_text_model(models, fallback=vision_model)


def _repair_json(text: str) -> str:
    """Best-effort repair for truncated model JSON."""
    text = text.strip()
    if not text:
        return text
    start = text.find("{")
    if start < 0:
        return text
    text = text[start:]
    # Truncate runaway keys arrays.
    text = re.sub(
        r'"keys"\s*:\s*\[(?:[^\[\]]|\[[^\]]*\]){0,800}',
        lambda m: m.group(0)[:80].rstrip(",") + '"]',
        text,
        count=1,
    )
    if text.count("{") > text.count("}"):
        text += "}" * (text.count("{") - text.count("}"))
    if text.count("[") > text.count("]"):
        text += "]" * (text.count("[") - text.count("]"))
    # Trim trailing commas before } ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def clamp_wait_seconds(value: Any, default: float = 0.5) -> float:
    try:
        sec = float(value)
    except (TypeError, ValueError):
        return default
    if sec != sec or sec in {float("inf"), float("-inf")}:
        return default
    return float(max(0.0, min(MAX_WAIT_SECONDS, sec)))


def _coord(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def parse_action(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise PolicyError("Empty model response")
    data: Any = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_RE.search(text)
        candidates = [match.group(0)] if match else []
        candidates.append(_repair_json(text))
        for cand in candidates:
            try:
                data = json.loads(cand)
                break
            except json.JSONDecodeError:
                continue
    if not isinstance(data, dict):
        # Last resort: pull fields with regex.
        action_m = re.search(r'"action"\s*:\s*"(\w+)"', text)
        if not action_m:
            raise PolicyError(f"No JSON in model response: {text[:160]}")
        data = {
            "action": action_m.group(1),
            "keys": re.findall(r'"([A-Za-z0-9]+)"', text[text.find("keys") : text.find("keys") + 80])
            if "keys" in text
            else [],
            "reason": "repaired",
        }

    action = str(data.get("action") or "wait").lower().strip()
    if action not in {"click", "drag", "key", "hotkey", "type", "wait", "stop"}:
        action = "wait"
    keys = data.get("keys") or []
    if isinstance(keys, str):
        keys = [keys]
    cleaned: list[str] = []
    for k in keys:
        label = str(k).strip()
        if not label:
            continue
        label = label.replace("Key.", "").replace("key.", "")
        cleaned.append(label)
        if len(cleaned) >= 3:
            break
    # Collapse spam like C,C,C -> single C
    if len(cleaned) >= 2 and len(set(x.lower() for x in cleaned)) == 1:
        cleaned = [cleaned[0]]

    return {
        "action": action,
        "target": str(data.get("target") or "")[:64],
        "x": _coord(data.get("x")),
        "y": _coord(data.get("y")),
        "x2": _coord(data.get("x2")),
        "y2": _coord(data.get("y2")),
        "keys": cleaned,
        "text": str(data.get("text") or "")[:64],
        "seconds": clamp_wait_seconds(data.get("seconds"), 0.5 if action == "wait" else 0.0),
        "reason": str(data.get("reason") or "")[:120],
    }


def _chat_vision(
    model: str,
    messages: list[dict[str, Any]],
    timeout: float = 120.0,
    *,
    num_predict: int = 96,
    temperature: float = 0.0,
    num_ctx: int = 8192,
) -> str:
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "keep_alive": "10m",
        "options": {
            "temperature": float(temperature),
            "num_predict": int(num_predict),
            "num_ctx": int(num_ctx),
        },
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL.rstrip("/") + "/api/chat",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            payload = json.loads(read_http_body(resp, _MAX_OLLAMA_BODY).decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            raw = read_http_body(exc, min(_MAX_OLLAMA_BODY, 1_000_000)).decode("utf-8", "replace")
        except Exception:
            raw = str(exc)
        raise PolicyError((raw or str(exc))[:300]) from exc
    except Exception as exc:
        raise PolicyError(f"Ollama chat failed: {exc}") from exc
    message = payload.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    if not content:
        raise PolicyError("Ollama returned an empty response")
    return str(content)


def chat_vision_json(
    model: str,
    *,
    system: str,
    user_text: str,
    images_b64: list[str] | None = None,
    num_predict: int = 512,
    timeout: float = 180.0,
) -> str:
    """Public helper for multi-image JSON vision calls (video teacher)."""
    content: Any = user_text
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": content,
            "images": list(images_b64 or [])[:4],
        },
    ]
    return _chat_vision(
        model,
        messages,
        timeout=timeout,
        num_predict=num_predict,
        temperature=0.1,
    )


def chat_text_json(
    model: str,
    *,
    system: str,
    user_text: str,
    num_predict: int = 600,
    temperature: float = 0.1,
    timeout: float = 180.0,
    num_ctx: int = 8192,
) -> str:
    """Text-only JSON chat (planner, video captions fallback)."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]
    # Rough token estimate (≈4 chars/token); Ollama silently drops the *head*
    # (our system prompt) when the prompt exceeds num_ctx.
    approx_tokens = (len(system) + len(user_text)) // 4 + int(num_predict)
    ctx = int(num_ctx)
    while ctx < approx_tokens + 512 and ctx < 32768:
        ctx *= 2
    return _chat_vision(
        model,
        messages,
        timeout=timeout,
        num_predict=num_predict,
        temperature=temperature,
        num_ctx=ctx,
    )


def create_blender_gui_modelfile(base_model: str, alias: str = "blender-gui") -> str:
    """Create a local Ollama alias FROM base_model with our system prompt (same weights)."""
    base_model = safe_ollama_name(base_model, field="base_model")
    alias = safe_ollama_name(alias, field="alias")
    system = " ".join(SYSTEM_PROMPT.split())
    # Current Ollama API: {"model","from","system"}; older servers only accept
    # {"name","modelfile"}. Try new first, fall back on 4xx.
    new_body = {"model": alias, "from": base_model, "system": system, "stream": False}
    old_body = {"name": alias, "modelfile": f"FROM {base_model}\nSYSTEM {json.dumps(system)}\n", "stream": False}
    last_err = ""
    for body in (new_body, old_body):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL.rstrip("/") + "/api/create",
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with _OPENER.open(req, timeout=300) as resp:
                read_http_body(resp, _MAX_OLLAMA_BODY)  # may be NDJSON even with stream:false
            return alias
        except urllib.error.HTTPError as exc:
            last_err = exc.read().decode("utf-8", "replace")[:200] or str(exc)
            if exc.code >= 500:
                break
            continue
        except Exception as exc:
            last_err = str(exc)[:200]
            break
    raise PolicyError(f"Failed to create {alias}: {last_err}")


def validate_and_repair_action(
    action: dict[str, Any],
    *,
    width: int = 0,
    height: int = 0,
    banned_sigs: set[str] | None = None,
) -> dict[str, Any]:
    """Clamp coords, block bare A, reject banned signatures."""
    out = dict(action or {})
    kind = str(out.get("action") or "wait").lower()
    keys = out.get("keys") or []
    if isinstance(keys, str):
        keys = [keys]
    cleaned: list[str] = []
    for k in keys:
        label = str(k).strip().replace("Key.", "").replace("key.", "")
        if not label:
            continue
        # Bare A is almost always select/deselect thrash.
        if label.lower() == "a" and kind in {"key", "hotkey"} and len(keys) == 1:
            return {"action": "wait", "seconds": 0.35, "reason": "blocked bare A"}
        cleaned.append(label)
        if len(cleaned) >= 3:
            break
    out["keys"] = cleaned
    out["seconds"] = clamp_wait_seconds(out.get("seconds"), 0.5 if kind == "wait" else 0.0)
    if width > 0 and height > 0 and kind in {"click", "drag"}:
        out["x"] = int(max(0, min(int(out.get("x") or 0), width - 1)))
        out["y"] = int(max(0, min(int(out.get("y") or 0), height - 1)))
        if kind == "drag":
            out["x2"] = int(max(0, min(int(out.get("x2") or out["x"]), width - 1)))
            out["y2"] = int(max(0, min(int(out.get("y2") or out["y"]), height - 1)))
    if banned_sigs:
        try:
            from agent.memory import action_signature
            sig = action_signature(out)
            if sig and sig in banned_sigs:
                return {"action": "wait", "seconds": 0.4, "reason": "blocked banned signature"}
        except Exception:
            pass
    return out


class VisionPolicy:
    def __init__(self, model: str = "llama3.2-vision"):
        self.model = model

    def decide(
        self,
        bgr_image: Any,
        *,
        goal: str,
        detections: list[dict[str, Any]],
        memory_hints: dict[str, list[dict[str, Any]]] | None = None,
        demos: list[dict[str, Any]] | None = None,
        last_reward: float | None = None,
        last_note: str = "",
        banned_sigs: set[str] | None = None,
        recent_failures: list[dict[str, Any]] | None = None,
        allowed_actions: set[str] | None = None,
    ) -> dict[str, Any]:
        small = downsample_for_vlm(bgr_image, max_side=896)
        b64 = encode_jpeg_b64(small, quality=85)
        det_lines = []
        for d in detections:
            x, y, w, h = d["bbox"]
            det_lines.append(
                f"- {d['class']} bbox=[{x},{y},{w},{h}] score={d.get('score', 0):.2f}"
            )
        det_block = "\n".join(det_lines) if det_lines else "(none — use hotkeys)"

        mem_lines: list[str] = []
        if memory_hints:
            for row in memory_hints.get("successes") or []:
                src = str(row.get("source") or "agent")
                label = "TEACHER" if src in {"teacher", "success"} else "SUCCESS"
                mem_lines.append(
                    f"{label} (reward={row.get('reward')}): action={json.dumps(row.get('action'))}"
                )
            for row in memory_hints.get("mistakes") or []:
                mem_lines.append(
                    f"AVOID (reward={row.get('reward')}): action={json.dumps(row.get('action'))} "
                    f"why={row.get('reason') or ''}"
                )
        mem_block = "\n".join(mem_lines) if mem_lines else "(no prior memory)"

        demo_lines: list[str] = []
        for demo in demos or []:
            demo_lines.append(f"HUMAN DEMO events: {json.dumps(demo.get('events') or [])[:300]}")
        demo_block = "\n".join(demo_lines) if demo_lines else "(no human demos)"

        ban_block = ""
        if banned_sigs:
            ban_block = "BANNED this run (do not repeat): " + ", ".join(sorted(banned_sigs)[:12])

        fail_lines: list[str] = []
        for row in recent_failures or []:
            fail_lines.append(
                f"FAILED: {json.dumps(row.get('action'))} reward={row.get('reward')} note={row.get('note')}"
            )
        fail_block = "\n".join(fail_lines) if fail_lines else ""

        critic = ""
        if last_reward is not None:
            critic = (
                f"Last step reward={last_reward:.3f}. Note: {last_note or 'n/a'}. "
                "If negative, do NOT repeat the same action — try a different approach."
            )

        tip = ""
        if re.search(r"\bcube\b", goal or "", re.I):
            tip = 'Hint: for a cube use {"action":"key","keys":["f3"],"reason":"operator search"} then type Add Cube'

        allow_block = ""
        if allowed_actions:
            allow_block = (
                "ALLOWED actions only: "
                + ", ".join(sorted(a for a in allowed_actions if a))
                + ". Do not invent other action types."
            )

        user_text = (
            f"Goal: {goal}\n{tip}\n\n"
            f"Vision detections:\n{det_block}\n\n"
            f"Memory (learn from these):\n{mem_block}\n\n"
            f"Demos:\n{demo_block}\n\n"
            f"{ban_block}\n"
            f"{allow_block}\n"
            f"{fail_block}\n"
            f"{critic}\n"
            "Output ONE short JSON action. Max 3 keys. Prefer SUCCESS patterns; never repeat AVOID."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_text,
                "images": [b64],
            },
        ]
        raw = _chat_vision(self.model, messages)
        action = parse_action(raw)
        h, w = 0, 0
        try:
            h, w = int(bgr_image.shape[0]), int(bgr_image.shape[1])
        except Exception:
            pass
        action = validate_and_repair_action(
            action, width=w, height=h, banned_sigs=banned_sigs
        )

        if allowed_actions and str(action.get("action") or "") not in allowed_actions:
            # Soft clamp: pick wait if model went outside the skill.
            action = {"action": "wait", "seconds": 0.4, "reason": "clamped to allowed skill actions"}

        target = action.get("target") or ""
        if target and action["action"] in {"click", "drag"}:
            for d in detections:
                if d["class"] == target or target in d["class"]:
                    cx = int(d["bbox"][0] + d["bbox"][2] / 2)
                    cy = int(d["bbox"][1] + d["bbox"][3] / 2)
                    action["x"], action["y"] = cx, cy
                    break
        return action
