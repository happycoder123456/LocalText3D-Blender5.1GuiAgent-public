"""Ollama HTTP client for Geometry Nodes generation. Stdlib only."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

OLLAMA_URL = "http://127.0.0.1:11434"
TIMEOUT = 120.0
_OLLAMA_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


class OllamaError(Exception):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError("Refusing HTTP redirect")


_OPENER = urllib.request.build_opener(_NoRedirect)


def _safe_model(name: str) -> str:
    text = (name or "").strip()
    if not text or not _OLLAMA_NAME_RE.fullmatch(text) or ".." in text:
        raise OllamaError("Invalid Ollama model name")
    return text


def _request(method: str, path: str, body: dict | None = None, timeout: float = TIMEOUT) -> dict:
    url = OLLAMA_URL.rstrip("/") + path
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw.decode("utf-8"))
            message = payload.get("error") or str(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            message = raw.decode("utf-8", "replace") or str(exc)
        raise OllamaError(message) from exc
    except urllib.error.URLError as exc:
        raise OllamaError(
            "Cannot reach Ollama at 127.0.0.1:11434. Start Ollama, then click refresh."
        ) from exc
    except (TimeoutError, ConnectionError, OSError) as exc:
        raise OllamaError(
            "Ollama is not running. Start Ollama, then click refresh."
        ) from exc
    if not raw:
        return {}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OllamaError(f"Ollama returned a non-JSON reply: {exc}") from exc
    if not isinstance(payload, dict):
        raise OllamaError("Ollama returned an unexpected reply type")
    return payload


def list_models(timeout: float = 5.0) -> list[str]:
    payload = _request("GET", "/api/tags", timeout=timeout)
    models = payload.get("models") or []
    names: list[str] = []
    for item in models:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            if name:
                names.append(name)
    return names


def chat(model: str, messages: list[dict], seed: int = 0, timeout: float = TIMEOUT) -> str:
    body: dict = {
        "model": _safe_model(model),
        "messages": messages,
        "stream": False,
        "format": "json",
    }
    if seed:
        body["options"] = {"seed": int(seed)}
    payload = _request("POST", "/api/chat", body=body, timeout=timeout)
    message = payload.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    if not content:
        raise OllamaError("Ollama returned an empty response")
    return str(content)
