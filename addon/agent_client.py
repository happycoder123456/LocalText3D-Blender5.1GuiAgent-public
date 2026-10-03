"""HTTP client for the Embodied GUI Agent sidecar. Stdlib only."""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_URL = "http://127.0.0.1:8766"
SIDECAR_HTTP_PORT = 8766
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v"}
MAX_LOCAL_VIDEO_BYTES = 4_000_000_000


class AgentClientError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError("Refusing HTTP redirect")


_OPENER = urllib.request.build_opener(_NoRedirect)


def require_loopback_http(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "http":
        raise AgentClientError("GUI Agent URL must be http://127.0.0.1 or http://localhost")
    if parsed.username or parsed.password:
        raise AgentClientError("GUI Agent URL must not include credentials")
    host = (parsed.hostname or "").lower()
    if host not in _LOOPBACK_HOSTS:
        raise AgentClientError(
            "GUI Agent URL must stay on this machine (127.0.0.1 / localhost). "
            f"Refusing {host or url}"
        )
    if parsed.port != SIDECAR_HTTP_PORT:
        raise AgentClientError(f"GUI Agent URL must use port {SIDECAR_HTTP_PORT}")


def default_agent_dataset_dir() -> Path:
    """Match agent.paths.default_dataset_dir. Do not take this root from the sidecar."""
    override = os.environ.get("LOCALTEXT3D_AGENT_DATASET")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / "LocalText3D" / "agent_dataset").resolve()


def stage_video_into_dataset(src: str, dataset_root: str | Path) -> str:
    """Copy a user-chosen video into the dataset when it is not already there.

    The sidecar refuses /video/learn paths outside its dataset. Blender's file
    picker is the user's own choice, so the addon stages that file first.
    The HTTP handler must not call this — that would re-open arbitrary paths.
    """
    text = str(src or "").strip()
    if not text:
        raise AgentClientError("Video file path is empty")
    try:
        resolved = Path(text).expanduser().resolve()
    except OSError as exc:
        raise AgentClientError("Video file path is not readable") from exc
    if resolved.suffix.lower() not in _VIDEO_SUFFIXES:
        raise AgentClientError("Local video must be mp4, webm, mkv, avi, mov, or m4v")
    if not resolved.is_file():
        raise AgentClientError("Video file path is not readable")
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise AgentClientError("Video file path is not readable") from exc
    if size < 0 or size > MAX_LOCAL_VIDEO_BYTES:
        raise AgentClientError("Local video is too large (over 4 GB)")
    root = Path(dataset_root).expanduser().resolve()
    try:
        resolved.relative_to(root)
        return str(resolved)
    except ValueError:
        pass
    inbox = root / "videos" / "inbox"
    try:
        inbox.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AgentClientError(f"Cannot create the dataset video folder ({exc})") from exc
    dest = inbox / resolved.name
    try:
        if dest.is_file() and dest.stat().st_size == size:
            existing = dest.resolve()
            existing.relative_to(root)
            if existing != resolved:
                return str(existing)
    except (OSError, ValueError):
        dest = inbox / f"{resolved.stem}-{os.urandom(4).hex()}{resolved.suffix.lower()}"
    try:
        if dest.exists():
            dest = inbox / f"{resolved.stem}-{os.urandom(4).hex()}{resolved.suffix.lower()}"
        shutil.copyfile(resolved, dest)
        staged = dest.resolve()
        staged.relative_to(root)
    except ValueError as exc:
        raise AgentClientError("Staged video escaped the dataset folder") from exc
    except OSError as exc:
        raise AgentClientError(f"Cannot copy the video into the dataset ({exc})") from exc
    return str(staged)


def _url(base: str, path: str) -> str:
    require_loopback_http(base)
    return base.rstrip("/") + path


def _request(
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        require_loopback_http(url)
        with _OPENER.open(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw.decode("utf-8"))
            message = payload.get("error") or payload.get("message") or raw.decode("utf-8", "replace")
        except (json.JSONDecodeError, UnicodeDecodeError):
            message = raw.decode("utf-8", "replace") or str(exc)
        raise AgentClientError(message, status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise AgentClientError(
            f"Cannot reach GUI Agent at {url}. Run scripts\\setup_agent.py then scripts\\start_agent.bat. ({exc.reason})"
        ) from exc
    except (TimeoutError, ConnectionError, OSError) as exc:
        raise AgentClientError(
            f"GUI Agent sidecar is not running at {url}. Start scripts\\start_agent.bat."
        ) from exc
    if not raw:
        return {}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        # Something else answered on this port (or a truncated response).
        raise AgentClientError(f"Unexpected reply from {url} (not JSON): {exc}") from exc
    if not isinstance(payload, dict):
        raise AgentClientError(f"Unexpected reply type from {url}")
    return payload


def health(base_url: str = DEFAULT_URL, timeout: float = 3.0) -> dict[str, Any]:
    return _request("GET", _url(base_url, "/health"), timeout=timeout)


def status(base_url: str = DEFAULT_URL, timeout: float = 3.0) -> dict[str, Any]:
    return _request("GET", _url(base_url, "/status"), timeout=timeout)


def record_start(
    base_url: str,
    interval_ms: float = 80.0,
    capture_mode: str = "video",
    goal: str = "",
    timeout: float = 5.0,
) -> dict[str, Any]:
    return _request(
        "POST",
        _url(base_url, "/record/start"),
        body={"interval_ms": interval_ms, "capture_mode": capture_mode, "goal": goal},
        timeout=timeout,
    )


def record_stop(base_url: str, goal: str = "", timeout: float = 5.0) -> dict[str, Any]:
    return _request(
        "POST",
        _url(base_url, "/record/stop"),
        body={"goal": goal},
        timeout=timeout,
    )


def agent_start(
    base_url: str,
    *,
    goal: str,
    model: str,
    max_steps: int = 160,
    templates_dir: str = "",
    run_mode: str = "learn",
    think_sec: float = 0.2,
    settle_sec: float = 0.12,
    planner_model: str = "auto",
    timeout: float = 60.0,
) -> dict[str, Any]:
    # Planning may call a local text model before the first action; allow time.
    return _request(
        "POST",
        _url(base_url, "/agent/start"),
        body={
            "goal": goal,
            "model": model,
            "max_steps": max_steps,
            "templates_dir": templates_dir,
            "run_mode": run_mode,
            "think_sec": think_sec,
            "settle_sec": settle_sec,
            "planner_model": planner_model,
        },
        timeout=timeout,
    )


def post_context(base_url: str, payload: dict[str, Any], timeout: float = 0.8) -> dict[str, Any]:
    """Send live Blender scene state (mode, selection, mesh stats, operator history)."""
    return _request("POST", _url(base_url, "/context"), body=payload, timeout=timeout)


def plan_preview(
    base_url: str,
    *,
    goal: str,
    model: str = "",
    planner_model: str = "auto",
    timeout: float = 10.0,
) -> dict[str, Any]:
    return _request(
        "POST",
        _url(base_url, "/plan"),
        body={"goal": goal, "model": model, "planner_model": planner_model},
        timeout=timeout,
    )


def skills(base_url: str, timeout: float = 5.0) -> dict[str, Any]:
    return _request("GET", _url(base_url, "/skills"), timeout=timeout)


def agent_stop(base_url: str, timeout: float = 5.0) -> dict[str, Any]:
    return _request("POST", _url(base_url, "/agent/stop"), body={}, timeout=timeout)


def memory_clear(base_url: str, timeout: float = 5.0) -> dict[str, Any]:
    return _request("POST", _url(base_url, "/memory/clear"), body={}, timeout=timeout)


def video_learn(
    base_url: str,
    *,
    url: str = "",
    path: str = "",
    goal: str = "",
    model: str = "llama3.2-vision",
    max_minutes: float = 6.0,
    timeout: float = 15.0,
) -> dict[str, Any]:
    return _request(
        "POST",
        _url(base_url, "/video/learn"),
        body={
            "url": url,
            "path": path,
            "goal": goal,
            "model": model,
            "max_minutes": max_minutes,
        },
        timeout=timeout,
    )


def create_modelfile(
    base_url: str,
    *,
    base_model: str,
    alias: str = "blender-gui",
    timeout: float = 300.0,
) -> dict[str, Any]:
    return _request(
        "POST",
        _url(base_url, "/modelfile/create"),
        body={"base_model": base_model, "alias": alias},
        timeout=timeout,
    )
