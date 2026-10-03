"""HTTP sidecar for the Embodied GUI Agent (localhost:8766)."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent.loop import AgentLoop
from agent.paths import default_dataset_dir
from agent.policy import (
    create_blender_gui_modelfile,
    list_ollama_models,
    prefer_vision_model,
    safe_ollama_name,
)
from worker.loopback import require_loopback_bind, require_loopback_port, request_is_local

_MAX_BODY = 16_000_000  # context posts are small; video/learn carries only URLs/paths
_MAX_GOAL_CHARS = 8192
_MAX_STEPS = 2500
_OLLAMA_CACHE_SEC = 4.0
_ollama_cache: dict[str, Any] = {"t": 0.0, "models": [], "ok": False, "err": ""}
_ollama_lock = threading.Lock()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _clip_text(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) > limit:
        raise ValueError(f"text is longer than {limit} characters")
    return text


def _ollama_models_cached() -> tuple[list[str], bool, str]:
    """Health is polled by the addon with a 1.5–3 s timeout; never block on Ollama longer."""
    with _ollama_lock:
        now = time.time()
        if now - float(_ollama_cache["t"]) < _OLLAMA_CACHE_SEC:
            return list(_ollama_cache["models"]), bool(_ollama_cache["ok"]), str(_ollama_cache["err"])
        try:
            models = list_ollama_models(timeout=1.2)
            _ollama_cache.update({"t": now, "models": models, "ok": True, "err": ""})
        except Exception as exc:
            _ollama_cache.update({"t": now, "models": [], "ok": False, "err": str(exc)})
        return list(_ollama_cache["models"]), bool(_ollama_cache["ok"]), str(_ollama_cache["err"])


class AgentHandler(BaseHTTPRequestHandler):
    loop: AgentLoop
    server_version = "LocalText3D-Agent"
    sys_version = ""

    def log_message(self, format: str, *args) -> None:
        import sys

        line = format % args
        if "/context" in line or "/status" in line:
            return  # high-frequency polls; keep the console readable
        sys.stderr.write("%s - %s\n" % (self.address_string(), line))

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = _json_bytes(payload)
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            return

    def _ensure_local(self) -> bool:
        if request_is_local(self.headers):
            return True
        self._send(403, {"error": "This sidecar only accepts localhost requests"})
        return False

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise ValueError("Bad Content-Length") from exc
        if length < 0 or length > _MAX_BODY:
            raise ValueError(f"Body length {length} out of range")
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def do_GET(self) -> None:
        if not self._ensure_local():
            return
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/health":
                models, ollama_ok, err = _ollama_models_cached()
                preferred = prefer_vision_model(models) if ollama_ok else err
                st = self.loop.status()
                self._send(
                    200,
                    {
                        "ok": True,
                        "service": "gui-agent",
                        "ollama_ok": ollama_ok,
                        "models": models,
                        "preferred_model": preferred,
                        **st,
                    },
                )
                return
            if parsed.path == "/status":
                self._send(200, self.loop.status())
                return
            if parsed.path == "/skills":
                cards = self.loop.memory.semantic_skills(limit=240)
                self._send(200, {"ok": True, "skills": cards})
                return
        except Exception as exc:
            # A closed socket with no response reads as "sidecar offline" in Blender.
            self._send(500, {"ok": False, "error": str(exc)[:300]})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._ensure_local():
            return
        parsed = urlparse(self.path)
        try:
            body = self._read_json()
        except Exception as exc:
            self._send(400, {"error": str(exc)})
            return

        try:
            if parsed.path == "/context":
                # High-frequency: Blender posts scene state; answer apply cmds too.
                self.loop.context.update(body)
                apply = []
                try:
                    apply = self.loop.take_blender_apply()
                except Exception:
                    apply = []
                self._send(200, {"ok": True, "apply": apply})
                return
            if parsed.path == "/plan":
                planner = str(body.get("planner_model") or "")
                if planner:
                    planner = safe_ollama_name(planner, field="planner_model")
                model = str(body.get("model") or "")
                if model:
                    model = safe_ollama_name(model, field="model")
                preview = self.loop.preview_plan(
                    goal=_clip_text(body.get("goal") or "", _MAX_GOAL_CHARS),
                    model=model,
                    planner_model=planner,
                )
                self._send(200, {"ok": True, "plan_preview": preview, **self.loop.status()})
                return
            if parsed.path == "/record/start":
                interval = float(body.get("interval_ms") or 80)
                interval = max(16.0, min(5000.0, interval))
                capture_mode = str(body.get("capture_mode") or "video")
                if capture_mode not in {"video", "screenshots", "screenshot", "png", "images", "image"}:
                    capture_mode = "video"
                goal = _clip_text(body.get("goal") or "", _MAX_GOAL_CHARS)
                self.loop.start_record(
                    interval_ms=interval,
                    capture_mode=capture_mode,
                    goal=goal,
                )
                self._send(200, {"ok": True, **self.loop.status()})
                return
            if parsed.path == "/record/stop":
                goal = _clip_text(body.get("goal") or "", _MAX_GOAL_CHARS)
                self.loop.stop_record(goal=goal)
                self._send(200, {"ok": True, **self.loop.status()})
                return
            if parsed.path == "/agent/start":
                goal = _clip_text(body.get("goal") or "", _MAX_GOAL_CHARS)
                model = safe_ollama_name(str(body.get("model") or "llama3.2-vision"), field="model")
                max_steps = int(body.get("max_steps") or 160)
                max_steps = max(1, min(_MAX_STEPS, max_steps))
                templates = str(body.get("templates_dir") or "")
                run_mode = str(body.get("run_mode") or "learn")
                think_sec = body.get("think_sec")
                settle_sec = body.get("settle_sec")
                planner = str(body.get("planner_model") or "")
                if planner:
                    planner = safe_ollama_name(planner, field="planner_model")
                self.loop.start_agent(
                    goal=goal,
                    model=model,
                    max_steps=max_steps,
                    templates_dir=templates,
                    run_mode=run_mode,
                    think_sec=float(think_sec) if think_sec is not None else None,
                    settle_sec=float(settle_sec) if settle_sec is not None else None,
                    planner_model=planner,
                )
                self._send(200, {"ok": True, **self.loop.status()})
                return
            if parsed.path == "/agent/stop":
                self.loop.stop()
                self._send(200, {"ok": True, **self.loop.status()})
                return
            if parsed.path == "/memory/clear":
                self.loop.clear_memory()
                self._send(200, {"ok": True, **self.loop.status()})
                return
            if parsed.path == "/video/learn":
                url = str(body.get("url") or "")
                path = str(body.get("path") or "")
                goal = _clip_text(body.get("goal") or "", _MAX_GOAL_CHARS)
                model = safe_ollama_name(str(body.get("model") or "llama3.2-vision"), field="model")
                max_minutes = float(body.get("max_minutes") or body.get("learn_minutes") or 6.0)
                self.loop.learn_from_video(
                    url=url,
                    path=path,
                    goal=goal,
                    model=model,
                    max_minutes=max_minutes,
                )
                self._send(200, {"ok": True, **self.loop.status()})
                return
            if parsed.path == "/modelfile/create":
                base = safe_ollama_name(
                    str(body.get("base_model") or "llama3.2-vision"),
                    field="base_model",
                )
                alias = safe_ollama_name(str(body.get("alias") or "blender-gui"), field="alias")
                name = create_blender_gui_modelfile(base, alias=alias)
                self._send(200, {"ok": True, "model": name})
                return
        except ValueError as exc:
            self._send(400, {"error": str(exc)[:300]})
            return
        except Exception as exc:
            self._send(500, {"error": str(exc)[:300], **self.loop.status()})
            return

        self._send(404, {"error": "not found"})


def serve(
    host: str = "127.0.0.1",
    port: int = 8766,
    dataset_dir: Path | None = None,
    templates_dir: Path | None = None,
) -> None:
    host = require_loopback_bind(host)
    port = require_loopback_port(port)
    from agent.window import ensure_dpi_awareness

    ensure_dpi_awareness()  # before pyautogui/mss decide it for us
    loop = AgentLoop(dataset_root=dataset_dir or default_dataset_dir(), templates_dir=templates_dir)
    AgentHandler.loop = loop
    httpd = ThreadingHTTPServer((host, port), AgentHandler)
    httpd.daemon_threads = True
    print(f"GUI Agent sidecar listening on http://{host}:{port}", flush=True)
    print(f"Dataset: {loop.root}", flush=True)
    print("Press ESC while the agent is running to stop the current run (Ctrl+C here quits).", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.stop()
        if loop.recorder.running:
            loop.stop_record()
        httpd.server_close()
