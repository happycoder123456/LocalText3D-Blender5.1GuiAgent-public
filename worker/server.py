from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from worker.device import device_info
from worker.engines.trellis import textured_export_available
from worker.jobs import JobBusy, JobStore
from worker.loopback import require_loopback_bind, require_loopback_port, request_is_local
from worker.manager import EngineManager
from worker.models import request_from_payload

# 4 images + mesh_glb are base64; keep headroom above decoded 3 MB / 8 MB caps.
_MAX_BODY = 32_000_000
_JOB_ID_MAX = 64
# Generated TRELLIS GLBs are typically tens of MB; this is a disaster cap.
MAX_GLB_DOWNLOAD_BYTES = 128_000_000


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _job_id_ok(raw: str) -> bool:
    return bool(raw) and len(raw) <= _JOB_ID_MAX and all(ch.isalnum() or ch in "-_" for ch in raw)


class WorkerHandler(BaseHTTPRequestHandler):
    store: JobStore
    manager: EngineManager
    server_version = "LocalText3D"
    sys_version = ""

    def log_message(self, format: str, *args) -> None:
        sys_stderr = self.sys_stderr()
        sys_stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def sys_stderr(self):
        import sys

        return sys.stderr

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
        self._send(403, {"error": "This worker only accepts localhost requests"})
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

    def _glb_on_disk(self, job) -> Path | None:
        path = Path(job.output_path)
        try:
            resolved = path.resolve()
            resolved.relative_to(self.store.output_dir.resolve())
        except (OSError, ValueError):
            return None
        if not resolved.is_file():
            return None
        return resolved

    def do_GET(self) -> None:
        if not self._ensure_local():
            return
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        if parsed.path == "/health":
            info = device_info()
            info.update(
                {
                    "ok": True,
                    "available_engines": self.manager.available_engines(),
                    "loaded_engine": self.manager.loaded_engine(),
                    "loaded_variant": self.manager.loaded_variant(),
                    "mock": self.manager.mock,
                    "trellis_textured_export": textured_export_available(),
                }
            )
            self._send(200, info)
            return
        if parsed.path == "/jobs":
            query = parse_qs(parsed.query)
            try:
                limit = int((query.get("limit") or ["24"])[0])
            except ValueError:
                limit = 24
            self._send(200, {"jobs": self.store.list_recent(limit=limit)})
            return
        if len(parts) == 2 and parts[0] == "jobs":
            if not _job_id_ok(parts[1]):
                self._send(404, {"error": "Unknown job id"})
                return
            job = self.store.get(parts[1])
            if job is None:
                self._send(404, {"error": "Unknown job id"})
                return
            self._send(200, job.to_dict())
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "file":
            if not _job_id_ok(parts[1]):
                self._send(404, {"error": "GLB is not ready"})
                return
            job = self.store.get(parts[1])
            if job is None or job.status != "done" or not job.output_path:
                self._send(404, {"error": "GLB is not ready"})
                return
            resolved = self._glb_on_disk(job)
            if resolved is None:
                self._send(404, {"error": "GLB file is missing on disk"})
                return
            try:
                size = resolved.stat().st_size
            except OSError:
                self._send(404, {"error": "GLB file is missing on disk"})
                return
            if size > MAX_GLB_DOWNLOAD_BYTES:
                self._send(413, {"error": "GLB is too large to download"})
                return
            filename = f"{job.job_id}.glb"
            try:
                self.send_response(200)
                self.send_header("Content-Type", "model/gltf-binary")
                self.send_header("Content-Length", str(size))
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.end_headers()
                with resolved.open("rb") as handle:
                    while True:
                        chunk = handle.read(256 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                # Blender closed the socket mid-download; nothing to report.
                return
            return
        self._send(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if not self._ensure_local():
            return
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "cancel":
            if not _job_id_ok(parts[1]):
                self._send(404, {"error": "No running job with that id"})
                return
            if self.store.cancel(parts[1]):
                self._send(200, {"ok": True, "job_id": parts[1], "status": "cancelling"})
            else:
                self._send(404, {"error": "No running job with that id"})
            return
        if parsed.path != "/generate":
            self._send(404, {"error": "Not found"})
            return
        try:
            payload = self._read_json()
            request = request_from_payload(payload)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
            return
        try:
            job = self.store.submit(request)
        except JobBusy as exc:
            self._send(409, {"error": str(exc)})
            return
        self._send(202, {"job_id": job.job_id, "status": job.status})


def make_server(
    host: str,
    port: int,
    store: JobStore,
    manager: EngineManager,
) -> ThreadingHTTPServer:
    host = require_loopback_bind(host)
    port = require_loopback_port(port)
    handler = type(
        "BoundWorkerHandler",
        (WorkerHandler,),
        {"store": store, "manager": manager},
    )
    return ThreadingHTTPServer((host, port), handler)


def serve(host: str, port: int, store: JobStore, manager: EngineManager) -> ThreadingHTTPServer:
    httpd = make_server(host, port, store, manager)
    print(f"Local Text to 3D worker listening on http://{host}:{port}")
    print(f"Available engines: {', '.join(manager.available_engines()) or 'none'}")
    if manager.mock:
        print("Mock mode: every engine writes a placeholder cube GLB.")
    httpd.serve_forever()
    return httpd
