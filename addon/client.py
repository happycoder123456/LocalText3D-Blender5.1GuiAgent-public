"""HTTP client for the local text-to-3D worker. Stdlib only."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse
from pathlib import Path


class WorkerError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError("Refusing HTTP redirect")


_OPENER = urllib.request.build_opener(_NoRedirect)
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
# The worker always listens here. Any other local port could be a different process.
WORKER_HTTP_PORT = 8765
_MAX_JSON_BYTES = 8_000_000
MAX_GLB_DOWNLOAD_BYTES = 128_000_000


def require_loopback_http(url: str, *, kind: str = "Worker", port: int = WORKER_HTTP_PORT) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "http":
        raise WorkerError(f"{kind} URL must be http://127.0.0.1 or http://localhost")
    if parsed.username or parsed.password:
        raise WorkerError(f"{kind} URL must not include credentials")
    host = (parsed.hostname or "").lower()
    if host not in _LOOPBACK_HOSTS:
        raise WorkerError(
            f"{kind} URL must stay on this machine (127.0.0.1 / localhost). "
            f"Refusing {host or url}"
        )
    if parsed.port != int(port):
        raise WorkerError(f"{kind} URL must use port {port}")


def default_worker_output_dir() -> Path:
    """Match worker.cli.default_output_dir. The addon must not trust a server-supplied root."""
    override = os.environ.get("LOCALTEXT3D_OUTPUT_DIR")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "localtext3d" / "outputs"
    return Path.home() / ".cache" / "localtext3d" / "outputs"


def glb_under_worker_output(raw: str) -> str | None:
    """Accept a GLB only when it resolves to a file inside the worker output directory.

    Job JSON `output_path` is a hint. A path outside that directory (or a symlink
    that escapes it) is ignored so the addon downloads via /jobs/{id}/file instead.
    """
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        resolved = Path(text).expanduser().resolve()
        root = default_worker_output_dir().expanduser().resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    if resolved.suffix.lower() != ".glb" or not resolved.is_file():
        return None
    return str(resolved)


def _read_limited(resp, max_bytes: int) -> bytes:
    header = ""
    try:
        header = resp.headers.get("Content-Length") or ""
    except Exception:
        header = ""
    if header:
        try:
            length = int(header)
        except ValueError as exc:
            raise WorkerError("Worker returned a bad Content-Length") from exc
        if length < 0 or length > max_bytes:
            raise WorkerError("Worker response is too large")
    data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise WorkerError("Worker response is too large")
    return data


def _url(base: str, path: str) -> str:
    require_loopback_http(base)
    return base.rstrip("/") + path


def _request(
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float = 10.0,
    accept: str = "application/json",
) -> tuple[int, bytes, str]:
    require_loopback_http(url)
    data = None
    headers = {"Accept": accept}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            return resp.status, _read_limited(resp, _MAX_JSON_BYTES), resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        raw = _read_limited(exc, min(_MAX_JSON_BYTES, 1_000_000))
        try:
            payload = json.loads(raw.decode("utf-8"))
            message = payload.get("error") or payload.get("message") or raw.decode("utf-8", "replace")
        except (json.JSONDecodeError, UnicodeDecodeError):
            message = raw.decode("utf-8", "replace") or str(exc)
        raise WorkerError(message, status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise WorkerError(
            f"Cannot reach worker at {url}. Double-click START.bat and leave that window open. ({exc.reason})"
        ) from exc
    except (TimeoutError, ConnectionError, OSError) as exc:
        # Includes http.client.RemoteDisconnected when the worker window was closed.
        raise WorkerError(
            f"Worker is not running at {url}. Double-click START.bat and leave that window open."
        ) from exc


def _decode_json(raw: bytes) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise WorkerError(f"Worker returned a non-JSON reply: {exc}") from exc
    if not isinstance(payload, dict):
        raise WorkerError("Worker returned an unexpected reply type")
    return payload


def health(base_url: str, timeout: float = 3.0) -> dict[str, Any]:
    _status, raw, _ct = _request("GET", _url(base_url, "/health"), timeout=timeout)
    return _decode_json(raw)


def start_generate(base_url: str, payload: dict[str, Any], timeout: float = 15.0) -> dict[str, Any]:
    _status, raw, _ct = _request(
        "POST",
        _url(base_url, "/generate"),
        body=payload,
        timeout=timeout,
    )
    return _decode_json(raw)


def job_status(base_url: str, job_id: str, timeout: float = 5.0) -> dict[str, Any]:
    _status, raw, _ct = _request("GET", _url(base_url, f"/jobs/{job_id}"), timeout=timeout)
    return _decode_json(raw)


def list_jobs(base_url: str, timeout: float = 5.0, limit: int = 24) -> list[dict[str, Any]]:
    _status, raw, _ct = _request(
        "GET",
        _url(base_url, f"/jobs?limit={int(limit)}"),
        timeout=timeout,
    )
    payload = _decode_json(raw)
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    return jobs if isinstance(jobs, list) else []


def cancel_job(base_url: str, job_id: str, timeout: float = 5.0) -> dict[str, Any]:
    """Ask the worker to stop the running job at its next stage boundary."""
    _status, raw, _ct = _request(
        "POST",
        _url(base_url, f"/jobs/{job_id}/cancel"),
        body={},
        timeout=timeout,
    )
    return _decode_json(raw)


def download_job_file(base_url: str, job_id: str, dest_path: str, timeout: float = 120.0) -> str:
    url = _url(base_url, f"/jobs/{job_id}/file")
    req = urllib.request.Request(
        url,
        headers={"Accept": "model/gltf-binary,application/octet-stream,*/*"},
        method="GET",
    )
    written = 0
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            header = resp.headers.get("Content-Length") or ""
            if header:
                try:
                    length = int(header)
                except ValueError as exc:
                    raise WorkerError("Worker returned a bad Content-Length") from exc
                if length < 0 or length > MAX_GLB_DOWNLOAD_BYTES:
                    raise WorkerError("GLB is too large to download")
            with open(dest_path, "wb") as handle:
                while True:
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_GLB_DOWNLOAD_BYTES:
                        raise WorkerError("GLB is too large to download")
                    handle.write(chunk)
    except urllib.error.HTTPError as exc:
        raw = _read_limited(exc, min(_MAX_JSON_BYTES, 1_000_000))
        try:
            payload = json.loads(raw.decode("utf-8"))
            message = payload.get("error") or payload.get("message") or raw.decode("utf-8", "replace")
        except (json.JSONDecodeError, UnicodeDecodeError):
            message = raw.decode("utf-8", "replace") or str(exc)
        raise WorkerError(message, status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise WorkerError(
            f"Cannot reach worker at {url}. Double-click START.bat and leave that window open. ({exc.reason})"
        ) from exc
    except WorkerError:
        try:
            Path(dest_path).unlink(missing_ok=True)
        except OSError:
            pass
        raise
    except (TimeoutError, ConnectionError, OSError) as exc:
        raise WorkerError(
            f"Worker is not running at {url}. Double-click START.bat and leave that window open."
        ) from exc
    if written == 0:
        try:
            Path(dest_path).unlink(missing_ok=True)
        except OSError:
            pass
        raise WorkerError("Worker returned an empty GLB")
    return dest_path
