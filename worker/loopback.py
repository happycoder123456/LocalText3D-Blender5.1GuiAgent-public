"""Loopback bind and Host/Origin checks for the localhost HTTP services."""

from __future__ import annotations

from urllib.parse import urlparse

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def is_loopback_host(hostname: str | None) -> bool:
    host = (hostname or "").strip().lower().rstrip(".")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host in LOOPBACK_HOSTS


def host_header_ok(raw: str) -> bool:
    """Accept missing Host (HTTP/1.0) or loopback Host; reject DNS-rebinding names."""
    value = (raw or "").strip()
    if not value:
        return True
    if value.startswith("["):
        end = value.find("]")
        if end == -1:
            return False
        return is_loopback_host(value[1:end])
    host, _sep, port = value.rpartition(":")
    if host and port.isdigit():
        return is_loopback_host(host)
    return is_loopback_host(value)


def origin_header_ok(raw: str) -> bool:
    """Non-browser clients send no Origin. Browser Origins must be loopback HTTP."""
    value = (raw or "").strip()
    if not value:
        return True
    parsed = urlparse(value)
    return parsed.scheme == "http" and is_loopback_host(parsed.hostname)


def request_is_local(headers) -> bool:
    return host_header_ok(headers.get("Host", "")) and origin_header_ok(headers.get("Origin", ""))


def require_loopback_bind(host: str) -> str:
    """Refuse 0.0.0.0 / LAN binds. Documented behavior is 127.0.0.1 only."""
    raw = (host or "").strip() or "127.0.0.1"
    name = raw[1:-1] if raw.startswith("[") and raw.endswith("]") else raw
    if not is_loopback_host(name):
        raise ValueError(
            f"Refusing to bind {host!r}. LocalText3D is localhost-only; use 127.0.0.1."
        )
    return raw


def require_loopback_port(port: int) -> int:
    try:
        value = int(port)
    except (TypeError, ValueError) as exc:
        raise ValueError("port must be an integer") from exc
    if value < 1 or value > 65535:
        raise ValueError("port must be between 1 and 65535")
    return value


def read_http_body(resp, max_bytes: int) -> bytes:
    """Read an HTTP body with a hard cap. Used by local clients talking to Ollama/worker."""
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    headers = getattr(resp, "headers", None)
    raw_len = headers.get("Content-Length") if headers is not None else None
    if raw_len is not None and str(raw_len).strip() != "":
        try:
            length = int(raw_len)
        except (TypeError, ValueError) as exc:
            raise ValueError("Bad Content-Length") from exc
        if length < 0 or length > max_bytes:
            raise ValueError("response is too large")
    data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("response is too large")
    return data
