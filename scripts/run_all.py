#!/usr/bin/env python3
"""Install Shap-E (and optionally TRELLIS), build the addon zip, and start the worker."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
TRELLIS_REPO = ROOT / "vendor" / "TRELLIS"


def shap_e_venv_python() -> Path:
    if os.name == "nt":
        return ROOT / ".venv" / "Scripts" / "python.exe"
    return ROOT / ".venv" / "bin" / "python"


def trellis_venv_python() -> Path:
    if os.name == "nt":
        return ROOT / ".venv-trellis" / "Scripts" / "python.exe"
    return ROOT / ".venv-trellis" / "bin" / "python"


def worker_python() -> Path:
    """Prefer the TRELLIS venv when present so both engines are available."""
    trellis = trellis_venv_python()
    if trellis.exists():
        return trellis
    shap = shap_e_venv_python()
    if shap.exists():
        return shap
    return Path(sys.executable)


def _ensure_ninja_on_path(py: Path) -> None:
    """PyTorch JIT extensions look up ninja via PATH, not the Python package."""
    import shutil

    if shutil.which("ninja"):
        return
    scripts = py.resolve().parent
    ninja_bin = scripts / ("ninja.exe" if os.name == "nt" else "ninja")
    if not ninja_bin.is_file():
        return
    path = os.environ.get("PATH", "")
    parts = [p for p in path.split(os.pathsep) if p]
    scripts_str = str(scripts)
    if scripts_str not in parts:
        parts.insert(0, scripts_str)
        os.environ["PATH"] = os.pathsep.join(parts)


def configure_trellis_env() -> None:
    repo = TRELLIS_REPO
    if not repo.is_dir():
        return
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
    _ensure_ninja_on_path(worker_python())
    # Mirror worker CUDA_HOME discovery so start scripts match runtime behavior.
    # The worker package needs ROOT on sys.path (this script lives in scripts/).
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from worker.engines.trellis import _ensure_cuda_home

        _ensure_cuda_home()
    except Exception as exc:  # numpy/torch may be missing in the host interpreter
        print(f"(CUDA_HOME discovery skipped: {exc})", flush=True)
    trellis_root = str(repo.resolve())

    if trellis_root not in sys.path:
        sys.path.insert(0, trellis_root)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    if trellis_root not in parts:
        parts.insert(0, trellis_root)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup + build addon + start worker")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Loopback only (127.0.0.1, localhost, or ::1). 0.0.0.0 and LAN addresses are refused.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Local worker port. The Blender addon only connects to 8765 on this machine.",
    )
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument(
        "--with-trellis",
        action="store_true",
        help="Install TRELLIS into .venv-trellis before starting (Windows, several GB)",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    print(f"Using {sys.executable} ({sys.version.split()[0]})", flush=True)

    if args.with_trellis:
        if os.name != "nt":
            raise SystemExit("--with-trellis is the Windows native installer. On Linux use microsoft/TRELLIS setup.sh.")
        trellis_setup = [sys.executable, str(SCRIPTS / "setup_trellis.py")]
        if args.recreate:
            trellis_setup.append("--recreate")
        run(trellis_setup)

    if not args.skip_setup and not trellis_venv_python().exists():
        setup = [sys.executable, str(SCRIPTS / "setup_worker.py")]
        if args.cpu:
            setup.append("--cpu")
        if args.recreate:
            setup.append("--recreate")
        run(setup)

    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import build_addon  # noqa: E402  (same folder as this script)

    zip_path = build_addon.build()
    print(f"Addon zip: {zip_path}")
    print("In Blender 5.1: Get Extensions -> Install from Disk -> that zip")
    print("Then N-panel -> Text to 3D -> Engine: Shap-E or TRELLIS -> Generate")

    configure_trellis_env()
    py = worker_python()
    print(f"Worker Python: {py}", flush=True)
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from worker.loopback import require_loopback_bind, require_loopback_port

        host = require_loopback_bind(args.host)
        port = require_loopback_port(args.port)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    serve = [str(py), "-m", "worker", "serve", "--host", host, "--port", str(port)]
    if args.mock:
        serve.append("--mock")
    print(f"Starting worker on http://{host}:{port}", flush=True)
    # os.execv is unreliable on Windows (window can close immediately after setup).
    if os.name == "nt":
        try:
            return subprocess.call(serve)
        except KeyboardInterrupt:
            # Ctrl+C in the worker window also reaches this parent; that is a
            # normal shutdown, not an error to pause on.
            return 0
    os.execv(str(py), serve)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
