from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from worker.jobs import JobStore
from worker.loopback import require_loopback_bind, require_loopback_port
from worker.manager import EngineManager
from worker.models import GenerateRequest
from worker.server import serve


def default_output_dir() -> Path:
    override = os.environ.get("LOCALTEXT3D_OUTPUT_DIR")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "localtext3d" / "outputs"
    return Path.home() / ".cache" / "localtext3d" / "outputs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m worker",
        description="Local text-to-3D worker for the Blender addon",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve_p = sub.add_parser("serve", help="Start the localhost HTTP worker")
    serve_p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Loopback only (127.0.0.1, localhost, or ::1). 0.0.0.0 and LAN addresses are refused.",
    )
    serve_p.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Local port. The Blender addon only connects to 8765 on this machine.",
    )
    serve_p.add_argument("--mock", action="store_true", help="Use placeholder cubes instead of real models")
    serve_p.add_argument("--output-dir", type=Path, default=None)

    gen = sub.add_parser("generate", help="Generate a GLB from the command line")
    gen.add_argument("prompt")
    gen.add_argument("--engine", choices=("trellis", "shap_e", "mock"), default="trellis")
    gen.add_argument("--seed", type=int, default=1)
    gen.add_argument("--trellis-variant", default="text-large", choices=("text-base", "text-large", "text-xlarge"))
    gen.add_argument("--output", type=Path, default=Path("generated.glb"))
    gen.add_argument("--shap-e-steps", type=int, default=64)
    gen.add_argument("--shap-e-guidance", type=float, default=15.0)
    gen.add_argument("--mock", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mock = bool(args.mock or os.environ.get("LOCALTEXT3D_MOCK"))
    manager = EngineManager(mock=mock)

    if args.command == "serve":
        try:
            host = require_loopback_bind(args.host)
            port = require_loopback_port(args.port)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        output_dir = args.output_dir or default_output_dir()
        store = JobStore(manager, output_dir)
        serve(host, port, store, manager)
        return 0

    request = GenerateRequest(
        engine=args.engine,
        prompt=args.prompt,
        seed=args.seed,
        trellis_variant=args.trellis_variant,
        shap_e_steps=getattr(args, "shap_e_steps", 64),
        shap_e_guidance=getattr(args, "shap_e_guidance", 15.0),
        output_path=args.output,
    )

    def progress(status: str, message: str) -> None:
        print(f"[{status}] {message}")

    dest = args.output.expanduser().resolve()
    path = manager.generate(request, dest, progress=progress)
    print(f"Wrote {path}")
    return 0


def wait_for_job(store: JobStore, job_id: str, timeout: float = 600.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = store.get(job_id)
        if job and job.status in {"done", "failed"}:
            return
        time.sleep(0.2)
    raise TimeoutError("Timed out waiting for the job")
