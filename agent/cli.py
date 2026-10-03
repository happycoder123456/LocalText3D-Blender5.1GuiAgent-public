"""CLI for the Embodied GUI Agent sidecar."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent.paths import default_dataset_dir, default_templates_dir
from agent.server import serve
from worker.loopback import require_loopback_bind, require_loopback_port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent",
        description="Local Embodied GUI Agent for Blender (separate from TRELLIS/Shap-E)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    serve_p = sub.add_parser("serve", help="Start the localhost HTTP sidecar")
    serve_p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Loopback only (127.0.0.1, localhost, or ::1). 0.0.0.0 and LAN addresses are refused.",
    )
    serve_p.add_argument(
        "--port",
        type=int,
        default=8766,
        help="Local port. The Blender addon only connects to 8766 on this machine.",
    )
    serve_p.add_argument("--dataset-dir", type=Path, default=None)
    serve_p.add_argument("--templates-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        try:
            host = require_loopback_bind(args.host)
            port = require_loopback_port(args.port)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        serve(
            host=host,
            port=port,
            dataset_dir=args.dataset_dir or default_dataset_dir(),
            templates_dir=args.templates_dir or default_templates_dir(),
        )
        return 0
    return 1
