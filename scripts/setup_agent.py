#!/usr/bin/env python3
"""Create .venv-agent and install Embodied GUI Agent deps (no PyTorch)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv-agent"
REQ = ROOT / "agent" / "requirements.txt"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the GUI Agent sidecar environment")
    parser.add_argument("--recreate", action="store_true", help="Delete .venv-agent and recreate")
    parser.add_argument("--skip-venv", action="store_true", help="Use current interpreter")
    args = parser.parse_args()

    print(f"Host Python {sys.version.split()[0]} ({sys.executable})", flush=True)

    if args.skip_venv:
        py = Path(sys.executable)
    else:
        if args.recreate and VENV.exists():
            import shutil

            shutil.rmtree(VENV)
        # A half-created venv (folder present, no python.exe) must be rebuilt.
        if not VENV.exists() or not venv_python().exists():
            if VENV.exists():
                import shutil

                shutil.rmtree(VENV)
            run([sys.executable, "-m", "venv", str(VENV)])
        py = venv_python()
        if not py.exists():
            raise SystemExit(f"venv python missing at {py}")
        run([str(py), "-m", "pip", "install", "--upgrade", "pip"])

    if not REQ.is_file():
        raise SystemExit(f"Missing {REQ}")
    run([str(py), "-m", "pip", "install", "-r", str(REQ)])

    print(flush=True)
    print("GUI Agent env ready.", flush=True)
    print(f"Start sidecar: {py} -m agent serve", flush=True)
    print("Keep Ollama running with your llava / vision model.", flush=True)
    print("Do not run TRELLIS generate at the same time (VRAM).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
