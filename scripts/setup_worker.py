#!/usr/bin/env python3
"""Create a local venv and install Shap-E for the worker (Windows 11 / Linux)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
REQ = ROOT / "worker" / "requirements-shap-e.txt"

# PyTorch CUDA wheel indexes for Python 3.13. Try the CUDA 12.6 wheels first, then 12.4.
CUDA_INDEXES = (
    "https://download.pytorch.org/whl/cu126",
    "https://download.pytorch.org/whl/cu124",
)
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def python_tag(executable: Path | str) -> str:
    out = subprocess.check_output(
        [str(executable), "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
        text=True,
    )
    return out.strip()


def detect_indexes(force_cpu: bool) -> list[str]:
    if force_cpu:
        return [CPU_INDEX]
    try:
        has_nvidia = subprocess.call(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) == 0
    except OSError:
        has_nvidia = False
    if has_nvidia:
        return list(CUDA_INDEXES)
    # No NVIDIA driver found: the CUDA wheel is ~2.5 GB and would install fine
    # but never use the GPU. Use CPU wheels unless the user forces CUDA later.
    print("nvidia-smi not found: installing CPU PyTorch (re-run with NVIDIA drivers for GPU).", flush=True)
    return [CPU_INDEX]


def install_torch(py: Path, force_cpu: bool) -> None:
    last_error: Exception | None = None
    for index in detect_indexes(force_cpu):
        cmd = [str(py), "-m", "pip", "install", "torch", "--index-url", index]
        print("+", " ".join(cmd), flush=True)
        try:
            subprocess.check_call(cmd)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(f"Torch install failed from {index}, trying the next index...", flush=True)
    raise SystemExit(
        "Could not install PyTorch. On Windows 11 with Python 3.13, install the 64-bit "
        "python.org build and current NVIDIA drivers, then double-click START.bat again."
    ) from last_error


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the Shap-E worker environment")
    parser.add_argument("--cpu", action="store_true", help="Force CPU PyTorch wheels")
    parser.add_argument("--skip-venv", action="store_true", help="Use the current interpreter")
    parser.add_argument("--recreate", action="store_true", help="Delete .venv and create it again")
    args = parser.parse_args()

    print(f"Host Python {sys.version.split()[0]} ({sys.executable})", flush=True)
    if sys.version_info < (3, 10):
        raise SystemExit("Need Python 3.10 or newer. Python 3.13 from python.org is fine.")
    if sys.maxsize <= 2**32:
        raise SystemExit(
            "32-bit Python detected. PyTorch needs 64-bit Python: install the 64-bit "
            "Python 3.13 build from python.org, then run START.bat again."
        )

    py = Path(sys.executable)
    if not args.skip_venv:
        if args.recreate and VENV.exists():
            import shutil

            shutil.rmtree(VENV)
        if VENV.exists() and venv_python().exists():
            existing = python_tag(venv_python())
            host = python_tag(sys.executable)
            if existing != host:
                print(f".venv is Python {existing}, host is {host}. Recreating the venv.", flush=True)
                import shutil

                shutil.rmtree(VENV)
        if not VENV.exists() or not venv_python().exists():
            try:
                run([sys.executable, "-m", "venv", str(VENV)])
            except subprocess.CalledProcessError as exc:
                hint = (
                    "Could not create .venv. On Ubuntu/Debian: sudo apt install python3-venv. "
                    "On Windows 11: install 64-bit Python 3.13 from python.org and check "
                    "Add python.exe to PATH, then run: py -3.13 scripts\\run_all.py"
                )
                raise SystemExit(hint) from exc
        py = venv_python()
        if not py.exists():
            raise SystemExit(f"venv python missing at {py}")

    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel"])
    install_torch(py, force_cpu=args.cpu)
    run([str(py), "-m", "pip", "install", "-r", str(REQ)])

    code = (
        "import torch; from diffusers import ShapEPipeline; "
        "print('python', __import__('sys').version.split()[0]); "
        "print('torch', torch.__version__); "
        "print('cuda', torch.cuda.is_available()); "
        "print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); "
        "print('shap-e-import-ok')"
    )
    run([str(py), "-c", code])
    print(f"Worker Python: {py}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
