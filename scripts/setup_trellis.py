#!/usr/bin/env python3
"""Install microsoft/TRELLIS into .venv-trellis (Python 3.10) for Windows + CUDA.

Uses pinned PyTorch CUDA wheels so no WSL/conda or local CUDA Toolkit compile is
required for mesh generation. Also installs Shap-E into the same venv so one
worker can serve both engines.

Unhashed community wheels and third-party extra indexes are not installed.
Textured GLB export needs those wheels; mesh-only generation does not.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
TRELLIS_REPO = VENDOR / "TRELLIS"
VENV = ROOT / ".venv-trellis"
SHAP_E_REQ = ROOT / "worker" / "requirements-shap-e.txt"
TRELLIS_GIT = "https://github.com/microsoft/TRELLIS.git"
# Pin so setup does not silently follow microsoft/TRELLIS main.
TRELLIS_REF = "442aa1e1afb9014e80681d3bf604e8d728a86ee7"

TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
BASIC_PACKAGES = [
    "pillow",
    "imageio",
    "imageio-ffmpeg",
    "tqdm",
    "easydict",
    "opencv-python-headless",
    "scipy",
    "ninja",
    "rembg",
    "onnxruntime",
    "trimesh",
    "xatlas",
    "pyvista",
    "pymeshfix",
    "igraph",
    "transformers",
    "open3d",
]
UTILS3D = "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# NVIDIA's own find-links page for this torch build. Not a community extra-index.
# The wheel filename on that page is not hash-pinned here: vendoring the large
# kaolin wheel is a separate packaging step. Mesh generation needs kaolin/spconv.
KAOLIN_FIND = "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html"
# Official PyTorch index build that matches torch==2.5.1+cu124.
TORCHVISION_SPEC = "torchvision==0.20.1"
XFORMERS_SPEC = "xformers==0.0.28.post3"


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, env=env)


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def _uv_python310() -> str | None:
    """Locate a uv-managed CPython 3.10 (any patch level) on Windows."""
    uv_root = Path.home() / "AppData" / "Roaming" / "uv" / "python"
    if uv_root.is_dir():
        hits = sorted(uv_root.glob("cpython-3.10*-windows-x86_64-none/python.exe"), reverse=True)
        if hits:
            return str(hits[0])
    try:
        out = subprocess.check_output(["uv", "python", "find", "3.10"], text=True, stderr=subprocess.DEVNULL)
        found = out.strip().splitlines()[-1].strip() if out.strip() else ""
        if found and Path(found).exists():
            return found
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        pass
    return None


def find_python310() -> str:
    candidates: list[list[str]] = [
        ["py", "-3.10"],
        ["python3.10"],
        ["python"],
    ]
    uv_py = _uv_python310()
    if uv_py:
        return uv_py

    for parts in candidates:
        try:
            out = subprocess.check_output(
                parts + ["-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}'); print(sys.executable)"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            lines = [line.strip() for line in out.splitlines() if line.strip()]
            if lines and lines[0] == "3.10":
                return lines[1]
        except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
            continue

    # Last resort: ask uv to install 3.10.
    try:
        run(["uv", "python", "install", "3.10"])
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(
            "Need Python 3.10 for TRELLIS Windows wheels. Install with: uv python install 3.10"
        ) from exc
    uv_py = _uv_python310()
    if uv_py:
        return uv_py
    raise SystemExit("Python 3.10 install finished but the interpreter was not found.")


def ensure_repo() -> None:
    VENDOR.mkdir(parents=True, exist_ok=True)
    if (TRELLIS_REPO / "trellis" / "pipelines").is_dir():
        _checkout_trellis_ref()
        print(f"TRELLIS repo already present at {TRELLIS_REPO}", flush=True)
        return
    if TRELLIS_REPO.exists():
        shutil.rmtree(TRELLIS_REPO)
    run(["git", "clone", TRELLIS_GIT, str(TRELLIS_REPO)])
    _checkout_trellis_ref(required=True)


def _checkout_trellis_ref(*, required: bool = False) -> None:
    git_dir = TRELLIS_REPO / ".git"
    if not git_dir.exists() and not (TRELLIS_REPO / ".git").is_file():
        if required:
            raise SystemExit("TRELLIS clone is missing .git metadata; cannot pin the commit.")
        return
    try:
        current = subprocess.check_output(
            ["git", "-C", str(TRELLIS_REPO), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        if required:
            raise SystemExit("Could not read TRELLIS HEAD") from exc
        return
    if current == TRELLIS_REF:
        return
    print(f"Checking out pinned TRELLIS {TRELLIS_REF[:12]}…", flush=True)
    try:
        run(["git", "-C", str(TRELLIS_REPO), "fetch", "--depth", "1", "origin", TRELLIS_REF])
    except subprocess.CalledProcessError:
        # Full clones already have history; fetch --depth can fail offline.
        pass
    try:
        run(["git", "-C", str(TRELLIS_REPO), "checkout", TRELLIS_REF])
        run(["git", "-C", str(TRELLIS_REPO), "submodule", "update", "--init", "--recursive"])
    except subprocess.CalledProcessError as exc:
        if required:
            raise SystemExit(f"Could not check out pinned TRELLIS commit {TRELLIS_REF}") from exc
        print("Could not move the existing TRELLIS clone to the pinned commit; using what is on disk.", flush=True)


def ensure_venv(py310: str, recreate: bool) -> Path:
    if recreate and VENV.exists():
        shutil.rmtree(VENV)
    py = venv_python()
    if not py.exists():
        run([py310, "-m", "venv", str(VENV)])
    if not py.exists():
        raise SystemExit(f"venv python missing at {py}")
    tag = subprocess.check_output(
        [str(py), "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
        text=True,
    ).strip()
    if tag != "3.10":
        raise SystemExit(f".venv-trellis is Python {tag}; need 3.10. Re-run with --recreate.")
    return py


def pip_install(py: Path, *args: str) -> None:
    run([str(py), "-m", "pip", "install", *args])


def _skip_unhashed_texture_wheels() -> None:
    print(
        "Supply chain: not installing unhashed community wheels "
        "(sdbds nvdiffrast, bdashore3 flash-attn) or the miropsota torch extra index. "
        "Not cloning unpinned rasterizer repos. "
        "Mesh generation uses pinned torch/xformers from download.pytorch.org "
        "(ATTN_BACKEND=xformers). Textured GLB export stays off until those "
        "wheels are vendored with sha256.",
        flush=True,
    )


def install_stack(py: Path) -> None:
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])
    # Idempotent: already-installed packages are reused on resume.
    # torchvision is pinned to the release that matches torch 2.5.1.
    pip_install(py, "torch==2.5.1", TORCHVISION_SPEC, "--index-url", TORCH_INDEX)
    pip_install(py, XFORMERS_SPEC, "--index-url", TORCH_INDEX)
    pip_install(py, *BASIC_PACKAGES)
    pip_install(py, UTILS3D)
    _skip_unhashed_texture_wheels()

    try:
        pip_install(py, "kaolin", "-f", KAOLIN_FIND)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Failed to install kaolin from {KAOLIN_FIND}. "
            "Need torch 2.5.1 + cu124 on Python 3.10."
        ) from exc

    # spconv-cu120 works with CUDA 12.x runtimes from PyTorch wheels.
    # PyPI name is pinned to the cu120 build; the patch version is not hashed.
    pip_install(py, "spconv-cu120")
    if SHAP_E_REQ.exists():
        pip_install(py, "-r", str(SHAP_E_REQ))


def configure_env(py: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    # xformers is the pinned official wheel. flash_attn was a community wheel and is not installed.
    # Honor an explicit ATTN_BACKEND=flash_attn only when the user set it themselves.
    backend = os.environ.get("ATTN_BACKEND", "").strip()
    if backend not in {"flash_attn", "xformers"}:
        backend = "xformers"
    env["ATTN_BACKEND"] = backend
    env["SPCONV_ALGO"] = "native"
    env["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    trellis_root = str(TRELLIS_REPO.resolve())
    existing = env.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    if trellis_root not in parts:
        parts.insert(0, trellis_root)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    scripts = (py or venv_python()).resolve().parent
    ninja_bin = scripts / ("ninja.exe" if os.name == "nt" else "ninja")
    path_parts = [p for p in env.get("PATH", "").split(os.pathsep) if p]
    if ninja_bin.is_file():
        scripts_str = str(scripts)
        if scripts_str not in path_parts:
            path_parts.insert(0, scripts_str)
    # Prefer an existing CUDA Toolkit if present (nvdiffrast JIT needs nvcc).
    cuda_candidates = []
    for key in ("CUDA_HOME", "CUDA_PATH"):
        if env.get(key):
            cuda_candidates.append(Path(env[key]))
    toolkit_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if toolkit_root.is_dir():
        cuda_candidates.extend(
            sorted(
                (p for p in toolkit_root.iterdir() if p.is_dir()),
                key=lambda p: p.name,
                reverse=True,
            )
        )
    for root in cuda_candidates:
        nvcc = root / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
        if nvcc.is_file():
            env.setdefault("CUDA_HOME", str(root))
            env.setdefault("CUDA_PATH", str(root))
            bin_dir = str(root / "bin")
            if bin_dir not in path_parts:
                path_parts.insert(0, bin_dir)
            break
    env["PATH"] = os.pathsep.join(path_parts)
    return env


def verify(py: Path) -> None:
    env = configure_env(py)
    code = (
        "import shutil, os; "
        "from pathlib import Path; "
        "import torch; "
        "from trellis.pipelines import TrellisTextTo3DPipeline; "
        "g={}; "
        "exec('try:\\n import nvdiffrast.torch\\n "
        "from diff_gaussian_rasterization import GaussianRasterizer\\n pkgs=True\\n"
        "except Exception: pkgs=False', g); "
        "cuda_home=os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH') or ''; "
        "nvcc=shutil.which('nvcc'); "
        "toolkit=bool(cuda_home and Path(cuda_home,'bin','nvcc.exe' if os.name=='nt' else 'nvcc').is_file()); "
        "print('python', __import__('sys').version.split()[0]); "
        "print('torch', torch.__version__); "
        "print('cuda', torch.cuda.is_available()); "
        "print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); "
        "print('packages-ok', g.get('pkgs', False)); "
        "print('cuda-toolkit', toolkit); "
        "print('nvcc', nvcc or 'none'); "
        "print('textured-export', bool(g.get('pkgs') and toolkit)); "
        "print('trellis-import-ok')"
    )
    run([str(py), "-c", code], env=env)
    if not env.get("CUDA_HOME"):
        print(
            "Note: no CUDA Toolkit found. Mesh generation still works. "
            "For baked textures later, install CUDA Toolkit 12.4 (matches torch cu124) "
            "and Visual Studio Build Tools (C++).",
            flush=True,
        )


def main() -> int:
    if os.name != "nt":
        raise SystemExit(
            "This Windows wheel installer is for Windows only. "
            "On Linux use the official microsoft/TRELLIS setup.sh."
        )

    parser = argparse.ArgumentParser(description="Install TRELLIS for LocalText3D on Windows")
    parser.add_argument("--recreate", action="store_true", help="Delete .venv-trellis and recreate")
    parser.add_argument("--skip-clone", action="store_true", help="Reuse existing vendor/TRELLIS")
    args = parser.parse_args()

    print("Installing TRELLIS (Python 3.10 + Windows CUDA wheels). This downloads several GB.", flush=True)
    if not shutil.which("git"):
        raise SystemExit(
            "git is required to fetch TRELLIS and its extensions. "
            "Install Git for Windows (https://git-scm.com/download/win), reopen the window, then re-run."
        )
    py310 = find_python310()
    print(f"Host Python 3.10: {py310}", flush=True)

    if not args.skip_clone:
        ensure_repo()
    elif not (TRELLIS_REPO / "trellis").is_dir():
        raise SystemExit(f"--skip-clone set but {TRELLIS_REPO} is missing trellis/")

    py = ensure_venv(py310, recreate=args.recreate)
    install_stack(py)
    verify(py)
    print(f"TRELLIS worker Python: {py}", flush=True)
    print("Restart the worker (START.bat or scripts\\run_all.py --skip-setup).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
