from __future__ import annotations

import base64
import gc
import os
from pathlib import Path

import numpy as np

from worker.models import GenerateRequest, ProgressFn

VARIANT_REPOS = {
    "text-base": "microsoft/TRELLIS-text-base",
    "text-large": "microsoft/TRELLIS-text-large",
    "text-xlarge": "microsoft/TRELLIS-text-xlarge",
}

_INSTALL_HINT = (
    "TRELLIS is not installed in this worker. "
    "On Windows run: py -3.13 scripts\\setup_trellis.py "
    "then restart with START.bat (uses .venv-trellis)."
)

_OOM_HINT = (
    "GPU ran out of memory. Close other GPU apps (games, Cycles), "
    "use text-base, or lower texture size / step count."
)


def _trellis_repo() -> Path:
    return Path(__file__).resolve().parents[2] / "vendor" / "TRELLIS"


def _ensure_ninja_on_path() -> None:
    """PyTorch JIT extensions look up ninja via PATH, not the Python package."""
    import shutil
    import sys

    if shutil.which("ninja"):
        return
    scripts = Path(sys.executable).resolve().parent
    ninja_bin = scripts / ("ninja.exe" if os.name == "nt" else "ninja")
    if not ninja_bin.is_file():
        return
    path = os.environ.get("PATH", "")
    parts = [p for p in path.split(os.pathsep) if p]
    scripts_str = str(scripts)
    if scripts_str not in parts:
        parts.insert(0, scripts_str)
        os.environ["PATH"] = os.pathsep.join(parts)


def _nvcc_name() -> str:
    return "nvcc.exe" if os.name == "nt" else "nvcc"


def _find_cuda_home() -> str | None:
    """Locate a real CUDA Toolkit install (needs nvcc — PyTorch wheels alone are not enough)."""
    import shutil

    for key in ("CUDA_HOME", "CUDA_PATH"):
        raw = os.environ.get(key)
        if not raw:
            continue
        root = Path(raw)
        if (root / "bin" / _nvcc_name()).is_file():
            return str(root.resolve())

    nvcc = shutil.which("nvcc")
    if nvcc:
        # .../CUDA/v12.4/bin/nvcc -> toolkit root
        return str(Path(nvcc).resolve().parent.parent)

    if os.name == "nt":
        base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if base.is_dir():
            versions = sorted(
                (p for p in base.iterdir() if p.is_dir() and (p / "bin" / "nvcc.exe").is_file()),
                key=lambda p: p.name,
                reverse=True,
            )
            if versions:
                return str(versions[0].resolve())
    return None


def _ensure_cuda_home() -> str | None:
    home = _find_cuda_home()
    if home:
        os.environ.setdefault("CUDA_HOME", home)
        os.environ.setdefault("CUDA_PATH", home)
        bin_dir = str(Path(home) / "bin")
        path_parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
        if bin_dir not in path_parts:
            path_parts.insert(0, bin_dir)
            os.environ["PATH"] = os.pathsep.join(path_parts)
    return home


def _msvc_available() -> bool:
    """nvdiffrast JIT on Windows needs cl.exe (VS Build Tools)."""
    import glob
    import shutil

    if os.name != "nt":
        return True
    if shutil.which("cl"):
        return True
    patterns = [
        r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
        r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
    ]
    for pattern in patterns:
        if glob.glob(pattern):
            return True
    return False


def _cuda_toolkit_ready() -> bool:
    """True when nvdiffrast can JIT-compile its CUDA plugin."""
    home = _ensure_cuda_home()
    if not home:
        return False
    if not (Path(home) / "bin" / _nvcc_name()).is_file():
        return False
    return _msvc_available()


def _ensure_runtime_env() -> None:
    # xformers ships from the pinned PyTorch index. The community flash-attn wheel is not installed.
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
    _ensure_ninja_on_path()
    _ensure_cuda_home()
    repo = _trellis_repo()
    if not repo.is_dir():
        return
    root = str(repo.resolve())
    import sys

    if root not in sys.path:
        sys.path.insert(0, root)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    if root not in parts:
        parts.insert(0, root)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)


def textured_export_available() -> bool:
    """True when TRELLIS can bake textured GLBs.

    Needs nvdiffrast + gaussian rasterizer packages, plus a CUDA Toolkit (nvcc)
    and MSVC so nvdiffrast can compile its plugin. Import alone is not enough.
    """
    _ensure_runtime_env()
    if not _cuda_toolkit_ready():
        return False
    try:
        import nvdiffrast.torch  # noqa: F401
        from diff_gaussian_rasterization import GaussianRasterizer  # noqa: F401
        return True
    except Exception:
        return False


def _is_texture_bake_env_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    needles = (
        "cuda_home",
        "cuda_path",
        "ninja is required",
        "could not locate a supported microsoft visual c++",
        "nvcc",
        "cl.exe",
    )
    return any(needle in message for needle in needles)


def _clear_cuda_cache() -> None:
    try:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        gc.collect()


def _export_mesh_glb(mesh_obj, dest: Path, simplify: float) -> None:
    """Fallback GLB export without Gaussian texture baking."""
    import trimesh

    vertices = mesh_obj.vertices.detach().cpu().numpy()
    faces = mesh_obj.faces.detach().cpu().numpy()
    # TRELLIS is z-up; Blender expects y-up.
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # simplify matches TRELLIS to_glb(): fraction of faces to keep (0.95 = keep 95%).
    if simplify > 0 and simplify < 1 and faces.shape[0] > 100:
        target_faces = max(100, int(faces.shape[0] * simplify))
        if target_faces < faces.shape[0]:
            try:
                # trimesh >= 4 takes `percent` positionally; pass the face count by name.
                mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
            except TypeError:
                try:
                    mesh = mesh.simplify_quadric_decimation(target_faces)
                except Exception as exc:
                    print(f"TRELLIS mesh decimation skipped: {exc}", flush=True)
            except Exception as exc:
                print(f"TRELLIS mesh decimation skipped: {exc}", flush=True)
    mesh.export(str(dest), file_type="glb")


def _open3d_mesh_from_glb(mesh_glb: str):
    import tempfile

    import open3d as o3d

    from worker.images import parse_mesh_glb

    raw = base64.b64decode(parse_mesh_glb(mesh_glb), validate=False)
    with tempfile.TemporaryDirectory(prefix="localtext3d_var_") as tmp:
        dest = Path(tmp) / "scan.glb"
        dest.write_bytes(raw)
        mesh = o3d.io.read_triangle_mesh(str(dest))
        if mesh.has_triangles() and len(mesh.vertices) > 0:
            return mesh
        import trimesh

        loaded = trimesh.load(str(dest), force="mesh")
    converted = o3d.geometry.TriangleMesh()
    converted.vertices = o3d.utility.Vector3dVector(np.asarray(loaded.vertices))
    converted.triangles = o3d.utility.Vector3iVector(np.asarray(loaded.faces))
    return converted


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch
    except Exception:
        return False
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


class TrellisEngine:
    name = "trellis"

    def __init__(self) -> None:
        self._pipeline = None
        self._variant: str | None = None

    def is_available(self) -> bool:
        _ensure_runtime_env()
        try:
            from trellis.pipelines import TrellisTextTo3DPipeline  # noqa: F401
        except Exception:
            return False
        return True

    def load(self, request: GenerateRequest, progress: ProgressFn) -> None:
        _ensure_runtime_env()
        if not self.is_available():
            raise RuntimeError(_INSTALL_HINT)
        if request.source in {"image", "views"}:
            progress("downloading_weights", "Loading microsoft/TRELLIS-image-large. First run downloads several GB")
            from trellis.pipelines import TrellisImageTo3DPipeline

            pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
            pipeline.cuda()
            self._pipeline = pipeline
            self._variant = "image-large"
            return
        repo = VARIANT_REPOS[request.trellis_variant]
        progress("downloading_weights", f"Loading {repo}. First run downloads several GB")
        from trellis.pipelines import TrellisTextTo3DPipeline

        pipeline = TrellisTextTo3DPipeline.from_pretrained(repo)
        pipeline.cuda()
        self._pipeline = pipeline
        self._variant = request.trellis_variant

    def unload(self) -> None:
        self._pipeline = None
        self._variant = None
        _clear_cuda_cache()

    def _sampler_params(self, request: GenerateRequest, formats: list[str]) -> dict:
        return {
            "sparse_structure_sampler_params": {
                "steps": request.structure_steps,
                "cfg_strength": request.cfg_strength,
            },
            "slat_sampler_params": {
                "steps": request.slat_steps,
                "cfg_strength": request.slat_cfg_strength,
            },
            "formats": formats,
        }

    def _run_pipeline(self, request: GenerateRequest, sampler: dict):
        if request.source in {"image", "views"}:
            from worker.images import pil_images

            images = pil_images(request.images)
            try:
                if len(images) == 1:
                    return self._pipeline.run(images[0], seed=request.seed, **sampler)
                return self._pipeline.run_multi_image(images, seed=request.seed, **sampler)
            finally:
                for owned in images:
                    try:
                        owned.close()
                    except Exception:
                        pass
        if request.source == "variant":
            mesh = _open3d_mesh_from_glb(request.mesh_glb)
            return self._pipeline.run_variant(
                mesh,
                request.prompt,
                seed=request.seed,
                slat_sampler_params=sampler["slat_sampler_params"],
                formats=sampler["formats"],
            )
        return self._pipeline.run(request.prompt, seed=request.seed, **sampler)

    def _write_glb(
        self,
        outputs: dict,
        dest: Path,
        *,
        textured: bool,
        simplify: float,
        texture_size: int,
    ) -> None:
        if textured:
            from trellis.utils import postprocessing_utils

            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=simplify,
                texture_size=texture_size,
            )
            glb.export(str(dest))
            return
        _export_mesh_glb(outputs["mesh"][0], dest, simplify)

    def generate(self, request: GenerateRequest, dest: Path, progress: ProgressFn) -> Path:
        want = request.cache_key()[1]
        if self._pipeline is None or self._variant != want:
            self.load(request, progress)
        dest.parent.mkdir(parents=True, exist_ok=True)
        progress("generating", "TRELLIS is sampling the 3D latent")
        textured = textured_export_available()
        formats = ["mesh", "gaussian"] if textured else ["mesh"]
        sampler = self._sampler_params(request, formats)

        try:
            outputs = self._run_pipeline(request, sampler)
        except Exception as exc:
            if _is_cuda_oom(exc):
                _clear_cuda_cache()
                raise RuntimeError(_OOM_HINT) from exc
            raise

        texture_sizes = [request.texture_size]
        if request.texture_size > 1024:
            texture_sizes.append(1024)

        last_exc: Exception | None = None
        for index, texture_size in enumerate(texture_sizes):
            if index > 0:
                progress(
                    "extracting_mesh",
                    "VRAM tight — retrying texture bake at 1024 instead of 2048",
                )
            elif textured:
                progress("extracting_mesh", f"Baking a {texture_size} texture and exporting GLB")
            else:
                progress(
                    "extracting_mesh",
                    "Exporting mesh GLB (mesh-only; textured export wheels are not installed)",
                )
            try:
                self._write_glb(
                    outputs,
                    dest,
                    textured=textured,
                    simplify=request.simplify_ratio,
                    texture_size=texture_size,
                )
                _clear_cuda_cache()
                return dest
            except Exception as exc:
                last_exc = exc
                if textured and _is_texture_bake_env_error(exc):
                    progress(
                        "extracting_mesh",
                        "Texture bake unavailable — exporting mesh-only GLB instead",
                    )
                    _clear_cuda_cache()
                    _export_mesh_glb(outputs["mesh"][0], dest, request.simplify_ratio)
                    _clear_cuda_cache()
                    return dest
                if _is_cuda_oom(exc) and index + 1 < len(texture_sizes):
                    _clear_cuda_cache()
                    continue
                if _is_cuda_oom(exc):
                    _clear_cuda_cache()
                    raise RuntimeError(_OOM_HINT) from exc
                raise

        if last_exc is not None:
            raise last_exc
        return dest
