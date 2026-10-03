from __future__ import annotations

import gc
from pathlib import Path

from worker.models import GenerateRequest, ProgressFn


class ShapEEngine:
    name = "shap_e"

    def __init__(self) -> None:
        self._pipeline = None
        self._mode: str | None = None

    def is_available(self) -> bool:
        try:
            from diffusers import ShapEPipeline  # noqa: F401
        except Exception:
            return False
        return True

    def _device_and_dtype(self):
        import torch

        if torch.cuda.is_available():
            return "cuda", torch.float16
        return "cpu", torch.float32

    def load(self, request: GenerateRequest, progress: ProgressFn) -> None:
        if not self.is_available():
            raise RuntimeError(
                "Shap-E is not installed. In the worker env run: "
                "pip install -r worker/requirements-shap-e.txt"
            )
        device, dtype = self._device_and_dtype()
        image_mode = request.source in {"image", "views"}
        if self._pipeline is not None:
            # Release the previous text/img2img pipeline before the new one lands on the GPU.
            self.unload()
        if image_mode:
            try:
                from diffusers import ShapEImg2ImgPipeline as Pipeline
            except Exception as exc:
                raise RuntimeError(
                    "Image mode for Shap-E needs ShapEImg2ImgPipeline. "
                    "Use TRELLIS for viewport/orbit/reference images, or reinstall Shap-E."
                ) from exc
            repo = "openai/shap-e-img2img"
            progress("downloading_weights", f"Loading {repo} on {device}. First run downloads a few GB")
        else:
            from diffusers import ShapEPipeline as Pipeline

            repo = "openai/shap-e"
            progress("downloading_weights", f"Loading {repo} on {device}. First run downloads a few GB")
        try:
            pipeline = Pipeline.from_pretrained(repo, torch_dtype=dtype)
        except TypeError:
            pipeline = Pipeline.from_pretrained(repo, dtype=dtype)
        self._pipeline = pipeline.to(device)
        self._mode = "img2img" if image_mode else "text"

    def unload(self) -> None:
        self._pipeline = None
        self._mode = None
        try:
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            gc.collect()

    def generate(self, request: GenerateRequest, dest: Path, progress: ProgressFn) -> Path:
        want = "img2img" if request.source in {"image", "views"} else "text"
        if self._pipeline is None or self._mode != want:
            self.load(request, progress)

        dest.parent.mkdir(parents=True, exist_ok=True)
        progress("generating", "Shap-E is denoising the implicit 3D field")
        kwargs = {
            "guidance_scale": request.shap_e_guidance,
            "num_inference_steps": request.shap_e_steps,
            "output_type": "mesh",
        }
        if want == "img2img":
            from worker.images import pil_images

            loaded = pil_images(request.images)
            image = loaded[0].convert("RGB")
            try:
                try:
                    result = self._pipeline(image, frame_size=256, **kwargs)
                except TypeError:
                    result = self._pipeline(image, size=256, **kwargs)
            finally:
                image.close()
                for owned in loaded:
                    try:
                        owned.close()
                    except Exception:
                        pass
        else:
            try:
                result = self._pipeline(request.prompt, frame_size=256, **kwargs)
            except TypeError:
                result = self._pipeline(request.prompt, size=256, **kwargs)
        meshes = result.images
        if not meshes:
            raise RuntimeError("Shap-E returned no mesh")

        progress("extracting_mesh", "Converting the Shap-E mesh to GLB")
        ply_path = dest.with_suffix(".ply")
        try:
            self._write_glb(meshes[0], dest, ply_path)
        finally:
            if ply_path.exists():
                ply_path.unlink()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return dest

    def _write_glb(self, mesh_obj, dest: Path, ply_path: Path) -> None:
        from diffusers.utils import export_to_ply
        import numpy as np
        import trimesh

        export_to_ply(mesh_obj, str(ply_path))
        mesh = trimesh.load(str(ply_path), force="mesh")
        try:
            from trimesh.transformations import rotation_matrix
        except ImportError:
            rotation_matrix = trimesh.transformations.rotation_matrix
        mesh.apply_transform(rotation_matrix(-np.pi / 2, [1.0, 0.0, 0.0]))
        mesh.export(str(dest), file_type="glb")
