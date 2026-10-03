from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from worker.images import parse_images, parse_mesh_glb

ProgressFn = Callable[[str, str], None]

# High-quality text defaults (microsoft/TRELLIS app_text.py). Safe on 16 GB VRAM.
DEFAULT_STRUCTURE_STEPS = 25
DEFAULT_SLAT_STEPS = 25
DEFAULT_CFG_STRENGTH = 7.5
DEFAULT_SLAT_CFG_TEXT = 7.5
DEFAULT_SLAT_CFG_IMAGE = 3.0
DEFAULT_SIMPLIFY_RATIO = 0.95
DEFAULT_TEXTURE_SIZE = 2048
MAX_PROMPT_CHARS = 8192


@dataclass
class GenerateRequest:
    engine: str
    prompt: str
    seed: int = 1
    trellis_variant: str = "text-large"
    structure_steps: int = DEFAULT_STRUCTURE_STEPS
    slat_steps: int = DEFAULT_SLAT_STEPS
    cfg_strength: float = DEFAULT_CFG_STRENGTH
    slat_cfg_strength: float = DEFAULT_SLAT_CFG_TEXT
    simplify_ratio: float = DEFAULT_SIMPLIFY_RATIO
    texture_size: int = DEFAULT_TEXTURE_SIZE
    shap_e_steps: int = 64
    shap_e_guidance: float = 15.0
    source: str = "text"
    images: list[str] = field(default_factory=list)
    mesh_glb: str = ""
    output_path: Path | None = None

    def cache_key(self) -> tuple[str, str]:
        if self.engine == "trellis":
            if self.source in {"image", "views"}:
                return ("trellis", "image-large")
            return ("trellis", self.trellis_variant)
        if self.engine == "shap_e" and self.source in {"image", "views"}:
            return ("shap_e", "img2img")
        return (self.engine, "")


@dataclass
class Job:
    job_id: str
    request: GenerateRequest
    status: str = "queued"
    message: str = "Waiting for a free worker slot"
    output_path: str | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "message": self.message,
            "output_path": self.output_path,
            "error": self.error,
            "engine": self.request.engine,
            "prompt": self.request.prompt,
            "seed": self.request.seed,
            "trellis_variant": self.request.trellis_variant,
            "source": self.request.source,
            "created": self.created,
        }


def _bounded_int(value, default: int, lo: int, hi: int, name: str) -> int:
    if value is None or value == "":
        n = default
    else:
        n = int(value)
    if n < lo or n > hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return n


def _bounded_float(value, default: float, lo: float, hi: float, name: str) -> float:
    if value is None or value == "":
        n = float(default)
    else:
        n = float(value)
    if n < lo or n > hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return n


def request_from_payload(data: dict) -> GenerateRequest:
    engine = str(data.get("engine") or "trellis").strip().lower()
    if engine in {"shape", "shape_e", "shape-e"}:
        engine = "shap_e"
    if engine not in {"trellis", "shap_e", "mock"}:
        raise ValueError("engine must be 'trellis' or 'shap_e'")

    prompt = str(data.get("prompt") or "").strip()
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"prompt must be at most {MAX_PROMPT_CHARS} characters")
    source = str(data.get("source") or "text").strip().lower()
    if source in {"viewport", "file"}:
        source = "image"
    if source in {"orbit", "multi", "multiview"}:
        source = "views"
    images = parse_images(data.get("images") or [])
    mesh_glb = parse_mesh_glb(str(data.get("mesh_glb") or ""))
    if source in {"scan", "megascans", "fab", "quixel", "bridge"}:
        source = "variant" if mesh_glb else ("views" if len(images) > 1 else "image")
    if source in {"variant", "mesh"}:
        source = "variant"
    if source not in {"text", "image", "views", "variant"}:
        raise ValueError("source must be text, image, views, or variant")
    if source == "text":
        if images:
            source = "views" if len(images) > 1 else "image"
        elif not prompt:
            raise ValueError("prompt is required")
    elif source == "variant":
        if not mesh_glb:
            raise ValueError("variant source needs a GLB mesh from the selected scan")
        if not prompt:
            raise ValueError("variant source needs a prompt")
        if engine == "shap_e":
            raise ValueError("mesh variants need TRELLIS")
    else:
        if not images:
            raise ValueError("image source needs at least one PNG or JPEG")
        if source == "image" and len(images) > 1:
            source = "views"
        if source == "views" and len(images) < 2:
            source = "image"
        if not prompt:
            prompt = "orbit" if source == "views" else "image"

    variant = str(data.get("trellis_variant") or "text-large")
    if variant not in {"text-base", "text-large", "text-xlarge"}:
        raise ValueError("trellis_variant must be text-base, text-large, or text-xlarge")

    texture = int(data.get("texture_size") or DEFAULT_TEXTURE_SIZE)
    if texture not in {512, 1024, 2048}:
        raise ValueError("texture_size must be 512, 1024, or 2048")

    slat_default = DEFAULT_SLAT_CFG_IMAGE if source in {"image", "views"} else DEFAULT_SLAT_CFG_TEXT

    return GenerateRequest(
        engine=engine,
        prompt=prompt,
        seed=_bounded_int(data.get("seed") or 1, 1, 0, 2_147_483_647, "seed"),
        trellis_variant=variant,
        structure_steps=_bounded_int(
            data.get("structure_steps") or DEFAULT_STRUCTURE_STEPS,
            DEFAULT_STRUCTURE_STEPS,
            1,
            64,
            "structure_steps",
        ),
        slat_steps=_bounded_int(
            data.get("slat_steps") or DEFAULT_SLAT_STEPS,
            DEFAULT_SLAT_STEPS,
            1,
            64,
            "slat_steps",
        ),
        cfg_strength=_bounded_float(
            data.get("cfg_strength") or DEFAULT_CFG_STRENGTH,
            DEFAULT_CFG_STRENGTH,
            0.0,
            30.0,
            "cfg_strength",
        ),
        slat_cfg_strength=_bounded_float(
            data.get("slat_cfg_strength") or slat_default,
            slat_default,
            0.0,
            30.0,
            "slat_cfg_strength",
        ),
        simplify_ratio=_bounded_float(
            data.get("simplify_ratio") or DEFAULT_SIMPLIFY_RATIO,
            DEFAULT_SIMPLIFY_RATIO,
            0.5,
            1.0,
            "simplify_ratio",
        ),
        texture_size=texture,
        shap_e_steps=_bounded_int(data.get("shap_e_steps") or 64, 64, 1, 128, "shap_e_steps"),
        shap_e_guidance=_bounded_float(
            data.get("shap_e_guidance") or 15.0,
            15.0,
            1.0,
            40.0,
            "shap_e_guidance",
        ),
        source=source,
        images=images,
        mesh_glb=mesh_glb,
    )


def noop_progress(_status: str, _message: str) -> None:
    return None
