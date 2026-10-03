"""Decode and validate reference images sent to the worker."""

from __future__ import annotations

import base64
from io import BytesIO

MAX_IMAGES = 4
MAX_BYTES = 3_000_000
MAX_MESH_BYTES = 8_000_000
# Decode bombs: a 3 MB PNG can claim tens of thousands of pixels on a side.
MAX_IMAGE_SIDE = 8192
MAX_IMAGE_PIXELS = 25_000_000
MAX_DECODE_SIDE = 2048
_PNG = b"\x89PNG"
_JPEG = b"\xff\xd8"
_GLB = b"glTF"


def _raw_bytes(value: str) -> bytes:
    text = str(value or "").strip()
    if not text:
        raise ValueError("image is empty")
    if text.lower().startswith("data:") and "," in text:
        text = text.split(",", 1)[1]
    try:
        data = base64.b64decode(text, validate=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("image must be base64 PNG or JPEG") from exc
    if len(data) > MAX_BYTES:
        raise ValueError("each image must be under 3 MB")
    if not (data.startswith(_PNG) or data.startswith(_JPEG)):
        raise ValueError("images must be PNG or JPEG")
    return data


def parse_images(values) -> list[str]:
    if not values:
        return []
    if not isinstance(values, list):
        raise ValueError("images must be a list of base64 strings")
    if len(values) > MAX_IMAGES:
        raise ValueError("at most 4 images")
    return [base64.b64encode(_raw_bytes(item)).decode("ascii") for item in values]


def _close_quietly(image) -> None:
    try:
        image.close()
    except Exception:
        pass


def pil_images(values: list[str]):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    images = []
    try:
        for item in values:
            stream = BytesIO(_raw_bytes(item))
            try:
                image = Image.open(stream)
            except Image.DecompressionBombError as exc:
                raise ValueError("image is too large") from exc
            except Exception as exc:
                raise ValueError("image could not be decoded") from exc
            try:
                width, height = image.size
                if (
                    width < 1
                    or height < 1
                    or width > MAX_IMAGE_SIDE
                    or height > MAX_IMAGE_SIDE
                    or width * height > MAX_IMAGE_PIXELS
                ):
                    raise ValueError("image dimensions are too large")
                converted = image.convert("RGBA")
            finally:
                _close_quietly(image)
            converted.thumbnail((MAX_DECODE_SIDE, MAX_DECODE_SIDE), Image.Resampling.LANCZOS)
            images.append(converted)
    except Exception:
        for owned in images:
            _close_quietly(owned)
        raise
    if not images:
        raise ValueError("at least one image is required")
    return images


def parse_mesh_glb(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.lower().startswith("data:") and "," in text:
        text = text.split(",", 1)[1]
    try:
        data = base64.b64decode(text, validate=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("mesh_glb must be a base64 GLB") from exc
    if len(data) > MAX_MESH_BYTES:
        raise ValueError("mesh_glb must be under 8 MB")
    if not data.startswith(_GLB):
        raise ValueError("mesh_glb must be a GLB")
    return base64.b64encode(data).decode("ascii")
