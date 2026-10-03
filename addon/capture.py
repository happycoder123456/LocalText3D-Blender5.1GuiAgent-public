"""Capture a 3D view, an orbit of four views, or a still image as PNG base64."""

from __future__ import annotations

import base64
import math
import tempfile
from contextlib import contextmanager
from pathlib import Path

import bpy
from mathutils import Quaternion, Vector


def encode_png_bytes(data: bytes) -> str:
    if len(data) > 3_000_000:
        raise RuntimeError("Captured image is too large to send. Use a smaller viewport.")
    if not (data.startswith(b"\x89PNG") or data.startswith(b"\xff\xd8")):
        raise RuntimeError("Capture did not produce a PNG or JPEG")
    return base64.b64encode(data).decode("ascii")


def encode_image_file(path: str) -> str:
    src = Path(path)
    if not src.is_file():
        raise RuntimeError("Choose a reference image first")
    try:
        size = src.stat().st_size
    except OSError as exc:
        raise RuntimeError("Choose a reference image first") from exc
    if size > 12_000_000:
        raise RuntimeError("Reference image is too large to send. Use a smaller still.")
    data = src.read_bytes()
    if data.startswith(b"\x89PNG") or data.startswith(b"\xff\xd8"):
        return encode_png_bytes(data)
    image = bpy.data.images.load(str(src), check_existing=False)
    try:
        with tempfile.TemporaryDirectory(prefix="localtext3d_img_") as tmp:
            dest = Path(tmp) / "ref.png"
            # save_render() writes in the SCENE's output format (could be EXR /
            # FFmpeg); force PNG through the scene settings for this one save.
            with _png_render_settings(bpy.context.scene, str(dest.with_suffix(""))):
                image.save_render(str(dest))
            written = _written_png(dest.with_suffix(""))
            if written.stat().st_size > 3_000_000:
                raise RuntimeError("Captured image is too large to send. Use a smaller viewport.")
            return encode_png_bytes(written.read_bytes())
    finally:
        bpy.data.images.remove(image)


def _view3d(context):
    space = getattr(context, "space_data", None)
    area = getattr(context, "area", None)
    region = getattr(context, "region", None)
    if (
        space
        and space.type == "VIEW_3D"
        and getattr(space, "region_3d", None)
        and area
        and area.type == "VIEW_3D"
    ):
        if region is None or region.type != "WINDOW":
            region = next((item for item in area.regions if item.type == "WINDOW"), None)
        if region is not None:
            return area, region, space.region_3d
    screen = context.screen
    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        space = area.spaces.active
        region = next((item for item in area.regions if item.type == "WINDOW"), None)
        if space and region and getattr(space, "region_3d", None):
            return area, region, space.region_3d
    raise RuntimeError("Open a 3D Viewport to capture a view")


@contextmanager
def _png_render_settings(scene, filepath_no_ext: str):
    rnd = scene.render
    img = rnd.image_settings
    old = {
        "filepath": rnd.filepath,
        "film_transparent": rnd.film_transparent,
        "resolution_x": rnd.resolution_x,
        "resolution_y": rnd.resolution_y,
        "resolution_percentage": rnd.resolution_percentage,
        "file_format": img.file_format,
        "color_mode": img.color_mode,
    }
    rnd.filepath = filepath_no_ext
    rnd.film_transparent = True
    rnd.resolution_x = 512
    rnd.resolution_y = 512
    rnd.resolution_percentage = 100
    img.file_format = "PNG"
    img.color_mode = "RGBA"
    try:
        yield
    finally:
        rnd.filepath = old["filepath"]
        rnd.film_transparent = old["film_transparent"]
        rnd.resolution_x = old["resolution_x"]
        rnd.resolution_y = old["resolution_y"]
        rnd.resolution_percentage = old["resolution_percentage"]
        img.file_format = old["file_format"]
        img.color_mode = old["color_mode"]


def _written_png(stem: Path) -> Path:
    for candidate in (stem.with_suffix(".png"), Path(str(stem) + ".png"), stem):
        if candidate.is_file() and candidate.stat().st_size > 32:
            return candidate
    raise RuntimeError("Blender did not write a viewport PNG")


def _opengl_png(context, area, region) -> str:
    # TemporaryDirectory: the old mkdtemp folders piled up in %TEMP% per capture.
    with tempfile.TemporaryDirectory(prefix="localtext3d_view_") as tmp:
        stem = Path(tmp) / "view"
        with _png_render_settings(context.scene, str(stem)):
            with context.temp_override(
                window=context.window,
                screen=context.screen,
                area=area,
                region=region,
            ):
                bpy.ops.render.opengl(write_still=True)
        return encode_png_bytes(_written_png(stem).read_bytes())


def capture_viewport(context) -> list[str]:
    area, region, _rv3d = _view3d(context)
    return [_opengl_png(context, area, region)]


def capture_orbit(context) -> list[str]:
    area, region, rv3d = _view3d(context)
    original = rv3d.view_rotation.copy()
    turn = Quaternion(Vector((0.0, 0.0, 1.0)), math.radians(90.0))
    frames: list[str] = []
    try:
        for _ in range(4):
            frames.append(_opengl_png(context, area, region))
            rv3d.view_rotation = turn @ rv3d.view_rotation
            context.view_layer.update()
    finally:
        rv3d.view_rotation = original
        context.view_layer.update()
    return frames
