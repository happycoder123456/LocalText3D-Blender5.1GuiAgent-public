"""Blender helpers for local Megascans / Fab / Bridge assets."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

import bpy

from . import capture
from . import library
from .prefs import get_prefs
from .props import get_props

_BRIDGE_HOSTS = {"127.0.0.1", "localhost"}
_BRIDGE_PORTS = {5017, 28241}
_MAX_PREVIEW_BYTES = 3_000_000


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Preview URL redirected; refusing")


def library_roots(context) -> list[Path]:
    extra = []
    try:
        for lib in context.preferences.filepaths.asset_libraries:
            extra.append(Path(lib.path))
    except Exception:
        pass
    roots = library.all_library_roots(get_prefs(context).library_path)
    seen = {path.resolve() for path in roots}
    for path in extra:
        name = path.name.lower()
        if not path.is_dir():
            continue
        if not any(key in name for key in ("megascan", "quixel", "fab", "bridge")):
            continue
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            roots.append(resolved)
    return roots


def library_root(context) -> Path | None:
    roots = library_roots(context)
    return roots[0] if roots else None


def _local_bridge_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "http":
        return False
    host = (parsed.hostname or "").lower()
    port = parsed.port or 80
    return host in _BRIDGE_HOSTS and port in _BRIDGE_PORTS


def fetch_bridge_preview(url: str) -> str:
    if not _local_bridge_url(url):
        raise RuntimeError("Preview URL is not the local Quixel Bridge")
    req = Request(url, headers={"Accept": "image/jpeg,image/png,*/*"})
    opener = build_opener(_NoRedirect)
    with opener.open(req, timeout=2.0) as resp:
        final = resp.geturl()
        if final != url and not _local_bridge_url(final):
            raise RuntimeError("Preview URL is not the local Quixel Bridge")
        data = resp.read(_MAX_PREVIEW_BYTES + 1)
    if len(data) > _MAX_PREVIEW_BYTES:
        raise RuntimeError("Bridge preview is too large")
    return capture.encode_png_bytes(data)


def _item_get(item, key: str) -> str:
    if isinstance(item, dict):
        return str(item.get(key) or "")
    return str(getattr(item, key, "") or "")


def encode_asset_preview(item) -> str:
    preview_file = _item_get(item, "preview_file")
    path = Path(preview_file) if preview_file else None
    if path and path.is_file():
        return capture.encode_image_file(str(path))
    url = _item_get(item, "preview_url")
    if url:
        try:
            return fetch_bridge_preview(url)
        except Exception:
            pass
    embedded = _item_get(item, "embedded")
    if embedded:
        # Bridge sometimes stores "data:image/png;base64,...." — strip the header.
        payload = str(embedded).split(",", 1)[-1] if str(embedded).startswith("data:") else str(embedded)
        if len(payload) > 4_200_000:
            raise RuntimeError("Embedded scan preview is too large")
        return capture.encode_png_bytes(base64.b64decode(payload, validate=False))
    raise RuntimeError(
        "No preview on disk. Start Quixel Bridge, or import the scan and use Selected."
    )


def snapshot_materials(context) -> list:
    mats = []
    seen = set()
    for obj in list(context.selected_objects) + list(getattr(context, "objects_in_mode", []) or []):
        data = getattr(obj, "data", None)
        materials = getattr(data, "materials", None)
        if not materials:
            continue
        for mat in materials:
            if mat is None or mat.name in seen:
                continue
            seen.add(mat.name)
            mats.append(mat)
    return mats


def apply_materials(root, materials) -> None:
    if not materials or root is None:
        return
    # Materials were snapshotted minutes ago; the user may have deleted one since.
    live = []
    for mat in materials:
        try:
            name = mat.name
        except ReferenceError:
            continue
        found = bpy.data.materials.get(name)
        if found is not None:
            live.append(found)
    if not live:
        return
    meshes = [root] if root.type == "MESH" else []
    meshes.extend([child for child in root.children_recursive if child.type == "MESH"])
    for mesh_obj in meshes:
        mesh_obj.data.materials.clear()
        for mat in live:
            mesh_obj.data.materials.append(mat)


def _mesh_list(root) -> list:
    meshes = [root] if getattr(root, "type", "") == "MESH" else []
    meshes.extend([child for child in root.children_recursive if child.type == "MESH"])
    return meshes


def apply_folder_material(root, folder: str) -> None:
    folder_path = Path(folder)
    albedo = library._best_preview_file(folder_path) if folder_path.is_dir() else None
    if root is None or albedo is None:
        return
    mat = bpy.data.materials.new(name=folder_path.name or "ScanLook")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    principled = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(albedo), check_existing=True)
    links.new(tex.outputs["Color"], principled.inputs["Base Color"])
    apply_materials(root, [mat])
    meshes = _mesh_list(root)
    if not meshes:
        return
    try:
        bpy.ops.object.select_all(action="DESELECT")
        for mesh_obj in meshes:
            mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass


def finish_look(root, materials, folder: str = "") -> None:
    if materials:
        apply_materials(root, materials)
        return
    if folder:
        apply_folder_material(root, folder)


def export_selected_glb(context) -> str:
    meshes = [obj for obj in context.selected_objects if obj.type == "MESH"]
    if not meshes:
        for obj in list(context.selected_objects):
            meshes.extend([child for child in obj.children_recursive if child.type == "MESH"])
    if not meshes:
        raise RuntimeError("Select a Megascans/Fab mesh imported into this scene")
    with tempfile.TemporaryDirectory(prefix="localtext3d_scan_") as tmp:
        dest = Path(tmp) / "scan.glb"
        view = context.view_layer
        previous = [obj for obj in context.view_layer.objects if obj.select_get()]
        try:
            bpy.ops.object.select_all(action="DESELECT")
            for obj in meshes:
                obj.select_set(True)
            view.objects.active = meshes[0]
            bpy.ops.export_scene.gltf(
                filepath=str(dest),
                export_format="GLB",
                use_selection=True,
                export_apply=True,
            )
        finally:
            bpy.ops.object.select_all(action="DESELECT")
            for obj in previous:
                try:
                    obj.select_set(True)
                except Exception:
                    pass
        try:
            size = dest.stat().st_size
        except OSError as exc:
            raise RuntimeError("Blender did not export a GLB") from exc
        if size > 8_000_000:
            raise RuntimeError("Selected scan is over 8 MB. Export a lower LOD, then try again.")
        data = dest.read_bytes()
        if not data.startswith(b"glTF"):
            raise RuntimeError("Blender did not export a GLB")
        return base64.b64encode(data).decode("ascii")


def auto_scan_fields(context, prompt: str, required: bool = False) -> dict | None:
    props = get_props(context)
    props.last_scan_pick = ""
    props.last_scan_folder = ""
    prompt = (prompt or "").strip()
    if not prompt:
        if required:
            raise RuntimeError("Type what you want. A matching local scan is picked automatically.")
        return None
    assets = library.list_all_scan_assets(library_roots(context), limit=400)
    picked = library.pick_auto_scan(assets, prompt)
    if picked is None:
        if required:
            raise RuntimeError(
                "No matching local Megascans/Fab scan for that prompt. Import one, or type a closer name."
            )
        return None
    role, asset = picked
    name = asset.get("name") or asset.get("id") or "scan"
    if role == "shape":
        try:
            image = encode_asset_preview(asset)
        except Exception:
            role = "look"
        else:
            props.last_scan_pick = f"Using {name}"
            return {
                "prompt": prompt,
                "source": "image",
                "images": [image],
                "mesh_glb": "",
                "name": name,
            }
    props.last_scan_folder = asset.get("folder") or ""
    props.last_scan_pick = f"Texturing with {name}"
    return {
        "prompt": prompt,
        "source": "text",
        "images": [],
        "mesh_glb": "",
        "name": name,
    }


def build_scan_fields(context, props) -> dict:
    prompt = props.prompt.strip()
    if props.scan_use == "selected":
        if props.engine != "trellis":
            raise RuntimeError("Mesh variants from Megascans/Fab need TRELLIS")
        if not prompt:
            raise RuntimeError("Describe the variant, like weathered limestone stairs")
        return {
            "prompt": prompt,
            "source": "variant",
            "images": [],
            "mesh_glb": export_selected_glb(context),
        }
    picked = auto_scan_fields(context, prompt, required=True)
    if not picked:
        raise RuntimeError("No matching local Megascans/Fab scan for that prompt.")
    return picked


def fill_scan_list(context) -> str:
    props = get_props(context)
    roots = library_roots(context)
    props.scan_assets.clear()
    if not roots:
        return "Set a Megascans/Fab/Bridge folder in addon preferences"
    items = library.list_all_scan_assets(roots, query=props.scan_filter, limit=40)
    for item in items:
        row = props.scan_assets.add()
        row.asset_id = item.get("id") or ""
        row.name = item.get("name") or row.asset_id
        row.preview_file = item.get("preview_file") or ""
        row.preview_url = item.get("preview_url") or ""
        row.embedded = item.get("embedded") or ""
        row.folder = item.get("folder") or ""
        row.asset_type = item.get("type") or ""
    if items:
        props.scan_index = 0
        return f"{len(items)} local assets across {len(roots)} folder(s)"
    return "No local scans found. Download Megascans/Fab assets, then generate again."


class LOCALTEXT3D_OT_refresh_scans(bpy.types.Operator):
    bl_idname = "localtext3d.refresh_scans"
    bl_label = "Refresh Library"
    bl_description = "Read the local Megascans / Fab / Bridge folder"

    def execute(self, context):
        message = fill_scan_list(context)
        get_props(context).status_detail = message
        self.report({"INFO"}, message)
        return {"FINISHED"}


class LOCALTEXT3D_OT_use_scan_asset(bpy.types.Operator):
    bl_idname = "localtext3d.use_scan_asset"
    bl_label = "Use Scan"
    bl_description = "Use this local Megascans/Fab preview for TRELLIS image-to-3D"

    index: bpy.props.IntProperty()

    def execute(self, context):
        props = get_props(context)
        if self.index < 0 or self.index >= len(props.scan_assets):
            return {"CANCELLED"}
        props.scan_index = self.index
        item = props.scan_assets[self.index]
        if not props.prompt.strip():
            props.prompt = item.name
        return {"FINISHED"}


CLASSES = (
    LOCALTEXT3D_OT_refresh_scans,
    LOCALTEXT3D_OT_use_scan_asset,
)
