from __future__ import annotations

import os
import random
import re
import tempfile
from pathlib import Path

import bpy
from mathutils import Vector

from . import capture
from . import client
from . import scan
from . import studio
from .prefs import get_prefs
from .props import get_props


def _set_status(props, status: str, detail: str) -> None:
    props.status = status
    props.status_detail = detail


def _slug(prompt: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", prompt.strip())[:48].strip("_")
    return cleaned or "Text3D"


def _temp_glb_path(job_id: str) -> str:
    # job_id comes from the worker response; never let it carry path separators.
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(job_id or ""))[:64] or "job"
    return os.path.join(tempfile.gettempdir(), f"localtext3d_{safe}.glb")


def _build_payload(context, props) -> dict:
    source = props.source
    prompt = props.prompt.strip()
    images: list[str] = []
    worker_source = "text"
    mesh_glb = ""
    if source == "image":
        # File browser stores "//ref.png" (blend-relative) once the file is saved.
        image_path = bpy.path.abspath(props.image_path) if props.image_path else ""
        images = [capture.encode_image_file(image_path)]
        worker_source = "image"
        prompt = prompt or Path(image_path).stem or "image"
    elif source == "viewport":
        images = capture.capture_viewport(context)
        worker_source = "image"
        prompt = prompt or "viewport"
    elif source == "orbit":
        if props.engine == "trellis":
            images = capture.capture_orbit(context)
            worker_source = "views"
            prompt = prompt or "orbit"
        else:
            images = capture.capture_viewport(context)
            worker_source = "image"
            prompt = prompt or "viewport"
    elif source == "scan":
        extra = scan.build_scan_fields(context, props)
        prompt = extra["prompt"]
        worker_source = extra["source"]
        images = extra.get("images") or []
        mesh_glb = extra.get("mesh_glb") or ""
    elif not prompt:
        raise RuntimeError("Type what you want to generate.")
    else:
        extra = scan.auto_scan_fields(context, prompt) if props.auto_scan else None
        if extra:
            worker_source, images = extra["source"], extra.get("images") or []

    payload = {
        "engine": props.engine,
        "prompt": prompt,
        "seed": int(props.seed),
        "source": worker_source,
    }
    if images:
        payload["images"] = images
    if mesh_glb:
        payload["mesh_glb"] = mesh_glb
    if props.engine == "trellis":
        payload.update(
            {
                "trellis_variant": props.trellis_variant,
                "structure_steps": int(props.structure_steps),
                "slat_steps": int(props.slat_steps),
                "cfg_strength": float(props.cfg_strength),
                "slat_cfg_strength": float(props.slat_cfg_strength),
                "simplify_ratio": float(props.simplify_ratio),
                "texture_size": int(props.texture_size),
            }
        )
    else:
        payload.update(
            {
                "shap_e_steps": int(props.shap_e_steps),
                "shap_e_guidance": float(props.shap_e_guidance),
            }
        )
    return payload


def _format_health(info: dict) -> str:
    device = info.get("device_name") or info.get("device") or "unknown device"
    loaded = info.get("loaded_engine") or "none"
    engines = ", ".join(info.get("available_engines") or [])
    mock = " · mock" if info.get("mock") else ""
    vram = info.get("vram_free_mb")
    vram_bit = f" · {vram} MB free" if isinstance(vram, int) else ""
    texture_bit = ""
    if "trellis" in (info.get("available_engines") or []):
        if info.get("trellis_textured_export"):
            texture_bit = " · textured GLB"
        else:
            texture_bit = " · mesh-only"
    return (
        f"Connected · {device}{vram_bit} · engines: {engines or 'none'}"
        f" · loaded: {loaded}{texture_bit}{mock}"
    )


def _refresh_history(context) -> None:
    props = get_props(context)
    prefs = get_prefs(context)
    try:
        jobs = client.list_jobs(prefs.worker_url, timeout=min(prefs.request_timeout, 5.0))
    except client.WorkerError:
        return
    props.history.clear()
    for job in jobs[:8]:
        if not isinstance(job, dict):
            continue
        item = props.history.add()
        item.job_id = str(job.get("job_id") or "")
        item.prompt = str(job.get("prompt") or "")
        item.engine = str(job.get("engine") or "")
        try:
            item.seed = int(job.get("seed") or 0)
        except (TypeError, ValueError):
            item.seed = 0
        item.variant = str(job.get("trellis_variant") or "")
        item.output_path = str(job.get("output_path") or "")


def _world_bbox(meshes):
    corners = []
    for obj in meshes:
        for corner in obj.bound_box:
            corners.append(obj.matrix_world @ Vector(corner))
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _studio_place(context, imported, object_name: str, props, meta: dict):
    imported_set = set(imported)
    meshes = [obj for obj in imported if obj.type == "MESH"]
    roots = [obj for obj in imported if obj.parent not in imported_set]
    if not roots:
        roots = list(imported)

    if len(roots) > 1:
        parent = bpy.data.objects.new(object_name, None)
        context.collection.objects.link(parent)
        for root in roots:
            world = root.matrix_world.copy()
            root.parent = parent
            root.matrix_parent_inverse = parent.matrix_world.inverted()
            root.matrix_world = world
        roots = [parent]
    root = roots[0]
    root.name = object_name
    if root.data:
        root.data.name = object_name

    if meshes:
        bbox_min, bbox_max = _world_bbox(meshes)
        size = tuple(bbox_max[i] - bbox_min[i] for i in range(3))
        root.scale *= studio.fit_scale(size, float(props.target_size))
        context.view_layer.update()
        bbox_min, bbox_max = _world_bbox(meshes)
        cursor = context.scene.cursor.location
        dest_xy = studio.destination_xy(
            (cursor.x, cursor.y),
            slot=int(props.compare_slot) if props.line_up else 0,
            target_size=float(props.target_size),
            place_at_cursor=bool(props.place_at_cursor),
            line_up=bool(props.line_up),
        )
        dest_z = float(cursor.z) if props.place_at_cursor else 0.0
        dx, dy, dz = studio.move_delta(bbox_min, bbox_max, dest_xy, dest_z, bool(props.sit_on_ground))
        root.location.x += dx
        root.location.y += dy
        root.location.z += dz
        if props.line_up:
            props.compare_slot += 1
        bpy.ops.object.select_all(action="DESELECT")
        for mesh_obj in meshes:
            mesh_obj.select_set(True)
        context.view_layer.objects.active = meshes[0]
        try:
            bpy.ops.object.shade_smooth()
        except Exception:
            pass

    root["localtext3d_prompt"] = meta.get("prompt") or ""
    root["localtext3d_engine"] = meta.get("engine") or ""
    root["localtext3d_seed"] = int(meta.get("seed") or 0)

    view_layer = context.view_layer
    bpy.ops.object.select_all(action="DESELECT")
    root.select_set(True)
    view_layer.objects.active = root
    if context.area and context.area.type == "VIEW_3D":
        bpy.ops.view3d.view_selected()
    return root


def _import_glb(context, filepath: str, object_name: str, props, meta: dict):
    # The job finishes minutes after the click; the user may be in Edit mode by
    # then, where select_all / import poll() fail and the result is lost.
    if getattr(context, "mode", "OBJECT") != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            pass
    bpy.ops.object.select_all(action="DESELECT")
    before = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.gltf(filepath=filepath)
    except AttributeError as exc:
        raise RuntimeError(
            "glTF importer is disabled. Enable the built-in 'glTF 2.0 format' extension in Preferences."
        ) from exc
    imported = [obj for obj in bpy.data.objects if obj not in before]
    if not imported:
        raise RuntimeError("GLB imported but no new objects appeared")
    return _studio_place(context, imported, object_name, props, meta)


class LOCALTEXT3D_OT_check_worker(bpy.types.Operator):
    bl_idname = "localtext3d.check_worker"
    bl_label = "Check Worker"
    bl_description = "Ping the local worker and show which engines are ready"

    def execute(self, context):
        props = get_props(context)
        prefs = get_prefs(context)
        try:
            info = client.health(prefs.worker_url, timeout=prefs.request_timeout)
        except client.WorkerError as exc:
            props.worker_summary = str(exc)
            _set_status(
                props,
                "Worker offline",
                "Start the worker: double-click START.bat and leave that window open.",
            )
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        props.worker_summary = _format_health(info)
        available = info.get("available_engines") or []
        props.available_engines = ",".join(available)
        _refresh_history(context)
        if props.engine not in available and not info.get("mock"):
            alt = "shap_e" if "shap_e" in available else (available[0] if available else "")
            hint = f" Switch Engine to {alt}." if alt else " Start the worker with Shap-E installed."
            _set_status(
                props,
                "Engine missing",
                f"{props.engine} is not installed.{hint}",
            )
        else:
            _set_status(props, "Idle", "Worker is ready. Enter a prompt, then generate.")
        self.report({"INFO"}, props.worker_summary)
        return {"FINISHED"}


class LOCALTEXT3D_OT_generate(bpy.types.Operator):
    bl_idname = "localtext3d.generate"
    bl_label = "Generate 3D Model"
    bl_description = "Send the prompt to the local worker and import the finished mesh"
    bl_options = {"REGISTER", "UNDO"}

    _timer = None
    _job_id = ""
    _object_name = "Text3D"
    _meta = None

    def modal(self, context, event):
        props = get_props(context)
        prefs = get_prefs(context)

        if event.type == "ESC" or (event.type == "TIMER" and not props.is_running):
            self._finish(context, cancelled=True)
            _set_status(props, "Cancelled", "Generation stopped.")
            return {"CANCELLED"}

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        try:
            job = client.job_status(prefs.worker_url, self._job_id, timeout=prefs.request_timeout)
        except client.WorkerError as exc:
            self._finish(context, cancelled=True)
            _set_status(props, "Failed", str(exc))
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        status = job.get("status") or "unknown"
        message = job.get("message") or ""
        _set_status(props, status.replace("_", " ").title(), message or "Working…")

        if status in {"queued", "switching_engine", "downloading_weights", "generating", "extracting_mesh", "cancelling"}:
            return {"PASS_THROUGH"}

        if status == "cancelled":
            self._finish(context, cancelled=True)
            _set_status(props, "Cancelled", "Worker stopped the job.")
            return {"CANCELLED"}

        if status == "failed":
            self._finish(context, cancelled=True)
            error = job.get("error") or message or "Generation failed"
            _set_status(props, "Failed", error)
            self.report({"ERROR"}, error)
            return {"CANCELLED"}

        if status != "done":
            return {"PASS_THROUGH"}

        try:
            glb_path = self._resolve_glb(prefs, job)
            _set_status(props, "Importing", f"Importing {os.path.basename(glb_path)}")
            root = _import_glb(context, glb_path, self._object_name, props, self._meta or {})
            scan.finish_look(root, getattr(self, "_borrow_mats", None) or [], props.last_scan_folder)
            _refresh_history(context)
        except (client.WorkerError, OSError, RuntimeError) as exc:
            self._finish(context, cancelled=True)
            _set_status(props, "Failed", str(exc))
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        self._finish(context, cancelled=False)
        _set_status(props, "Done", f"{props.last_scan_pick + '. ' if props.last_scan_pick else ''}Imported {self._object_name}")
        self.report({"INFO"}, f"Imported {self._object_name}")
        return {"FINISHED"}

    def _resolve_glb(self, prefs, job: dict) -> str:
        contained = client.glb_under_worker_output(str(job.get("output_path") or ""))
        if contained:
            return contained
        dest = _temp_glb_path(self._job_id)
        return client.download_job_file(
            prefs.worker_url,
            self._job_id,
            dest,
            timeout=prefs.download_timeout,
        )

    def execute(self, context):
        props = get_props(context)
        prefs = get_prefs(context)
        if props.is_running:
            self.report({"WARNING"}, "A generation is already running")
            return {"CANCELLED"}

        self._borrow_mats = []
        props.last_scan_pick = props.last_scan_folder = ""
        if props.source == "scan" and props.borrow_scan_materials:
            self._borrow_mats = scan.snapshot_materials(context)

        try:
            payload = _build_payload(context, props)
        except (RuntimeError, OSError) as exc:
            # OSError: unreadable image / broken symlink in the Megascans library walk.
            _set_status(props, "Need input", str(exc))
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}

        prompt = str(payload.get("prompt") or "Text3D")
        _set_status(props, "Connecting", props.last_scan_pick or f"Sending to {prefs.worker_url}")
        try:
            bulky = bool(payload.get("images") or payload.get("mesh_glb"))
            response = client.start_generate(
                prefs.worker_url,
                payload,
                timeout=max(prefs.request_timeout, 30.0) if bulky else prefs.request_timeout,
            )
        except client.WorkerError as exc:
            _set_status(props, "Failed", str(exc))
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        job_id = response.get("job_id")
        if not job_id:
            _set_status(props, "Failed", "Worker did not return a job id")
            self.report({"ERROR"}, "Worker did not return a job id")
            return {"CANCELLED"}

        self._job_id = job_id
        self._object_name = _slug(prompt)
        self._meta = {"prompt": prompt, "engine": props.engine, "seed": int(props.seed)}
        props.last_job_id = job_id
        props.is_running = True
        _set_status(props, "Queued", "Waiting for the worker to start the job")

        wm = context.window_manager
        self._timer = wm.event_timer_add(prefs.poll_interval, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _finish(self, context, cancelled: bool) -> None:
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        get_props(context).is_running = False
        if cancelled:
            return


class LOCALTEXT3D_OT_cancel(bpy.types.Operator):
    bl_idname = "localtext3d.cancel"
    bl_label = "Cancel"
    bl_description = "Stop the current generation. The worker aborts at its next stage so the GPU frees up"

    def execute(self, context):
        props = get_props(context)
        prefs = get_prefs(context)
        props.is_running = False
        note = "Stopped watching the current job."
        job_id = str(props.last_job_id or "")
        if job_id:
            try:
                client.cancel_job(prefs.worker_url, job_id, timeout=min(prefs.request_timeout, 5.0))
                note = "Cancel sent to the worker; it stops after the current stage."
            except client.WorkerError:
                # Older worker or already finished — the panel is free either way.
                pass
        _set_status(props, "Cancelled", note)
        return {"FINISHED"}


class LOCALTEXT3D_OT_randomize_seed(bpy.types.Operator):
    bl_idname = "localtext3d.randomize_seed"
    bl_label = "Randomize Seed"
    bl_description = "Pick a new seed so the next generate is a different variation"

    def execute(self, context):
        get_props(context).seed = random.randint(1, 2_147_483_647)
        return {"FINISHED"}


class LOCALTEXT3D_OT_refresh_history(bpy.types.Operator):
    bl_idname = "localtext3d.refresh_history"
    bl_label = "Refresh Keepers"
    bl_description = "Reload recent finished generations from the worker"

    def execute(self, context):
        _refresh_history(context)
        return {"FINISHED"}


class LOCALTEXT3D_OT_use_keeper(bpy.types.Operator):
    bl_idname = "localtext3d.use_keeper"
    bl_label = "Use Prompt"
    bl_description = "Copy this keeper's prompt, engine, and seed back into the panel"

    index: bpy.props.IntProperty()

    def execute(self, context):
        props = get_props(context)
        if self.index < 0 or self.index >= len(props.history):
            return {"CANCELLED"}
        item = props.history[self.index]
        if item.prompt:
            props.prompt = item.prompt
        if item.engine in {"trellis", "shap_e"}:
            props.engine = item.engine
        props.seed = int(item.seed)
        if item.variant in {"text-base", "text-large", "text-xlarge"}:
            try:
                props.trellis_variant = item.variant
            except TypeError:
                pass
        _set_status(props, "Idle", f"Restored “{item.prompt or item.job_id}”.")
        return {"FINISHED"}


class LOCALTEXT3D_OT_reimport_keeper(bpy.types.Operator):
    bl_idname = "localtext3d.reimport_keeper"
    bl_label = "Reimport"
    bl_description = "Import this keeper again using the current studio settings"
    bl_options = {"REGISTER", "UNDO"}

    index: bpy.props.IntProperty()

    def execute(self, context):
        props = get_props(context)
        prefs = get_prefs(context)
        if self.index < 0 or self.index >= len(props.history):
            return {"CANCELLED"}
        item = props.history[self.index]
        path = client.glb_under_worker_output(item.output_path)
        if not path:
            if not item.job_id:
                self.report({"ERROR"}, "Keeper has no mesh file")
                return {"CANCELLED"}
            dest = _temp_glb_path(item.job_id)
            try:
                path = client.download_job_file(
                    prefs.worker_url,
                    item.job_id,
                    dest,
                    timeout=prefs.download_timeout,
                )
            except client.WorkerError as exc:
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}
        name = _slug(item.prompt or "Text3D")
        meta = {"prompt": item.prompt, "engine": item.engine, "seed": int(item.seed)}
        try:
            _import_glb(context, path, name, props, meta)
        except (OSError, RuntimeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _set_status(props, "Done", f"Reimported {name}")
        self.report({"INFO"}, f"Reimported {name}")
        return {"FINISHED"}


CLASSES = (
    LOCALTEXT3D_OT_check_worker,
    LOCALTEXT3D_OT_generate,
    LOCALTEXT3D_OT_cancel,
    LOCALTEXT3D_OT_randomize_seed,
    LOCALTEXT3D_OT_refresh_history,
    LOCALTEXT3D_OT_use_keeper,
    LOCALTEXT3D_OT_reimport_keeper,
)
