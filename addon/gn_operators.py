"""Operators for Ollama-driven Geometry Nodes generation."""

from __future__ import annotations

import re
import threading

import bpy

from . import gn_builder
from . import gn_instructions
from . import gn_prompts
from . import gn_spec
from . import llm_client
from .props import get_props


def _set_gn_status(props, status: str, detail: str) -> None:
    props.gn_status = status
    props.gn_status_detail = detail


def _slug(prompt: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", prompt.strip())[:48].strip("_")
    return cleaned or "GN"


def refresh_ollama_models_props(props, *, report_error: bool = True) -> bool:
    try:
        models = llm_client.list_models(timeout=5.0)
    except llm_client.OllamaError as exc:
        props.ollama_models_cache = ""
        if report_error:
            _set_gn_status(props, "Offline", str(exc))
        return False

    props.ollama_models_cache = ",".join(models)
    if models:
        if props.ollama_model not in models:
            props.ollama_model = models[0]
        _set_gn_status(props, "Ready", f"{len(models)} model(s) available.")
    else:
        _set_gn_status(props, "No models", "Run ollama pull <model>, then refresh.")
    return bool(models)


class LOCALTEXT3D_OT_refresh_ollama_models(bpy.types.Operator):
    bl_idname = "localtext3d.refresh_ollama_models"
    bl_label = "Refresh Ollama Models"
    bl_description = "Reload the list of installed Ollama models"

    def execute(self, context):
        props = get_props(context)
        if refresh_ollama_models_props(props, report_error=True):
            self.report({"INFO"}, props.gn_status_detail)
            return {"FINISHED"}
        self.report({"ERROR"}, props.gn_status_detail)
        return {"CANCELLED"}


class LOCALTEXT3D_OT_generate_gn(bpy.types.Operator):
    bl_idname = "localtext3d.generate_gn"
    bl_label = "Generate Geometry Nodes"
    bl_description = "Ask Ollama to write a Geometry Nodes tree from your prompt"
    bl_options = {"REGISTER", "UNDO"}

    _timer = None
    _thread: threading.Thread | None = None
    _result: dict | None = None
    _object_name = "GN"
    _meta: dict | None = None

    def modal(self, context, event):
        props = get_props(context)

        if event.type == "ESC" or (event.type == "TIMER" and not props.is_gn_running):
            self._finish(context)
            _set_gn_status(props, "Cancelled", "Generation stopped.")
            return {"CANCELLED"}

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        if self._thread and self._thread.is_alive():
            return {"PASS_THROUGH"}

        if self._result is None:
            if self._thread is not None and not self._thread.is_alive():
                self._result = dict(self._messages_holder)
            if self._result is None:
                return {"PASS_THROUGH"}

        result = self._result
        self._result = None

        if result.get("error"):
            self._finish(context)
            error = str(result["error"])
            _set_gn_status(props, "Failed", error)
            self.report({"ERROR"}, error)
            return {"CANCELLED"}

        try:
            spec = gn_spec.parse_and_validate(str(result["text"]))
            instructions = gn_spec.format_instructions(spec)
            _set_gn_status(props, "Building", "Creating Geometry Nodes modifier…")
            meta = dict(self._meta or {})
            meta["instructions"] = instructions
            txt_path = gn_instructions.write_instructions_file(
                object_name=self._object_name,
                prompt=str(meta.get("prompt") or ""),
                model=str(meta.get("model") or ""),
                seed=int(meta.get("seed") or 0),
                spec=spec,
                instructions_text=instructions,
            )
            meta["instructions_path"] = txt_path
            gn_builder.apply_to_object(context, spec, self._object_name, meta)
            props.gn_instructions_path = txt_path
        except (gn_spec.GnSpecError, RuntimeError, OSError) as exc:
            self._finish(context)
            _set_gn_status(props, "Failed", str(exc))
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        self._finish(context)
        _set_gn_status(props, "Done", f"Created {self._object_name}. Open the instructions .txt below.")
        self.report({"INFO"}, f"Created {self._object_name}")
        return {"FINISHED"}

    def execute(self, context):
        props = get_props(context)
        if props.is_gn_running:
            self.report({"WARNING"}, "A Geometry Nodes generation is already running")
            return {"CANCELLED"}

        prompt = props.prompt.strip()
        if not prompt:
            _set_gn_status(props, "Need input", "Describe the procedural system you want.")
            self.report({"WARNING"}, "Enter a prompt first")
            return {"CANCELLED"}

        model = props.ollama_model
        if not model or model == "none":
            refresh_ollama_models_props(props, report_error=True)
            model = props.ollama_model
        if not model or model == "none":
            self.report({"ERROR"}, props.gn_status_detail or "No Ollama model selected")
            return {"CANCELLED"}

        self._object_name = _slug(prompt)
        self._meta = {"prompt": prompt, "seed": int(props.seed), "model": model}
        props.gn_instructions_path = ""
        props.is_gn_running = True
        _set_gn_status(props, "Thinking", f"Asking {model}…")

        seed = int(props.seed)
        self._messages_holder: dict = {}

        def worker():
            try:
                messages = gn_prompts.build_messages(prompt)
                text = llm_client.chat(model, messages, seed=seed)
                try:
                    gn_spec.parse_and_validate(text)
                except gn_spec.GnSpecError as first_err:
                    retry_messages = gn_prompts.build_messages(prompt, retry_error=str(first_err))
                    text = llm_client.chat(model, retry_messages, seed=seed)
                    gn_spec.parse_and_validate(text)
                self._messages_holder["text"] = text
            except (llm_client.OllamaError, gn_spec.GnSpecError) as exc:
                self._messages_holder["error"] = str(exc)
            except Exception as exc:
                self._messages_holder["error"] = f"Unexpected error: {exc}"

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
        self._result = None

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.3, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _finish(self, context) -> None:
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        get_props(context).is_gn_running = False
        self._thread = None


class LOCALTEXT3D_OT_open_gn_instructions(bpy.types.Operator):
    bl_idname = "localtext3d.open_gn_instructions"
    bl_label = "Open instructions (.txt)"
    bl_description = "Open the full how-to guide in your default text editor"

    def execute(self, context):
        props = get_props(context)
        path = props.gn_instructions_path.strip()
        try:
            gn_instructions.open_text_file(path)
        except OSError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, "Opened instructions file")
        return {"FINISHED"}


class LOCALTEXT3D_OT_reveal_gn_instructions(bpy.types.Operator):
    bl_idname = "localtext3d.reveal_gn_instructions"
    bl_label = "Show in folder"
    bl_description = "Reveal the instructions .txt file in File Explorer"

    def execute(self, context):
        props = get_props(context)
        path = props.gn_instructions_path.strip()
        try:
            gn_instructions.reveal_in_folder(path)
        except OSError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class LOCALTEXT3D_OT_cancel_gn(bpy.types.Operator):
    bl_idname = "localtext3d.cancel_gn"
    bl_label = "Cancel"
    bl_description = "Stop the current Geometry Nodes generation"

    def execute(self, context):
        props = get_props(context)
        props.is_gn_running = False
        _set_gn_status(props, "Cancelled", "Stopped.")
        return {"FINISHED"}


CLASSES = (
    LOCALTEXT3D_OT_refresh_ollama_models,
    LOCALTEXT3D_OT_generate_gn,
    LOCALTEXT3D_OT_open_gn_instructions,
    LOCALTEXT3D_OT_reveal_gn_instructions,
    LOCALTEXT3D_OT_cancel_gn,
)
