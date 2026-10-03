"""Operators for the Embodied GUI Agent panel."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import bpy

from . import agent_client
from . import agent_context
from . import llm_client
from .agent_props import get_agent_props
from .prefs import get_prefs


def _looks_like_project(path: Path) -> bool:
    return (path / "agent" / "__main__.py").is_file() or (path / ".venv-agent").is_dir()


def _project_root(context=None) -> Path:
    env = os.environ.get("LOCALTEXT3D_ROOT")
    if env:
        candidate = Path(env).expanduser().resolve()
        if _looks_like_project(candidate):
            return candidate

    if context is not None:
        try:
            prefs = get_prefs(context)
            if prefs.project_root.strip():
                candidate = Path(prefs.project_root).expanduser().resolve()
                if _looks_like_project(candidate):
                    return candidate
        except Exception:
            pass

    home = Path.home()
    candidates = [
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent.parent,
        Path.cwd(),
        home / "Downloads" / "LocalText3D" / "LocalText3D",
        home / "Downloads" / "LocalText3D",
        home / "Desktop" / "LocalText3D" / "LocalText3D",
        home / "Desktop" / "LocalText3D",
    ]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if _looks_like_project(resolved):
            return resolved
    return candidates[0].resolve()


def _venv_agent_python(context=None) -> Path | None:
    root = _project_root(context)
    if os.name == "nt":
        candidate = root / ".venv-agent" / "Scripts" / "python.exe"
    else:
        candidate = root / ".venv-agent" / "bin" / "python"
    return candidate if candidate.is_file() else None


def _start_bat(context=None) -> Path | None:
    root = _project_root(context)
    for name in ("Start GUI Agent.bat", "scripts\\start_agent.bat", "scripts/start_agent.bat"):
        path = root / name
        if path.is_file():
            return path
    desktop = Path.home() / "Desktop" / "Start GUI Agent.bat"
    if desktop.is_file():
        return desktop
    return None


def _apply_status(props, payload: dict) -> None:
    props.is_recording = bool(payload.get("recording"))
    props.is_running = bool(payload.get("running"))
    props.is_learning_video = bool(payload.get("learning_video"))
    props.status = str(payload.get("status") or props.status)
    props.status_detail = str(payload.get("detail") or "")
    props.last_reward = float(payload.get("last_reward") or 0.0)
    props.memory_count = int(payload.get("memory_count") or 0)
    if "skills_count" in payload:
        props.skills_count = int(payload.get("skills_count") or 0)
    skills = payload.get("skills")
    if isinstance(skills, list):
        props.skills_summary = ", ".join(str(s) for s in skills[:24])
    elif "skills_count" in payload:
        props.skills_summary = f"{int(payload.get('skills_count') or 0)} saved"
    if "concepts_count" in payload:
        props.concepts_count = int(payload.get("concepts_count") or 0)
    # Always bind before use (avoids UnboundLocalError if skills branch changes).
    _concepts_summary = payload.get("concepts_summary", None)
    if _concepts_summary is not None:
        props.concepts_summary = str(_concepts_summary)
    else:
        concepts = payload.get("concepts")
        if isinstance(concepts, list) and concepts:
            bits = []
            for c in concepts[:200]:
                if not isinstance(c, dict):
                    continue
                name = str(c.get("name") or "").strip()
                if name:
                    bits.append(name)
            if bits:
                props.concepts_summary = " | ".join(bits)
            if not getattr(props, "concepts_count", 0):
                props.concepts_count = len(bits)
    props.replay_available = bool(payload.get("replay_available"))
    props.last_session_events = int(payload.get("last_session_events") or 0)
    # Semantic layer: Blender context, plan preview, learned techniques.
    if "context" in payload:
        props.context_summary = str(payload.get("context") or "")
        props.context_ok = bool(payload.get("context_ok"))
    if "learned_note" in payload:
        props.learned_note = str(payload.get("learned_note") or "")
    preview = payload.get("plan_preview")
    if isinstance(preview, dict):
        steps = preview.get("steps") if isinstance(preview.get("steps"), list) else []
        # Keep enough lines for the N-panel (UI shows up to 120; rest noted as "+N more").
        props.plan_steps = "\n".join(str(s) for s in steps[:240])
        src = str(preview.get("source") or "")
        used = preview.get("used_skills") if isinstance(preview.get("used_skills"), list) else []
        why = str(preview.get("why") or "")
        n_ops = int(preview.get("n_ops") or preview.get("n_steps") or len(steps) or 0)
        bits = []
        if n_ops:
            bits.append(f"{n_ops} ops")
        if src:
            bits.append({"rules": "from skills + knowledge", "llm": "planned by local LLM", "compose": "composed techniques"}.get(src, src))
        if used:
            bits.append("uses: " + ", ".join(str(u) for u in used[:5]))
        if why and src != "rules":
            bits.append(why)
        props.plan_summary = " · ".join(bits)
        props.plan_pending = bool(preview.get("pending"))
    detail = payload.get("skills_detail")
    if isinstance(detail, list):
        lines = []
        for card in detail[:80]:
            if not isinstance(card, dict):
                continue
            name = str(card.get("name") or "").strip()
            if not name:
                continue
            origin = {"recorded": "you", "video": "video", "planned": "agent"}.get(str(card.get("origin") or ""), "")
            summary = str(card.get("summary") or "").strip()
            line = f"{name} [{origin}]" if origin else name
            if summary:
                line = f"{line}: {summary}"
            lines.append(line)
        props.skills_detail = "\n".join(lines)
    # Nested recorder status also carries replay fields.
    rec = payload.get("recorder")
    if isinstance(rec, dict):
        if "replay_available" in rec:
            props.replay_available = bool(rec.get("replay_available"))
        if "last_session_events" in rec:
            props.last_session_events = int(rec.get("last_session_events") or 0)
    dataset = payload.get("dataset") or ""
    if dataset:
        props.dataset_path = str(dataset)
    models = payload.get("models")
    if isinstance(models, list) and models:
        props.vision_models_cache = ",".join(str(m) for m in models)
        preferred = str(payload.get("preferred_model") or "")
        if preferred and preferred in models:
            props.vision_model = preferred
        elif props.vision_model not in models:
            props.vision_model = models[0]
    err = payload.get("error") or ""
    if err:
        props.status_detail = str(err)
    _apply_learning_progress(props, payload)


def _apply_learning_progress(props, payload: dict) -> None:
    """Apply learning_progress safely — never let missing RNA props kill the poll timer."""
    progress = payload.get("learning_progress")
    if not isinstance(progress, dict):
        if not payload.get("learning_video") and str(payload.get("status") or "") != "Learning":
            try:
                props.learn_alive = False
            except Exception:
                pass
        return
    try:
        props.learn_phase = str(progress.get("phase") or "")
        props.learn_percent = float(progress.get("percent") or 0.0)
        props.learn_elapsed_sec = float(progress.get("elapsed_sec") or 0.0)
        props.learn_since_update_sec = float(progress.get("since_update_sec") or 0.0)
        props.learn_pulse = int(progress.get("pulse") or 0)
        props.learn_alive = bool(progress.get("alive"))
        props.learn_message = str(progress.get("message") or "")
        recent = progress.get("recent")
        if isinstance(recent, list):
            props.learn_recent = "\n".join(str(x) for x in recent[-5:] if str(x).strip())
        else:
            props.learn_recent = ""
        # If sidecar finished, never leave the Learn button disabled.
        if not payload.get("learning_video") and str(payload.get("status") or "") != "Learning":
            props.is_learning_video = False
            if props.learn_phase == "done" or props.learn_percent >= 100.0:
                detail = str(payload.get("detail") or "")
                if detail:
                    props.status_detail = detail
                    props.learn_message = detail[:240]
    except Exception:
        # Older Blender sessions without the new props still keep basic status working.
        pass


def refresh_vision_models_props(props, *, report_error: bool = True) -> bool:
    try:
        models = llm_client.list_models(timeout=5.0)
    except llm_client.OllamaError as exc:
        props.vision_models_cache = ""
        if report_error:
            props.status = "Ollama offline"
            props.status_detail = str(exc)
        return False
    props.vision_models_cache = ",".join(models)
    if models:
        if props.vision_model not in models:
            # Prefer vision-ish names.
            pick = models[0]
            for name in models:
                lower = name.lower()
                if any(k in lower for k in ("llama3.2-vision", "llama3.2", "llava", "vision", "qwen2.5vl", "moondream", "bakllava")):
                    pick = name
                    break
            props.vision_model = pick
        props.status_detail = f"{len(models)} Ollama model(s) available."
    else:
        props.status = "No models"
        props.status_detail = "Run ollama pull llama3.2-vision, then refresh."
    return bool(models)


class LOCALTEXT3D_OT_agent_check(bpy.types.Operator):
    bl_idname = "localtext3d.agent_check"
    bl_label = "Check GUI Agent"
    bl_description = "Ping the local GUI Agent sidecar on port 8766"

    def execute(self, context):
        props = get_agent_props(context)
        try:
            # Prefer /status so Learning/memory/skills match the sidecar after a stuck poll.
            payload = agent_client.status(props.sidecar_url, timeout=3.0)
        except agent_client.AgentClientError:
            try:
                payload = agent_client.health(props.sidecar_url)
            except agent_client.AgentClientError as exc:
                props.sidecar_summary = "Sidecar offline"
                props.status = "Offline"
                props.status_detail = str(exc)
                props.is_learning_video = False
                props.is_running = False
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}
        _apply_status(props, payload)
        # Heal sticky "Learning / Starting…" left behind when a poll timer died (e.g. Reload Scripts).
        if not payload.get("learning_video") and str(payload.get("status") or "") not in {
            "Learning",
            "Planning",
            "Running",
        }:
            props.is_learning_video = False
            props.is_running = False
        ollama_ok = payload.get("ollama_ok")
        if ollama_ok is None:
            try:
                health = agent_client.health(props.sidecar_url, timeout=2.0)
                ollama_ok = health.get("ollama_ok")
            except agent_client.AgentClientError:
                ollama_ok = False
        ollama = "Ollama OK" if ollama_ok else "Ollama offline"
        props.sidecar_summary = f"Sidecar online — {ollama}"
        refresh_vision_models_props(props, report_error=False)
        _resume_poll_if_busy(props)
        if props.is_learning_video or str(props.status or "") == "Learning" or props.learn_phase == "starting":
            _start_learning_watch()
        self.report({"INFO"}, props.sidecar_summary)
        return {"FINISHED"}


class LOCALTEXT3D_OT_agent_start_sidecar(bpy.types.Operator):
    bl_idname = "localtext3d.agent_start_sidecar"
    bl_label = "Start Sidecar"
    bl_description = "Spawn .venv-agent python -m agent serve in a new console"

    def execute(self, context):
        props = get_agent_props(context)
        root = _project_root(context)
        py = _venv_agent_python(context)
        try:
            agent_client.health(props.sidecar_url, timeout=1.5)
            props.sidecar_summary = "Sidecar already online"
            self.report({"INFO"}, props.sidecar_summary)
            return {"FINISHED"}
        except agent_client.AgentClientError:
            pass

        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)

        if py is not None:
            try:
                subprocess.Popen(
                    [str(py), "-m", "agent", "serve"],
                    cwd=str(root),
                    creationflags=creationflags,
                )
            except OSError as exc:
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}
        else:
            bat = _start_bat(context)
            if bat is None:
                msg = (
                    "Missing .venv-agent. Set Preferences → Local Text to 3D → "
                    "LocalText3D project folder to the folder that contains "
                    "'Start GUI Agent.bat', run 'Setup GUI Agent.bat' there once, "
                    "or double-click 'Start GUI Agent.bat' yourself."
                )
                props.status = "Setup needed"
                props.status_detail = msg
                self.report({"ERROR"}, msg)
                return {"CANCELLED"}
            try:
                subprocess.Popen(
                    ["cmd.exe", "/c", str(bat)],
                    cwd=str(bat.parent),
                    creationflags=creationflags,
                )
            except OSError as exc:
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}

        # Remember a working project path in prefs when we can.
        try:
            prefs = get_prefs(context)
            if not prefs.project_root.strip() and _looks_like_project(root):
                prefs.project_root = str(root)
        except Exception:
            pass

        props.sidecar_summary = "Sidecar starting…"
        props.status_detail = f"From {root}. Wait a second, then click Check."
        self.report({"INFO"}, "Sidecar process launched")
        return {"FINISHED"}


class LOCALTEXT3D_OT_agent_refresh_models(bpy.types.Operator):
    bl_idname = "localtext3d.agent_refresh_models"
    bl_label = "Refresh Vision Models"
    bl_description = "Reload installed Ollama models for the GUI Agent"

    def execute(self, context):
        props = get_agent_props(context)
        if refresh_vision_models_props(props, report_error=True):
            self.report({"INFO"}, props.status_detail)
            return {"FINISHED"}
        self.report({"ERROR"}, props.status_detail)
        return {"CANCELLED"}


class LOCALTEXT3D_OT_agent_record_start(bpy.types.Operator):
    bl_idname = "localtext3d.agent_record_start"
    bl_label = "Record"
    bl_description = "Start the Teacher recorder (video or screenshots + mouse/keyboard)"

    def execute(self, context):
        props = get_agent_props(context)
        goal = (props.goal or "").strip()
        if not goal or goal.lower() in {"teacher demo", "demo"}:
            self.report(
                {"ERROR"},
                "Set a Goal first (e.g. Extrude face), then click Record",
            )
            props.status = "Need goal"
            props.status_detail = "Type what you are about to demonstrate in the Goal box."
            return {"CANCELLED"}
        # Fresh operator baseline so only what you do from now on is learned.
        agent_context.reset_op_baseline()
        try:
            payload = agent_client.record_start(
                props.sidecar_url,
                interval_ms=props.capture_interval_ms,
                capture_mode=props.capture_mode or "video",
                goal=goal,
            )
        except agent_client.AgentClientError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _apply_status(props, payload)
        props.status = "Recording"
        props.is_recording = True
        # Stream Blender state + executed operators so the demo is understood as
        # operations with parameters (not pixel positions).
        agent_context.post_now(props.sidecar_url)
        agent_context.ensure_running(props.sidecar_url)
        mode = str(payload.get("recorder", {}).get("capture_mode") or props.capture_mode or "video")
        self.report({"INFO"}, f"Recording '{goal}' ({mode}) — do it once in the viewport, then Stop")
        return {"FINISHED"}


class LOCALTEXT3D_OT_agent_record_stop(bpy.types.Operator):
    bl_idname = "localtext3d.agent_record_stop"
    bl_label = "Stop Recording"
    bl_description = "Stop the Teacher recorder and save demo steps into memory"

    def execute(self, context):
        props = get_agent_props(context)
        goal = (props.goal or "").strip()
        if not goal or goal.lower() in {"teacher demo", "demo"}:
            self.report(
                {"ERROR"},
                "Set the Goal name before Stop (same text you will use for Learn)",
            )
            return {"CANCELLED"}
        # Flush the last operators (e.g. your final Tab) before the sidecar analyses the demo.
        agent_context.post_now(props.sidecar_url)
        try:
            payload = agent_client.record_stop(props.sidecar_url, goal=goal, timeout=20.0)
        except agent_client.AgentClientError as exc:
            self.report({"ERROR"}, str(exc)[:180])
            return {"CANCELLED"}
        _apply_status(props, payload)
        props.is_recording = False
        if str(payload.get("status") or "") == "Need goal":
            self.report({"ERROR"}, str(payload.get("detail") or "Need goal")[:180])
            return {"CANCELLED"}
        note = str(payload.get("learned_note") or "")
        self.report({"INFO"}, f"Learned '{goal}'" + (f": {note[:120]}" if note else ""))
        return {"FINISHED"}


_agent_poll_handle = None
_learning_watch_handle = None


def _cancel_agent_poll():
    global _agent_poll_handle
    if _agent_poll_handle is not None:
        try:
            bpy.app.timers.unregister(_agent_poll_handle)
        except Exception:
            pass
        _agent_poll_handle = None


def _cancel_learning_watch():
    global _learning_watch_handle
    if _learning_watch_handle is not None:
        try:
            bpy.app.timers.unregister(_learning_watch_handle)
        except Exception:
            pass
        _learning_watch_handle = None


def _start_learning_watch():
    """Dedicated poll for Learn-from-Video so a failed agent poll can't freeze the UI."""
    global _learning_watch_handle
    _cancel_learning_watch()
    _learning_watch_handle = _learning_watch_tick
    bpy.app.timers.register(_learning_watch_tick, first_interval=0.25)


def _learning_watch_tick():
    """Keep Progress live until the sidecar leaves Learning — heals sticky Starting…."""
    global _learning_watch_handle
    try:
        window = bpy.context.window
        screen = window.screen if window else None
        scene = screen.scene if screen else bpy.context.scene
        if scene is None or not hasattr(scene, "localtext3d_agent"):
            _learning_watch_handle = None
            return None
        props = scene.localtext3d_agent
        try:
            payload = agent_client.status(props.sidecar_url, timeout=2.5)
        except agent_client.AgentClientError as exc:
            props.status = "Offline"
            props.status_detail = str(exc)[:200]
            props.is_learning_video = False
            props.learn_alive = False
            props.learn_phase = "error"
            _learning_watch_handle = None
            _tag_view3d_redraw()
            return None
        _apply_status(props, payload)
        learning = bool(payload.get("learning_video")) or str(payload.get("status") or "") == "Learning"
        if learning:
            props.is_learning_video = True
            _tag_view3d_redraw()
            return 0.6
        # Sidecar done (or never started) — unstick the button and show final detail.
        props.is_learning_video = False
        if str(payload.get("status") or "") == "Error":
            props.learn_phase = "error"
        elif float(getattr(props, "learn_percent", 0) or 0) >= 95 or str(
            getattr(props, "learn_phase", "") or ""
        ) == "done":
            props.learn_phase = "done"
            props.learn_percent = max(float(props.learn_percent or 0), 100.0)
            props.learn_alive = False
        else:
            # Heal "Starting…" seed when sidecar is already Idle from a finished run.
            detail = str(payload.get("detail") or "")
            if detail and ("Saved" in detail or "concept" in detail.lower()):
                props.learn_phase = "done"
                props.learn_percent = 100.0
                props.learn_message = detail[:240]
                props.status_detail = detail
                props.learn_alive = False
            elif props.learn_phase == "starting" and props.learn_pulse <= 1:
                props.is_learning_video = False
                props.learn_phase = ""
                props.learn_alive = False
                props.status = str(payload.get("status") or "Idle")
                props.status_detail = detail or "Sidecar idle — click Learn from Video again."
        _tag_view3d_redraw()
        _learning_watch_handle = None
        return None
    except Exception:
        _learning_watch_handle = None
        return None


def _resume_poll_if_busy(props) -> None:
    """Restart the status poll when Learning/Running is active (or UI thinks it is)."""
    global _agent_poll_handle
    busy = bool(
        getattr(props, "is_running", False)
        or getattr(props, "is_learning_video", False)
        or str(getattr(props, "status", "") or "") in {"Learning", "Planning", "Running"}
    )
    if not busy:
        return
    if _agent_poll_handle is not None and bpy.app.timers.is_registered(_agent_poll_handle):
        return
    _agent_poll_handle = _poll_agent_status
    bpy.app.timers.register(_poll_agent_status, first_interval=0.4)


def _bootstrap_agent_status():
    """One-shot after addon register/reload: sync props and resume a lost Learning poll."""
    try:
        window = bpy.context.window
        screen = window.screen if window else None
        scene = screen.scene if screen else bpy.context.scene
        if scene is None or not hasattr(scene, "localtext3d_agent"):
            return None
        props = scene.localtext3d_agent
        try:
            payload = agent_client.status(props.sidecar_url, timeout=2.0)
        except agent_client.AgentClientError:
            return None
        _apply_status(props, payload)
        if not payload.get("learning_video") and str(payload.get("status") or "") not in {
            "Learning",
            "Planning",
            "Running",
        }:
            props.is_learning_video = False
            props.is_running = False
        _resume_poll_if_busy(props)
        # Heal sticky Learning UI after Reload Scripts.
        if (
            props.is_learning_video
            or str(props.status or "") == "Learning"
            or str(getattr(props, "learn_phase", "") or "") == "starting"
        ):
            _start_learning_watch()
    except Exception:
        pass
    return None


def _poll_agent_status():
    """Timer callback: refresh GUI Agent props without a modal (ESC-safe)."""
    global _agent_poll_handle
    try:
        window = bpy.context.window
        screen = window.screen if window else None
        scene = screen.scene if screen else bpy.context.scene
        if scene is None or not hasattr(scene, "localtext3d_agent"):
            _agent_poll_handle = None
            return None
        props = scene.localtext3d_agent
        try:
            payload = agent_client.status(props.sidecar_url, timeout=2.0)
        except agent_client.AgentClientError as exc:
            props.is_running = False
            props.is_learning_video = False
            props.status = "Offline"
            props.status_detail = str(exc)[:200]
            _agent_poll_handle = None
            return None
        _apply_status(props, payload)
        preview = payload.get("plan_preview")
        plan_pending = bool(isinstance(preview, dict) and preview.get("pending"))
        status = str(payload.get("status") or "")
        # Keep polling through Learning so "Starting…" is replaced by progress/Error.
        if (
            payload.get("running")
            or payload.get("learning_video")
            or plan_pending
            or status in {"Learning", "Planning", "Running"}
        ):
            _tag_view3d_redraw()
            return 0.8
        props.is_running = False
        props.is_learning_video = False
        _tag_view3d_redraw()
        _agent_poll_handle = None
        return None
    except Exception:
        # Never leave the N-panel stuck on Learning with the button disabled.
        try:
            window = bpy.context.window
            screen = window.screen if window else None
            scene = screen.scene if screen else bpy.context.scene
            if scene is not None and hasattr(scene, "localtext3d_agent"):
                scene.localtext3d_agent.is_learning_video = False
                scene.localtext3d_agent.is_running = False
        except Exception:
            pass
        _agent_poll_handle = None
        return None


def _tag_view3d_redraw() -> None:
    """Force N-panel redraw so Learning progress updates without mouse move."""
    try:
        wm = bpy.context.window_manager
        if wm is None:
            return
        for window in wm.windows:
            screen = window.screen
            if screen is None:
                continue
            for area in screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
    except Exception:
        pass


class LOCALTEXT3D_OT_agent_run(bpy.types.Operator):
    bl_idname = "localtext3d.agent_run"
    bl_label = "Run Agent"
    bl_description = (
        "Learn: follow your recorded demo; on failure recover from teacher memory only. "
        "Replay: exact copy of the last recording."
    )

    def execute(self, context):
        global _agent_poll_handle
        props = get_agent_props(context)
        if props.is_running:
            self.report({"WARNING"}, "Agent is already running")
            return {"CANCELLED"}
        run_mode = props.run_mode or "learn"
        can_replay = bool(props.replay_available)
        if run_mode == "replay":
            if not can_replay:
                self.report({"ERROR"}, "No recording to replay. Record a demo first.")
                return {"CANCELLED"}
        elif not props.goal.strip():
            self.report({"ERROR"}, "Enter a goal, then Record a demo for Learn mode")
            return {"CANCELLED"}
        model = props.vision_model
        if not model or model == "none":
            refresh_vision_models_props(props, report_error=False)
            model = props.vision_model
        # The planner needs to know the scene (mode, selection) before it starts.
        agent_context.reset_op_baseline()
        agent_context.post_now(props.sidecar_url)
        try:
            payload = agent_client.agent_start(
                props.sidecar_url,
                goal=props.goal or "Replay last recording",
                model=model if model and model != "none" else "none",
                max_steps=props.max_steps,
                templates_dir=bpy.path.abspath(props.templates_dir) if props.templates_dir else "",
                run_mode=run_mode,
                think_sec=props.think_sec,
                settle_sec=props.settle_sec,
                planner_model=props.planner_model or "auto",
            )
        except agent_client.AgentClientError as exc:
            self.report({"ERROR"}, str(exc)[:180])
            return {"CANCELLED"}
        _apply_status(props, payload)
        props.is_running = True
        agent_context.ensure_running(props.sidecar_url)
        _cancel_agent_poll()
        _agent_poll_handle = _poll_agent_status
        bpy.app.timers.register(_poll_agent_status, first_interval=0.5)
        mode = str(payload.get("mode") or "")
        if mode == "replay":
            self.report({"INFO"}, "Replaying last recording — click Stop Agent to cancel")
        elif mode == "planning":
            self.report({"INFO"}, "Planning with learned skills — Stop to cancel")
        elif mode == "master":
            self.report({"INFO"}, "Running plan (skills + knowledge) — Stop to cancel")
        else:
            self.report({"INFO"}, "Agent running — click Stop Agent to cancel")
        return {"FINISHED"}


class LOCALTEXT3D_OT_agent_plan_preview(bpy.types.Operator):
    bl_idname = "localtext3d.agent_plan_preview"
    bl_label = "Preview Plan"
    bl_description = (
        "Show what the agent would do for this Goal — composed from the skills you "
        "taught it, video concepts and Blender knowledge — without touching the scene"
    )

    def execute(self, context):
        global _agent_poll_handle
        props = get_agent_props(context)
        goal = (props.goal or "").strip()
        if not goal:
            self.report({"ERROR"}, "Type a Goal first")
            return {"CANCELLED"}
        agent_context.post_now(props.sidecar_url)
        try:
            payload = agent_client.plan_preview(
                props.sidecar_url,
                goal=goal,
                model=props.vision_model if props.vision_model != "none" else "",
                planner_model=props.planner_model or "auto",
            )
        except agent_client.AgentClientError as exc:
            if getattr(exc, "status", None) == 404:
                msg = "Sidecar is an old version — restart Start GUI Agent.bat"
            else:
                msg = str(exc)[:180]
            self.report({"ERROR"}, msg)
            props.status_detail = msg
            return {"CANCELLED"}
        _apply_status(props, payload)
        if props.plan_pending:
            # LLM is still composing; poll until it lands.
            _cancel_agent_poll()
            _agent_poll_handle = _poll_agent_status
            bpy.app.timers.register(_poll_agent_status, first_interval=1.0)
            self.report({"INFO"}, "Rules gave a partial plan; asking the local LLM…")
        elif props.plan_steps:
            self.report({"INFO"}, "Plan ready — see the Plan box")
        else:
            self.report({"WARNING"}, "No plan: teach this by Record, or pick a Planner model")
        return {"FINISHED"}


class LOCALTEXT3D_OT_agent_stop(bpy.types.Operator):
    bl_idname = "localtext3d.agent_stop"
    bl_label = "Stop Agent"
    bl_description = "Stop the embodied agent loop (sidecar stays alive)"

    def execute(self, context):
        props = get_agent_props(context)
        _cancel_agent_poll()
        try:
            payload = agent_client.agent_stop(props.sidecar_url)
        except agent_client.AgentClientError as exc:
            # Sidecar unreachable: it cannot be running anything for us, so free
            # the panel instead of leaving Run disabled with no poller to heal it.
            props.is_running = False
            props.status = "Offline"
            props.status_detail = str(exc)[:200]
            self.report({"ERROR"}, str(exc)[:180])
            return {"CANCELLED"}
        _apply_status(props, payload)
        props.is_running = False
        self.report({"INFO"}, "Agent stopped")
        return {"FINISHED"}


class LOCALTEXT3D_OT_agent_video_learn(bpy.types.Operator):
    bl_idname = "localtext3d.agent_video_learn"
    bl_label = "Learn from Video"
    bl_description = (
        "Watch a YouTube or local tutorial and save reusable modeling concepts "
        "(not a mouse replay). Then select your mesh and Run Agent"
    )

    _timer = None
    _ticks = 0

    def execute(self, context):
        props = get_agent_props(context)
        # If the panel is stuck on Starting… but the sidecar is idle, clear and continue.
        if props.is_running or props.is_recording:
            self.report({"WARNING"}, "Stop recording/agent before learning from video")
            return {"CANCELLED"}
        if props.is_learning_video:
            try:
                live = agent_client.status(props.sidecar_url, timeout=2.0)
            except agent_client.AgentClientError:
                live = {}
            if live.get("learning_video") or str(live.get("status") or "") == "Learning":
                # Already learning on sidecar — attach a live progress modal.
                _apply_status(props, live)
                props.is_learning_video = True
                return self._begin_modal(context)
            props.is_learning_video = False
            props.learn_alive = False

        url = (props.video_url or "").strip()
        path = (props.video_path or "").strip()
        if path:
            # File browser may store a blend-relative "//clip.mp4"; the sidecar
            # is a separate process and needs an absolute path inside the dataset.
            path = bpy.path.abspath(path)
            dataset = str(props.dataset_path or "").strip()
            if not dataset:
                try:
                    live = agent_client.status(props.sidecar_url, timeout=3.0)
                except agent_client.AgentClientError:
                    live = {}
                dataset = str(live.get("dataset") or "").strip()
                if dataset:
                    props.dataset_path = dataset
            if not dataset:
                dataset = str(agent_client.default_agent_dataset_dir())
            try:
                path = agent_client.stage_video_into_dataset(path, dataset)
            except agent_client.AgentClientError as exc:
                props.status = "Error"
                props.status_detail = str(exc)[:240]
                self.report({"ERROR"}, str(exc)[:180])
                return {"CANCELLED"}
        if not url and not path:
            self.report({"ERROR"}, "Paste a YouTube URL or choose a video file")
            return {"CANCELLED"}
        model = props.vision_model
        if not model or model == "none":
            refresh_vision_models_props(props, report_error=False)
            model = props.vision_model
        try:
            payload = agent_client.video_learn(
                props.sidecar_url,
                url=url,
                path=path,
                goal=props.goal or "",
                model=model if model and model != "none" else "llama3.2-vision",
                max_minutes=float(props.learn_minutes or 6.0),
            )
        except agent_client.AgentClientError as exc:
            props.is_learning_video = False
            props.status = "Error"
            props.status_detail = str(exc)[:240]
            props.learn_phase = "error"
            props.learn_message = props.status_detail
            self.report({"ERROR"}, str(exc)[:180])
            return {"CANCELLED"}
        _apply_status(props, payload)
        props.is_learning_video = True
        props.status = "Learning"
        props.learn_phase = "starting"
        props.learn_percent = max(2.0, float(getattr(props, "learn_percent", 0) or 0))
        props.learn_elapsed_sec = 0.0
        props.learn_since_update_sec = 0.0
        props.learn_alive = True
        props.learn_pulse = max(1, int(getattr(props, "learn_pulse", 0) or 0))
        props.learn_message = str(payload.get("detail") or "Starting video teacher…")[:240]
        props.status_detail = props.learn_message
        self.report({"INFO"}, "Learning — Progress updates below every ~0.5s")
        return self._begin_modal(context)

    def _begin_modal(self, context):
        wm = context.window_manager
        self._ticks = 0
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        _tag_view3d_redraw()
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        props = get_agent_props(context)
        self._ticks += 1
        try:
            payload = agent_client.status(props.sidecar_url, timeout=2.5)
        except agent_client.AgentClientError as exc:
            props.status = "Offline"
            props.status_detail = str(exc)[:200]
            props.is_learning_video = False
            props.learn_phase = "error"
            props.learn_alive = False
            props.learn_message = props.status_detail
            self._finish_timer(context)
            _tag_view3d_redraw()
            self.report({"ERROR"}, props.status_detail)
            return {"CANCELLED"}

        _apply_status(props, payload)
        learning = bool(payload.get("learning_video")) or str(payload.get("status") or "") == "Learning"
        # Client-side elapsed fallback so the clock never freezes at 0m 00s.
        prog = payload.get("learning_progress") if isinstance(payload.get("learning_progress"), dict) else {}
        elapsed = float(prog.get("elapsed_sec") or 0.0)
        if elapsed <= 0.0:
            props.learn_elapsed_sec = float(self._ticks) * 0.5
        if learning:
            props.is_learning_video = True
            if int(prog.get("pulse") or 0) > 0:
                props.learn_pulse = int(prog.get("pulse") or props.learn_pulse)
            _tag_view3d_redraw()
            return {"PASS_THROUGH"}

        # Finished (or never started).
        props.is_learning_video = False
        props.learn_alive = False
        detail = str(payload.get("detail") or "")
        if str(payload.get("status") or "") == "Error":
            props.learn_phase = "error"
            props.status = "Error"
            props.learn_message = detail or props.status_detail
            self.report({"ERROR"}, (detail or "Video learning failed")[:180])
        else:
            props.learn_phase = "done"
            props.learn_percent = 100.0
            if detail:
                props.status_detail = detail
                props.learn_message = detail[:240]
            self.report({"INFO"}, (detail or "Video learning finished")[:180])
        self._finish_timer(context)
        _tag_view3d_redraw()
        return {"FINISHED"}

    def _finish_timer(self, context):
        if self._timer is not None:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None

    def cancel(self, context):
        # Operator.cancel must return None (Blender ignores/complains about a set).
        self._finish_timer(context)
        props = get_agent_props(context)
        props.is_learning_video = False


class LOCALTEXT3D_OT_agent_sync_learning(bpy.types.Operator):
    bl_idname = "localtext3d.agent_sync_learning"
    bl_label = "Refresh Progress"
    bl_description = "Pull live video-learning status from the sidecar (fixes stuck Starting…)"

    def execute(self, context):
        props = get_agent_props(context)
        try:
            payload = agent_client.status(props.sidecar_url, timeout=3.0)
        except agent_client.AgentClientError as exc:
            props.status = "Offline"
            props.status_detail = str(exc)[:200]
            props.is_learning_video = False
            self.report({"ERROR"}, str(exc)[:160])
            return {"CANCELLED"}
        _apply_status(props, payload)
        learning = bool(payload.get("learning_video")) or str(payload.get("status") or "") == "Learning"
        if not learning:
            props.is_learning_video = False
            props.learn_alive = False
            detail = str(payload.get("detail") or "")
            prog = payload.get("learning_progress") if isinstance(payload.get("learning_progress"), dict) else {}
            if str(prog.get("phase") or "") == "done" or "Saved" in detail:
                props.learn_phase = "done"
                props.learn_percent = 100.0
                props.learn_message = detail[:240] if detail else "Learning finished"
                props.status_detail = props.learn_message
            elif props.learn_phase == "starting":
                props.learn_phase = ""
                props.learn_message = detail or "Sidecar idle — click Learn from Video again"
                props.status_detail = props.learn_message
            self.report({"INFO"}, props.status_detail[:160] or "Synced")
        else:
            props.is_learning_video = True
            # Hand off to the modal so Progress keeps moving.
            bpy.ops.localtext3d.agent_video_learn()
            self.report({"INFO"}, "Sidecar is learning — attached live progress")
        _tag_view3d_redraw()
        return {"FINISHED"}


class LOCALTEXT3D_OT_agent_clear_memory(bpy.types.Operator):
    bl_idname = "localtext3d.agent_clear_memory"
    bl_label = "Clear Memory"
    bl_description = "Delete cross-session success/mistake memory on this PC"

    def execute(self, context):
        props = get_agent_props(context)
        try:
            payload = agent_client.memory_clear(props.sidecar_url)
        except agent_client.AgentClientError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _apply_status(props, payload)
        props.memory_count = 0
        self.report({"INFO"}, "Memory cleared")
        return {"FINISHED"}


class LOCALTEXT3D_OT_agent_create_modelfile(bpy.types.Operator):
    bl_idname = "localtext3d.agent_create_modelfile"
    bl_label = "Create blender-gui"
    bl_description = "Create a local Ollama alias FROM your vision model with Blender system prompt"

    def execute(self, context):
        props = get_agent_props(context)
        base = props.vision_model
        if not base or base == "none":
            self.report({"ERROR"}, "Select a base vision model first")
            return {"CANCELLED"}
        try:
            payload = agent_client.create_modelfile(
                props.sidecar_url,
                base_model=base,
                alias="blender-gui",
            )
        except agent_client.AgentClientError as exc:
            msg = str(exc)[:180]
            props.status = "Error"
            props.status_detail = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}
        refresh_vision_models_props(props, report_error=False)
        name = payload.get("model") or "blender-gui"
        if name in (props.vision_models_cache or "").split(","):
            props.vision_model = name
        self.report({"INFO"}, f"Created Ollama model {name} (same weights + Blender instructions)")
        return {"FINISHED"}


CLASSES = (
    LOCALTEXT3D_OT_agent_check,
    LOCALTEXT3D_OT_agent_start_sidecar,
    LOCALTEXT3D_OT_agent_refresh_models,
    LOCALTEXT3D_OT_agent_record_start,
    LOCALTEXT3D_OT_agent_record_stop,
    LOCALTEXT3D_OT_agent_run,
    LOCALTEXT3D_OT_agent_plan_preview,
    LOCALTEXT3D_OT_agent_stop,
    LOCALTEXT3D_OT_agent_video_learn,
    LOCALTEXT3D_OT_agent_sync_learning,
    LOCALTEXT3D_OT_agent_clear_memory,
    LOCALTEXT3D_OT_agent_create_modelfile,
)
