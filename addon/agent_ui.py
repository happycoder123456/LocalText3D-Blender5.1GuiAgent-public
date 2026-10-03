"""N-panel for the Embodied GUI Agent — separate from mesh/GN logic."""

from __future__ import annotations

import bpy

from .agent_props import get_agent_props


class LOCALTEXT3D_PT_agent_panel(bpy.types.Panel):
    bl_label = "GUI Agent"
    bl_idname = "LOCALTEXT3D_PT_agent_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    # Same N-panel tab as mesh/GN so it is visible; code path stays separate.
    bl_category = "Text to 3D"
    # Do NOT use DEFAULT_CLOSED — status redraws were collapsing this panel mid-run.

    def draw(self, context):
        props = get_agent_props(context)
        layout = self.layout

        box = layout.box()
        box.label(text=props.sidecar_summary, icon="INFO")
        row = box.row(align=True)
        row.operator("localtext3d.agent_check", icon="FILE_REFRESH")
        row.operator("localtext3d.agent_start_sidecar", icon="PLAY")

        layout.separator()
        model_row = layout.row(align=True)
        model_row.prop(props, "vision_model", text="Vision")
        model_row.operator("localtext3d.agent_refresh_models", text="", icon="FILE_REFRESH")
        layout.prop(props, "planner_model", text="Planner")

        layout.prop(props, "goal", text="")
        row = layout.row(align=True)
        row.prop(props, "max_steps")
        row.prop(props, "capture_interval_ms", text="Interval ms")
        layout.prop(props, "capture_mode", text="Capture")
        layout.prop(props, "run_mode", text="Run mode")
        pace = layout.row(align=True)
        pace.prop(props, "think_sec", text="Think s")
        pace.prop(props, "settle_sec", text="Settle s")

        ctx_box = layout.box()
        if props.context_ok and props.context_summary:
            ctx_box.label(text=f"Blender: {props.context_summary}", icon="CHECKMARK")
        else:
            ctx_box.label(text="Blender state streams while Recording/Running", icon="INFO")

        layout.separator()
        vid = layout.box()
        vid.label(text="Learn concepts from video", icon="FILE_MOVIE")
        vid.prop(props, "video_url", text="YouTube")
        vid.prop(props, "video_path", text="File")
        vid.prop(props, "learn_minutes", text="Minutes")
        learn_row = vid.row()
        # Allow click even when stuck on Starting… (operator unsticks if sidecar is idle).
        learn_row.enabled = not props.is_running and not props.is_recording
        learn_row.operator("localtext3d.agent_video_learn", icon="IMPORT")
        sync_row = vid.row()
        sync_row.operator("localtext3d.agent_sync_learning", icon="FILE_REFRESH")

        show_learn = bool(props.is_learning_video) or (
            props.learn_phase in {"done", "error"} and props.learn_percent >= 95.0
        ) or (props.status == "Learning") or (
            props.learn_phase == "starting" and props.learn_pulse >= 1
        )
        if show_learn or (props.learn_message and props.learn_percent > 0 and props.learn_elapsed_sec < 600):
            prog = vid.box()
            phase = (props.learn_phase or "working").replace("_", " ").title() or "Working"
            pct = int(max(0, min(100, round(props.learn_percent))))
            filled = max(0, min(10, pct // 10))
            bar = "#" * filled + "-" * (10 - filled)
            prog.label(text=f"Progress  [{bar}]  {pct}%", icon="TIME" if props.is_learning_video else "CHECKMARK")
            prog.label(text=f"Stage: {phase}")
            elapsed = int(props.learn_elapsed_sec or 0)
            prog.label(text=f"Elapsed: {elapsed // 60}m {elapsed % 60:02d}s")
            since = float(props.learn_since_update_sec or 0.0)
            if props.is_learning_video:
                if props.learn_alive and since < 45.0:
                    prog.label(text=f"Live · last update {int(since)}s ago  (pulse {props.learn_pulse})")
                else:
                    prog.label(text=f"Waiting on model · no update {int(since)}s")
            elif props.learn_phase == "done" or pct >= 100:
                prog.label(text="Finished — concepts saved to memory")
            elif props.learn_phase == "error" or props.status == "Error":
                prog.label(text="Stopped with an error — see message")
            msg = props.learn_message or props.status_detail
            if msg:
                for i, chunk in enumerate(_wrap(msg, 40)):
                    if i > 3:
                        break
                    prog.label(text=chunk)
            if props.learn_recent and props.is_learning_video:
                recent = props.learn_recent.split("\n")
                if len(recent) > 1:
                    prog.label(text="Recent steps:")
                    for line in recent[-3:]:
                        if line.strip():
                            prog.label(text="· " + line.strip()[:40])

        if props.concepts_summary or int(getattr(props, "concepts_count", 0) or 0):
            concepts = layout.box()
            n_concepts = int(getattr(props, "concepts_count", 0) or 0)
            names = [p.strip() for p in props.concepts_summary.split("|") if p.strip()]
            if not n_concepts:
                n_concepts = len(names)
            head = concepts.row(align=True)
            head.prop(props, "show_concepts", text="", icon="TRIA_DOWN" if props.show_concepts else "TRIA_RIGHT", emboss=False)
            head.label(text=f"Concepts ({n_concepts})", icon="BOOKMARKS")
            if props.show_concepts:
                show = 80
                for name in names[:show]:
                    for i, chunk in enumerate(_wrap(name, 42)):
                        if i > 1:
                            break
                        concepts.label(text=("  " if i else "• ") + chunk)
                extra = max(0, n_concepts - min(show, len(names)))
                if extra:
                    concepts.label(text=f"… +{extra} more in memory")
            elif names:
                preview = ", ".join(names[:6])
                if n_concepts > 6:
                    preview = f"{preview}…"
                for i, chunk in enumerate(_wrap(preview, 42)):
                    if i > 1:
                        break
                    concepts.label(text=chunk)

        layout.separator()
        rec = layout.row(align=True)
        rec.enabled = not props.is_running and not props.is_learning_video
        if props.is_recording:
            rec.operator("localtext3d.agent_record_stop", icon="PAUSE")
        else:
            rec.operator("localtext3d.agent_record_start", icon="REC")
        if props.learned_note and not props.is_recording:
            learned = layout.box()
            learned.label(text="Learned technique", icon="LIGHT")
            for i, chunk in enumerate(_wrap(props.learned_note, 42)):
                if i > 4:
                    break
                learned.label(text=chunk)

        run = layout.row(align=True)
        if props.run_mode == "replay":
            can_run = bool(props.replay_available)
        else:
            can_run = bool(props.goal.strip())
        run.enabled = (
            not props.is_running
            and not props.is_learning_video
            and can_run
        )
        run.scale_y = 1.4
        run.operator("localtext3d.agent_run", icon="PLAY")
        if props.run_mode != "replay":
            run.operator("localtext3d.agent_plan_preview", text="", icon="VIEWZOOM")
        if props.is_running or props.status in {"Running", "Planning"}:
            layout.operator("localtext3d.agent_stop", icon="CANCEL")

        if props.plan_steps or props.plan_pending:
            plan = layout.box()
            title = "Plan (thinking…)" if props.plan_pending else "Plan"
            plan.label(text=title, icon="PRESET")
            if props.plan_summary:
                for i, chunk in enumerate(_wrap(props.plan_summary, 42)):
                    if i > 2:
                        break
                    plan.label(text=chunk)
            lines = [s for s in props.plan_steps.split("\n") if s.strip()]
            show = 120
            for i, step in enumerate(lines[:show]):
                plan.label(text=f"{i + 1}. {step.strip()[:44]}")
            if len(lines) > show:
                plan.label(text=f"… +{len(lines) - show} more ops (full plan runs)")
            elif lines:
                plan.label(text=f"({len(lines)} ops total)")

        status = layout.box()
        busy = props.is_running or props.is_recording or props.is_learning_video
        if props.status == "Error":
            status_icon = "ERROR"
        elif busy:
            status_icon = "TIME"
        else:
            status_icon = "CHECKMARK"
        status.label(text=props.status, icon=status_icon)
        if props.status_detail:
            # Errors need more room so YouTube failures aren't truncated to "Starting…".
            max_chunks = 8 if props.status == "Error" else 3
            for i, chunk in enumerate(_wrap(props.status_detail, 42)):
                if i > max_chunks:
                    break
                status.label(text=chunk)
        if props.replay_available:
            status.label(text=f"Demo ready: {props.last_session_events} events")
        status.label(text=f"Last reward: {props.last_reward:.3f}")
        status.label(text=f"Memory entries: {props.memory_count}")
        skills_row = status.row(align=True)
        skills_row.prop(props, "show_skills", text="", icon="TRIA_DOWN" if props.show_skills else "TRIA_RIGHT", emboss=False)
        n_skills = int(getattr(props, "skills_count", 0) or 0)
        if not n_skills and props.skills_detail:
            n_skills = len([s for s in props.skills_detail.split("\n") if s.strip()])
        skills_row.label(text=f"Skills known: {n_skills}")
        if props.show_skills and props.skills_detail:
            for line in props.skills_detail.split("\n")[:80]:
                for i, chunk in enumerate(_wrap(line, 42)):
                    if i > 1:
                        break
                    status.label(text=("  " if i else "• ") + chunk)
        status.operator("localtext3d.agent_clear_memory", icon="TRASH")

        tips = layout.box()
        tips.label(text="How it learns", icon="INFO")
        tips.label(text="Record: do it once. It stores the operations")
        tips.label(text="+ values (extrude 1.2…), not mouse pixels.")
        tips.label(text="Video: teaches named techniques the same way.")
        tips.label(text="Run: any goal — it composes skills + knowledge,")
        tips.label(text="varies numbers/axes/modes, checks each op.")
        tips.label(text="Preview (magnifier) shows the plan first.")
        tips.label(text="Restart Start GUI Agent.bat after updates.")

        adv = layout.box()
        adv.label(text="Paths", icon="FILEBROWSER")
        adv.prop(props, "templates_dir")
        if props.dataset_path:
            adv.label(text=props.dataset_path)


def _wrap(text: str, width: int) -> list[str]:
    words = (text or "").split()
    if not words:
        return [""]
    lines: list[str] = []
    cur = words[0]
    for word in words[1:]:
        if len(cur) + 1 + len(word) <= width:
            cur = cur + " " + word
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


CLASSES = (LOCALTEXT3D_PT_agent_panel,)
