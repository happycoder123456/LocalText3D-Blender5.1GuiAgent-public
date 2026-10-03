"""Scene properties for the Embodied GUI Agent (separate from mesh/GN props)."""

from __future__ import annotations

import bpy


# Blender keeps only borrowed references to dynamic enum item strings; the
# callback must hand back the SAME objects each time or the UI can show garbage / crash.
_VISION_ITEMS_CACHE: dict[str, list[tuple[str, str, str]]] = {}
_PLANNER_ITEMS_CACHE: dict[str, list[tuple[str, str, str]]] = {}
_NO_VISION_ITEMS = [("none", "(no models - start Ollama)", "Pull llama3.2-vision (or llava), then refresh")]
_PLANNER_AUTO = ("auto", "Auto (best local text model)", "Picks llama3.1 / qwen2.5 / gemma etc. for planning")
_PLANNER_NONE = ("none", "None (rules + compose only)", "Never call a text model; deterministic plans only")


def _cached_items(cache: dict, key: str, build):
    items = cache.get(key)
    if items is None:
        if len(cache) > 64:
            cache.clear()
        items = build()
        cache[key] = items
    return items


def _vision_model_items(self, _context):
    key = self.vision_models_cache or ""

    def _build():
        models = [name for name in key.split(",") if name]
        return [(name, name, "") for name in models] or list(_NO_VISION_ITEMS)

    return _cached_items(_VISION_ITEMS_CACHE, key, _build)


def _planner_model_items(self, _context):
    key = self.vision_models_cache or ""

    def _build():
        items = [_PLANNER_AUTO, _PLANNER_NONE]
        items.extend((name, name, "") for name in key.split(",") if name)
        return items

    return _cached_items(_PLANNER_ITEMS_CACHE, key, _build)


class LOCALTEXT3D_AgentProps(bpy.types.PropertyGroup):
    goal: bpy.props.StringProperty(
        name="Goal",
        description=(
            "What to do, in your words. Record: names the technique you demonstrate. "
            "Run: anything — the agent composes learned skills + Blender knowledge "
            "(e.g. 'add a cube then extrude the top face by 2')"
        ),
        default="Extrude the selected face",
    )
    sidecar_url: bpy.props.StringProperty(
        name="Sidecar URL",
        description="Local GUI Agent. Must be http://127.0.0.1:8766 (localhost or ::1 on that port only)",
        default="http://127.0.0.1:8766",
    )
    vision_models_cache: bpy.props.StringProperty(default="", options={"HIDDEN"})
    vision_model: bpy.props.EnumProperty(
        name="Vision model",
        description="Your local Ollama vision model (llava, etc.)",
        items=_vision_model_items,
    )
    planner_model: bpy.props.EnumProperty(
        name="Planner",
        description="Local text model that composes learned skills into a plan for new goals",
        items=_planner_model_items,
    )
    plan_summary: bpy.props.StringProperty(default="", options={"HIDDEN"})
    plan_steps: bpy.props.StringProperty(default="", options={"HIDDEN"})
    plan_pending: bpy.props.BoolProperty(default=False, options={"HIDDEN"})
    context_summary: bpy.props.StringProperty(default="", options={"HIDDEN"})
    context_ok: bpy.props.BoolProperty(default=False, options={"HIDDEN"})
    learned_note: bpy.props.StringProperty(default="", options={"HIDDEN"})
    skills_detail: bpy.props.StringProperty(default="", options={"HIDDEN"}, maxlen=8192)
    show_skills: bpy.props.BoolProperty(
        name="Show learned skills",
        description="List the techniques the agent knows (from your recordings, videos and its own runs)",
        default=False,
    )
    capture_mode: bpy.props.EnumProperty(
        name="Capture",
        description="Video is lighter and timestamp-synced; Screenshots is the classic PNG dump",
        items=(
            ("video", "Video", "Continuous window video + timestamped input (default)"),
            ("screenshots", "Screenshots", "Legacy PNG screenshots every interval"),
        ),
        default="video",
    )
    capture_interval_ms: bpy.props.FloatProperty(
        name="Capture interval",
        description="Milliseconds between video frames (FPS ≈ 1000/interval). Mouse path is sampled separately at ~80 Hz",
        default=80.0,
        min=16.0,
        max=2000.0,
    )
    run_mode: bpy.props.EnumProperty(
        name="Run mode",
        description="Learn masters distilled skills (adapt + remember). Replay copies your last recording exactly",
        items=(
            (
                "learn",
                "Learn",
                "Master skills from demos; retarget clicks, remember wins/fails, compose skills (not exact replay)",
            ),
            ("replay", "Replay", "Exactly repeat the last recorded demo"),
        ),
        default="learn",
    )
    think_sec: bpy.props.FloatProperty(
        name="Think",
        description="Learn: pause before each action so you can watch. Replay ignores this and uses recorded timing",
        default=0.2,
        min=0.0,
        max=8.0,
    )
    settle_sec: bpy.props.FloatProperty(
        name="Settle",
        description="Learn: wait after each action so Blender can catch up. Replay uses a short cap so mouse/keys stay snappy",
        default=0.12,
        min=0.0,
        max=8.0,
    )
    max_steps: bpy.props.IntProperty(
        name="Max steps",
        description=(
            "Safety ceiling. Object builds auto-raise this to fit the full plan "
            "(simple ~200, advanced ~500–700, up to 2500)."
        ),
        default=400,
        min=20,
        max=2500,
    )
    templates_dir: bpy.props.StringProperty(
        name="Templates folder",
        description="Optional PNG button templates for OpenCV matching. Leave empty for defaults",
        default="",
        subtype="DIR_PATH",
    )
    dataset_path: bpy.props.StringProperty(
        name="Dataset",
        description="Where recordings and memory are stored",
        default="",
        subtype="DIR_PATH",
    )
    video_url: bpy.props.StringProperty(
        name="YouTube URL",
        description="YouTube tutorial URL to learn modeling concepts from",
        default="",
    )
    video_path: bpy.props.StringProperty(
        name="Video file",
        description="Local tutorial video. The addon copies it into the agent dataset; the sidecar will not open videos elsewhere on disk",
        default="",
        subtype="FILE_PATH",
    )
    learn_minutes: bpy.props.FloatProperty(
        name="Learn minutes",
        description=(
            "Minutes of the tutorial to study (techniques, not every frame). "
            "Up to 1200 (20 hours). Long lessons take much longer to process."
        ),
        default=6.0,
        min=1.0,
        max=1200.0,
    )
    learn_phase: bpy.props.StringProperty(default="", options={"HIDDEN"})
    learn_percent: bpy.props.FloatProperty(default=0.0, options={"HIDDEN"})
    learn_elapsed_sec: bpy.props.FloatProperty(default=0.0, options={"HIDDEN"})
    learn_since_update_sec: bpy.props.FloatProperty(default=0.0, options={"HIDDEN"})
    learn_pulse: bpy.props.IntProperty(default=0, options={"HIDDEN"})
    learn_alive: bpy.props.BoolProperty(default=False, options={"HIDDEN"})
    learn_message: bpy.props.StringProperty(default="", options={"HIDDEN"})
    learn_recent: bpy.props.StringProperty(default="", options={"HIDDEN"})
    is_recording: bpy.props.BoolProperty(default=False, options={"HIDDEN"})
    is_running: bpy.props.BoolProperty(default=False, options={"HIDDEN"})
    is_learning_video: bpy.props.BoolProperty(default=False, options={"HIDDEN"})
    replay_available: bpy.props.BoolProperty(default=False, options={"HIDDEN"})
    last_session_events: bpy.props.IntProperty(default=0, options={"HIDDEN"})
    status: bpy.props.StringProperty(default="Idle")
    status_detail: bpy.props.StringProperty(default="Start the sidecar, then Record or Run Agent.")
    last_reward: bpy.props.FloatProperty(default=0.0)
    memory_count: bpy.props.IntProperty(default=0)
    skills_count: bpy.props.IntProperty(default=0, options={"HIDDEN"})
    skills_summary: bpy.props.StringProperty(default="", options={"HIDDEN"})
    concepts_count: bpy.props.IntProperty(default=0, options={"HIDDEN"})
    concepts_summary: bpy.props.StringProperty(default="", options={"HIDDEN"}, maxlen=8192)
    show_concepts: bpy.props.BoolProperty(
        name="Show learned concepts",
        description="List named techniques extracted from videos (not capped at 24)",
        default=False,
    )
    sidecar_summary: bpy.props.StringProperty(
        default="Sidecar offline. Run scripts\\setup_agent.py then scripts\\start_agent.bat"
    )


def get_agent_props(context) -> LOCALTEXT3D_AgentProps:
    return context.scene.localtext3d_agent


CLASSES = (LOCALTEXT3D_AgentProps,)
