import bpy
from bpy.app.handlers import persistent

from . import agent_context
from . import agent_ops
from . import agent_props
from . import agent_ui
from . import gn_operators
from . import operators
from . import prefs
from . import props
from . import scan
from . import ui

# Transient "busy" flags live in the .blend; after a crash/reload the modal
# operator that owned them is gone, so they must never survive a file load.
_TRANSIENT_MESH_FLAGS = ("is_running", "is_gn_running")
_TRANSIENT_AGENT_FLAGS = ("is_running", "is_recording", "is_learning_video", "plan_pending")


def _reset_transient_flags() -> None:
    for scene in bpy.data.scenes:
        mesh_props = getattr(scene, "localtext3d", None)
        agent_props_ = getattr(scene, "localtext3d_agent", None)
        for group, names in ((mesh_props, _TRANSIENT_MESH_FLAGS), (agent_props_, _TRANSIENT_AGENT_FLAGS)):
            if group is None:
                continue
            for name in names:
                try:
                    if getattr(group, name, False):
                        setattr(group, name, False)
                except Exception:
                    pass


@persistent
def _on_load_post(_dummy=None) -> None:
    try:
        _reset_transient_flags()
    except Exception:
        pass

CLASSES = (
    prefs.LOCALTEXT3D_Preferences,
    props.LOCALTEXT3D_ScanItem,
    props.LOCALTEXT3D_HistoryItem,
    props.LOCALTEXT3D_Props,
    *agent_props.CLASSES,
    *scan.CLASSES,
    *operators.CLASSES,
    *gn_operators.CLASSES,
    *agent_ops.CLASSES,
    *ui.CLASSES,
    *agent_ui.CLASSES,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.localtext3d = bpy.props.PointerProperty(type=props.LOCALTEXT3D_Props)
    bpy.types.Scene.localtext3d_agent = bpy.props.PointerProperty(type=agent_props.LOCALTEXT3D_AgentProps)
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)
    # Reload Scripts kills timers but leaves "Learning" sticky — resync once.
    try:
        bpy.app.timers.register(agent_ops._bootstrap_agent_status, first_interval=0.6)
    except Exception:
        pass


def unregister():
    try:
        _reset_transient_flags()
    except Exception:
        pass
    try:
        bpy.app.handlers.load_post.remove(_on_load_post)
    except (ValueError, Exception):
        pass
    try:
        agent_ops._cancel_agent_poll()
    except Exception:
        pass
    try:
        agent_ops._cancel_learning_watch()
    except Exception:
        pass
    try:
        agent_context.stop()
    except Exception:
        pass
    try:
        if bpy.app.timers.is_registered(agent_ops._bootstrap_agent_status):
            bpy.app.timers.unregister(agent_ops._bootstrap_agent_status)
    except Exception:
        pass
    if hasattr(bpy.types.Scene, "localtext3d_agent"):
        del bpy.types.Scene.localtext3d_agent
    if hasattr(bpy.types.Scene, "localtext3d"):
        del bpy.types.Scene.localtext3d
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
