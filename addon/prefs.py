import bpy


class LOCALTEXT3D_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    worker_url: bpy.props.StringProperty(
        name="Worker URL",
        description="Local worker. Must be http://127.0.0.1:8765 (localhost or ::1 on that port only)",
        default="http://127.0.0.1:8765",
    )
    poll_interval: bpy.props.FloatProperty(
        name="Poll Interval",
        description="How often to ask the worker for job progress",
        default=0.6,
        min=0.2,
        max=5.0,
        # Seconds, not scene frames: unit="TIME" would display frame counts.
        unit="TIME_ABSOLUTE",
    )
    request_timeout: bpy.props.FloatProperty(
        name="Request Timeout",
        description="Seconds to wait for a single HTTP call",
        default=15.0,
        min=2.0,
        max=120.0,
    )
    download_timeout: bpy.props.FloatProperty(
        name="Download Timeout",
        description="Seconds to wait when downloading the finished GLB",
        default=120.0,
        min=10.0,
        max=600.0,
    )
    library_path: bpy.props.StringProperty(
        name="Megascans / Fab folder",
        description="Local Quixel Bridge, Megascans Library, or Fab download folder. Leave empty to search common folders automatically",
        default="",
        subtype="DIR_PATH",
    )
    project_root: bpy.props.StringProperty(
        name="LocalText3D project folder",
        description="Folder that contains .venv-agent and the agent package (your LocalText3D repo). Used to Start Sidecar from Blender",
        default="",
        subtype="DIR_PATH",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "worker_url")
        row = layout.row(align=True)
        row.prop(self, "poll_interval")
        row.prop(self, "request_timeout")
        layout.prop(self, "download_timeout")
        layout.prop(self, "library_path")
        layout.prop(self, "project_root")
        layout.label(text="This only reads files you already downloaded. It does not log into Fab.")
        layout.label(text="Start the worker separately: python -m worker serve")
        layout.label(text="GUI Agent: set project folder above, or double-click Start GUI Agent.bat")


def get_prefs(context) -> LOCALTEXT3D_Preferences:
    return context.preferences.addons[__package__].preferences
