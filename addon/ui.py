import bpy

from .props import get_props


def _draw_mesh_panel(layout, props):
    worker_box = layout.box()
    worker_box.label(text=props.worker_summary, icon="INFO")
    row = worker_box.row(align=True)
    row.operator("localtext3d.check_worker", icon="FILE_REFRESH")

    layout.separator()
    layout.prop(props, "source", expand=True)
    if props.source == "text":
        layout.prop(props, "prompt", text="")
        layout.prop(props, "auto_scan")
    elif props.source == "image":
        layout.prop(props, "image_path")
        layout.prop(props, "prompt", text="Name")
    elif props.source == "scan":
        layout.prop(props, "scan_use", expand=True)
        layout.prop(props, "prompt", text="")
        layout.prop(props, "borrow_scan_materials")
        if props.scan_use == "selected":
            layout.label(text="Select an imported Megascans/Fab mesh. TRELLIS keeps the shape.")
        else:
            layout.label(text="Picks the best matching local Megascans/Fab/Bridge scan from the prompt.")
    else:
        layout.prop(props, "prompt", text="Name")
        if props.source == "viewport":
            layout.label(text="Uses the current 3D view, including your blockout.")
        else:
            layout.label(text="Spins the view 90° × 4 and fuses those shots.")
    layout.prop(props, "engine")
    seed_row = layout.row(align=True)
    seed_row.prop(props, "seed")
    seed_row.operator("localtext3d.randomize_seed", text="", icon="MOD_NOISE")

    settings = layout.box()
    available = [name for name in props.available_engines.split(",") if name]
    if props.engine == "trellis":
        settings.label(text="TRELLIS", icon="MESH_CUBE")
        if available and "trellis" not in available:
            settings.label(text="Not installed. Run: py -3.13 scripts\\setup_trellis.py")
        if props.source == "text" or (props.source == "scan" and props.scan_use == "selected"):
            settings.prop(props, "trellis_variant")
        col = settings.column(align=True)
        col.prop(props, "structure_steps")
        col.prop(props, "slat_steps")
        col.prop(props, "cfg_strength")
        col.prop(props, "slat_cfg_strength")
        settings.prop(props, "simplify_ratio")
        settings.prop(props, "texture_size")
        if props.source == "text":
            settings.label(text="3D scans become the shape. Decals/surfaces texture the mesh.")
            settings.label(text="Tip: Image or Viewport gives sharper results than text alone.")
        elif props.source == "scan" and props.scan_use == "selected":
            settings.label(text="Using TRELLIS text model to variant the selected scan.")
        else:
            settings.label(text="Using TRELLIS-image-large (not the text model).")
        settings.label(text="Do not GPU-render in Cycles while this runs.")
        if available and "trellis" in available and props.worker_summary.find("mesh-only") >= 0:
            settings.label(text="Textures need CUDA Toolkit 12.4 + VS Build Tools (optional).")
    else:
        settings.label(text="Shap-E", icon="MESH_UVSPHERE")
        if available and "shap_e" not in available:
            settings.label(text="Not installed. Run scripts\\start_worker.bat to add Shap-E.")
        settings.prop(props, "shap_e_steps")
        settings.prop(props, "shap_e_guidance")
        if props.source != "text":
            settings.label(text="Image mode uses shap-e-img2img. Orbit still sends 4 views only to TRELLIS.")
        settings.label(text="Draft quality. Switch to TRELLIS for a better mesh.")

    studio_box = layout.box()
    studio_box.label(text="Studio import", icon="SCENE_DATA")
    studio_box.prop(props, "target_size")
    row = studio_box.row(align=True)
    row.prop(props, "sit_on_ground")
    row.prop(props, "place_at_cursor")
    studio_box.prop(props, "line_up")

    layout.separator()
    gen = layout.row()
    gen.enabled = not props.is_running
    gen.scale_y = 1.4
    gen.operator("localtext3d.generate", icon="ADD")
    if props.is_running:
        layout.operator("localtext3d.cancel", icon="CANCEL")

    status = layout.box()
    status.label(text=props.status, icon="TIME" if props.is_running else "CHECKMARK")
    if props.status_detail:
        status.label(text=props.status_detail)
    if props.source == "text" and not props.prompt.strip() and not props.is_running:
        status.label(text="Empty prompt — describe an object, like a wooden stool.")

    keepers = layout.box()
    head = keepers.row()
    head.label(text="Keepers", icon="DOCUMENTS")
    head.operator("localtext3d.refresh_history", text="", icon="FILE_REFRESH")
    if not props.history:
        keepers.label(text="Finished meshes land here. Survives a worker restart.")
    else:
        for index, item in enumerate(props.history):
            row = keepers.row(align=True)
            label = (item.prompt or item.job_id)[:36]
            row.label(text=label)
            use = row.operator("localtext3d.use_keeper", text="", icon="PASTEDOWN")
            use.index = index
            reimport = row.operator("localtext3d.reimport_keeper", text="", icon="IMPORT")
            reimport.index = index


def _draw_gn_panel(layout, props):
    model_row = layout.row(align=True)
    model_row.prop(props, "ollama_model", text="Model")
    model_row.operator("localtext3d.refresh_ollama_models", text="", icon="FILE_REFRESH")

    layout.prop(props, "prompt", text="")
    seed_row = layout.row(align=True)
    seed_row.prop(props, "seed")
    seed_row.operator("localtext3d.randomize_seed", text="", icon="MOD_NOISE")

    layout.separator()
    gen = layout.row()
    # bool(): a stale enum index yields "" and RNA rejects a str for `enabled`.
    has_model = bool(props.ollama_model) and props.ollama_model != "none"
    gen.enabled = (not props.is_gn_running) and has_model and bool(props.prompt.strip())
    gen.scale_y = 1.4
    gen.operator("localtext3d.generate_gn", icon="NODETREE")
    if props.is_gn_running:
        layout.operator("localtext3d.cancel_gn", icon="CANCEL")

    status = layout.box()
    status.label(text=props.gn_status, icon="TIME" if props.is_gn_running else "CHECKMARK")
    if props.gn_status_detail:
        status.label(text=props.gn_status_detail)
    if not props.prompt.strip() and not props.is_gn_running:
        status.label(text="Example: procedural medieval stone wall with missing bricks.")

    if props.gn_instructions_path.strip():
        guide = layout.box()
        guide.label(text="How to use", icon="INFO")
        guide.label(text="Full steps saved as a .txt file (no truncation).")
        row = guide.row(align=True)
        row.operator("localtext3d.open_gn_instructions", text="Open .txt", icon="TEXT")
        row.operator("localtext3d.reveal_gn_instructions", text="Show folder", icon="FILE_FOLDER")


class LOCALTEXT3D_PT_panel(bpy.types.Panel):
    bl_label = "Local Text to 3D"
    bl_idname = "LOCALTEXT3D_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Text to 3D"

    def draw(self, context):
        props = get_props(context)
        layout = self.layout

        layout.prop(props, "output_mode", expand=True)
        layout.separator()

        if props.output_mode == "GEOMETRY_NODES":
            _draw_gn_panel(layout, props)
        else:
            _draw_mesh_panel(layout, props)


CLASSES = (LOCALTEXT3D_PT_panel,)
