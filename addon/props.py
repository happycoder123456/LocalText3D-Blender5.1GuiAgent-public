import bpy


ENGINE_ITEMS = (
    ("trellis", "TRELLIS", "Higher-quality textured mesh. Uses more VRAM than Shap-E"),
    ("shap_e", "Shap-E", "Faster draft mesh. Uses less VRAM"),
)

SOURCE_ITEMS = (
    ("text", "Text", "Classic prompt. Weaker than image-to-3D on TRELLIS"),
    ("image", "Image", "Use a concept still. TRELLIS image-large is the quality path"),
    ("viewport", "Viewport", "Turn the current 3D view into a mesh"),
    ("orbit", "Orbit", "Render 4 views around the object and fuse them with TRELLIS"),
    ("scan", "Scan", "Auto-pick a matching local Megascans / Fab / Bridge scan, or variant a selected mesh"),
)

TRELLIS_VARIANT_ITEMS = (
    ("text-large", "text-large", "Default. Fits 16 GB VRAM in fp16"),
    ("text-base", "text-base", "Lighter fallback if text-large runs out of memory"),
)

TEXTURE_SIZE_ITEMS = (
    ("512", "512", "Faster bake, softer textures"),
    ("1024", "1024", "Balanced. Safe fallback if 2048 runs out of VRAM"),
    ("2048", "2048", "Sharper textures. Needs more VRAM; the worker falls back if it is tight"),
)


def _uses_image_slat_cfg(source: str, scan_use: str) -> bool:
    if source in {"image", "viewport", "orbit"}:
        return True
    if source == "scan" and scan_use == "library":
        return True
    return False


def _update_slat_cfg_for_source(self, _context) -> None:
    self.slat_cfg_strength = 3.0 if _uses_image_slat_cfg(self.source, self.scan_use) else 7.5


SCAN_USE_ITEMS = (
    ("library", "Auto", "Pick the best matching local Megascans/Fab/Bridge asset from your prompt"),
    ("selected", "Selected", "TRELLIS variant of the selected imported scan mesh"),
)


OUTPUT_MODE_ITEMS = (
    ("MESH", "Mesh", "Generate a static mesh with TRELLIS or Shap-E"),
    ("GEOMETRY_NODES", "Geometry Nodes", "Generate a procedural Geometry Nodes tree with Ollama"),
)


# bpy requires Python to keep a reference to enum item strings returned from a
# dynamic items callback — returning fresh tuples every call can crash Blender.
_OLLAMA_ITEMS_CACHE: dict[str, list[tuple[str, str, str]]] = {}
_NO_OLLAMA_ITEMS = [("none", "(no models — start Ollama)", "Pull a model with ollama pull, then refresh")]


def _ollama_model_items(self, _context):
    key = self.ollama_models_cache or ""
    items = _OLLAMA_ITEMS_CACHE.get(key)
    if items is None:
        models = [name for name in key.split(",") if name]
        items = [(name, name, "") for name in models] or _NO_OLLAMA_ITEMS
        if len(_OLLAMA_ITEMS_CACHE) > 64:
            _OLLAMA_ITEMS_CACHE.clear()
        _OLLAMA_ITEMS_CACHE[key] = items
    return items


def _on_output_mode_change(self, context) -> None:
    if self.output_mode != "GEOMETRY_NODES":
        return
    if self.ollama_models_cache:
        return  # already know the models; the Refresh button re-queries
    # Network I/O must not run inside a property update callback: defer one tick.
    from . import gn_operators

    def _refresh():
        try:
            scene = bpy.context.scene
            props = getattr(scene, "localtext3d", None) if scene else None
            if props is not None:
                gn_operators.refresh_ollama_models_props(props, report_error=False)
        except Exception:
            pass
        return None

    try:
        bpy.app.timers.register(_refresh, first_interval=0.05)
    except Exception:
        pass


class LOCALTEXT3D_ScanItem(bpy.types.PropertyGroup):
    asset_id: bpy.props.StringProperty()
    name: bpy.props.StringProperty()
    preview_file: bpy.props.StringProperty()
    preview_url: bpy.props.StringProperty()
    embedded: bpy.props.StringProperty()
    folder: bpy.props.StringProperty()
    asset_type: bpy.props.StringProperty()


class LOCALTEXT3D_HistoryItem(bpy.types.PropertyGroup):
    job_id: bpy.props.StringProperty()
    prompt: bpy.props.StringProperty()
    engine: bpy.props.StringProperty()
    seed: bpy.props.IntProperty()
    variant: bpy.props.StringProperty()
    output_path: bpy.props.StringProperty()


class LOCALTEXT3D_Props(bpy.types.PropertyGroup):
    prompt: bpy.props.StringProperty(
        name="Prompt",
        description="Describe the object to generate",
        default="",
        options={"TEXTEDIT_UPDATE"},
    )
    engine: bpy.props.EnumProperty(
        name="Engine",
        description="TRELLIS for quality. Shap-E for faster drafts",
        items=ENGINE_ITEMS,
        default="trellis",
    )
    source: bpy.props.EnumProperty(
        name="Source",
        description="Text is the old path. Image, viewport, and orbit use TRELLIS image-to-3D",
        items=SOURCE_ITEMS,
        default="text",
        update=_update_slat_cfg_for_source,
    )
    image_path: bpy.props.StringProperty(
        name="Reference",
        description="Still image of the object. PNG or JPEG. Transparent background helps",
        default="",
        subtype="FILE_PATH",
    )
    scan_use: bpy.props.EnumProperty(
        name="Scan input",
        description="Auto picks a matching local scan from the prompt. Selected uses an imported mesh",
        items=SCAN_USE_ITEMS,
        default="library",
        update=_update_slat_cfg_for_source,
    )
    auto_scan: bpy.props.BoolProperty(
        name="Use matching scan",
        description="If a local Megascans/Fab/Bridge asset matches the prompt, use it automatically. No picker",
        default=True,
    )
    last_scan_pick: bpy.props.StringProperty(default="", options={"HIDDEN"})
    last_scan_folder: bpy.props.StringProperty(default="", options={"HIDDEN"})
    scan_filter: bpy.props.StringProperty(
        name="Filter",
        description="Filter local Megascans/Fab assets by name",
        default="",
        options={"TEXTEDIT_UPDATE"},
    )
    scan_index: bpy.props.IntProperty(default=0)
    scan_assets: bpy.props.CollectionProperty(type=LOCALTEXT3D_ScanItem)
    borrow_scan_materials: bpy.props.BoolProperty(
        name="Keep scan materials",
        description="Copy PBR materials from the selected Megascans/Fab object onto the new mesh",
        default=True,
    )
    seed: bpy.props.IntProperty(
        name="Seed",
        description="Change this to get a different variation of the same prompt",
        default=1,
        min=0,
        max=2_147_483_647,
    )

    trellis_variant: bpy.props.EnumProperty(
        name="Variant",
        description="TRELLIS text model. text-large is the default for 16 GB cards",
        items=TRELLIS_VARIANT_ITEMS,
        default="text-large",
    )
    structure_steps: bpy.props.IntProperty(
        name="Structure Steps",
        description="Sampling steps for the sparse structure stage. 25 is Microsoft's text quality default",
        default=25,
        min=4,
        max=64,
    )
    slat_steps: bpy.props.IntProperty(
        name="SLAT Steps",
        description="Sampling steps for the structured latent stage. 25 is Microsoft's text quality default",
        default=25,
        min=4,
        max=64,
    )
    cfg_strength: bpy.props.FloatProperty(
        name="CFG",
        description="How strongly the structure stage follows the prompt",
        default=7.5,
        min=0.0,
        max=20.0,
    )
    slat_cfg_strength: bpy.props.FloatProperty(
        name="SLAT CFG",
        description="How strongly the latent stage follows the prompt. 7.5 for text, 3.0 for image/viewport/orbit",
        default=7.5,
        min=0.0,
        max=20.0,
    )
    simplify_ratio: bpy.props.FloatProperty(
        name="Simplify",
        description="Fraction of triangles to keep when building the mesh (0.95 = keep 95%)",
        default=0.95,
        min=0.90,
        max=0.98,
    )
    texture_size: bpy.props.EnumProperty(
        name="Texture Size",
        description="Bake resolution. 2048 is default; worker falls back to 1024 if VRAM is tight",
        items=TEXTURE_SIZE_ITEMS,
        default="2048",
    )

    shap_e_steps: bpy.props.IntProperty(
        name="Steps",
        description="Denoising steps. Higher is slower and a bit cleaner",
        default=64,
        min=8,
        max=128,
    )
    shap_e_guidance: bpy.props.FloatProperty(
        name="Guidance",
        description="How strongly Shap-E follows the prompt",
        default=15.0,
        min=1.0,
        max=40.0,
    )

    status: bpy.props.StringProperty(
        name="Status",
        default="Idle",
    )
    status_detail: bpy.props.StringProperty(
        name="Detail",
        default="Enter a prompt, then generate.",
    )
    is_running: bpy.props.BoolProperty(default=False)
    last_job_id: bpy.props.StringProperty(default="")
    worker_summary: bpy.props.StringProperty(default="Worker not checked yet")
    available_engines: bpy.props.StringProperty(default="")

    target_size: bpy.props.FloatProperty(
        name="Size",
        description="Longest side after import, in meters. Makes TRELLIS/Shap-E scale usable in a scene",
        default=1.0,
        min=0.05,
        max=50.0,
        unit="LENGTH",
    )
    sit_on_ground: bpy.props.BoolProperty(
        name="Sit on ground",
        description="Drop the mesh so its lowest point rests on the 3D cursor height (or Z=0)",
        default=True,
    )
    place_at_cursor: bpy.props.BoolProperty(
        name="At 3D cursor",
        description="Place the mesh at the 3D cursor instead of world origin",
        default=True,
    )
    line_up: bpy.props.BoolProperty(
        name="Line up",
        description="Place each new mesh beside the last so you can compare seeds instead of stacking them",
        default=True,
    )
    compare_slot: bpy.props.IntProperty(default=0, options={"HIDDEN"})
    history: bpy.props.CollectionProperty(type=LOCALTEXT3D_HistoryItem)

    output_mode: bpy.props.EnumProperty(
        name="Output",
        description="Mesh uses TRELLIS/Shap-E. Geometry Nodes uses Ollama",
        items=OUTPUT_MODE_ITEMS,
        default="MESH",
        update=_on_output_mode_change,
    )
    ollama_models_cache: bpy.props.StringProperty(default="", options={"HIDDEN"})
    ollama_model: bpy.props.EnumProperty(
        name="Model",
        description="Ollama model for Geometry Nodes generation",
        items=_ollama_model_items,
        default=0,
    )
    is_gn_running: bpy.props.BoolProperty(default=False)
    gn_status: bpy.props.StringProperty(default="Ready")
    gn_status_detail: bpy.props.StringProperty(
        default="Switch to Geometry Nodes and pick an Ollama model.",
    )
    gn_instructions_path: bpy.props.StringProperty(
        name="Instructions file",
        description="Full how-to guide saved as a .txt file",
        default="",
        subtype="FILE_PATH",
    )


def get_props(context) -> LOCALTEXT3D_Props:
    return context.scene.localtext3d
