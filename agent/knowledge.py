"""Blender operator knowledge base: what an operator is, how to trigger it with
keyboard/mouse, which mode it needs, which parameters it takes and what effect
it should have on the scene. This is what lets a recording or a video lesson be
understood as *operations with parameters* instead of a pixel screenplay."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agent.op_params import normalize_idname as _normalize_idname
from agent.op_params import params_from_props  # noqa: F401  (re-exported for callers)

EDIT = "EDIT"
OBJECT = "OBJECT"
ANY = "ANY"

# Viewport / Principled Base Color names the user actually says in a goal.
COLORS: dict[str, tuple[float, float, float]] = {
    "red": (0.86, 0.12, 0.12),
    "blue": (0.15, 0.32, 0.85),
    "green": (0.18, 0.62, 0.22),
    "yellow": (0.92, 0.78, 0.12),
    "orange": (0.92, 0.42, 0.08),
    "purple": (0.52, 0.18, 0.72),
    "pink": (0.92, 0.42, 0.62),
    "brown": (0.42, 0.24, 0.12),
    "white": (0.92, 0.92, 0.92),
    "black": (0.04, 0.04, 0.04),
    "gray": (0.45, 0.45, 0.45),
    "grey": (0.45, 0.45, 0.45),
    "cyan": (0.12, 0.72, 0.78),
    "gold": (0.86, 0.68, 0.18),
    "silver": (0.72, 0.74, 0.78),
}
_COLOR_RE = re.compile(r"\b(" + "|".join(re.escape(n) for n in COLORS) + r")\b", re.I)


def parse_color(raw: Any) -> tuple[float, float, float]:
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        try:
            return tuple(max(0.0, min(1.0, float(x))) for x in raw[:3])  # type: ignore[return-value]
        except (TypeError, ValueError):
            pass
    name = str(raw or "").strip().lower()
    if name in COLORS:
        return COLORS[name]
    m = _COLOR_RE.search(name)
    if m:
        return COLORS[m.group(1).lower()]
    return COLORS["red"]


def color_name_in_text(text: str) -> str:
    m = _COLOR_RE.search(text or "")
    if not m:
        return ""
    name = m.group(1).lower()
    return "gray" if name == "grey" else name


_PRIMITIVES = {
    "cube": ("mesh.primitive_cube_add", "Add Cube", ("cube", "box", "block")),
    "uv_sphere": ("mesh.primitive_uv_sphere_add", "Add UV Sphere", ("sphere", "ball", "uv sphere")),
    "ico_sphere": ("mesh.primitive_ico_sphere_add", "Add Ico Sphere", ("ico sphere", "icosphere")),
    "cylinder": ("mesh.primitive_cylinder_add", "Add Cylinder", ("cylinder", "tube", "pipe")),
    "cone": ("mesh.primitive_cone_add", "Add Cone", ("cone", "spike")),
    "plane": ("mesh.primitive_plane_add", "Add Plane", ("plane", "floor", "ground", "flat square")),
    "torus": ("mesh.primitive_torus_add", "Add Torus", ("torus", "donut", "ring")),
    "circle": ("mesh.primitive_circle_add", "Add Circle", ("circle", "disc", "disk")),
    "monkey": ("mesh.primitive_monkey_add", "Add Monkey", ("monkey", "suzanne")),
}


@dataclass(frozen=True)
class OpSpec:
    op: str
    label: str
    mode: str = ANY
    hotkeys: tuple[tuple[str, ...], ...] = ()
    search: str = ""
    params: tuple[str, ...] = ()
    defaults: dict[str, Any] = field(default_factory=dict)
    effect: dict[str, Any] = field(default_factory=dict)
    synonyms: tuple[str, ...] = ()
    typed: str = ""  # numeric param typed after the hotkey (transform-style modal)
    axis_param: str = ""  # param that carries an axis letter (x/y/z)
    confirm: bool = False  # modal that needs Enter to finish
    pointer: bool = False  # needs a viewport click
    needs_selection: bool = False
    describe: str = ""


def _spec(*args: Any, **kw: Any) -> OpSpec:
    return OpSpec(*args, **kw)


_SPECS: list[OpSpec] = [
    _spec(
        "object.editmode_toggle", "Toggle Edit Mode", ANY, (("tab",),), "Toggle Edit Mode",
        params=("target",), defaults={"target": ""}, effect={"mode": "toggle"},
        synonyms=("edit mode", "object mode", "tab", "toggle mode"),
        describe="switch to <target> mode (EDIT or OBJECT; Tab)",
    ),
    _spec(
        "mesh.extrude_region_move", "Extrude Region", EDIT, (("e",),), "Extrude Region",
        params=("distance",), defaults={"distance": 1.0}, effect={"faces": "+", "verts": "+"},
        synonyms=("extrude", "pull out", "push out", "extend", "thicken", "pull up", "raise"),
        typed="distance", confirm=True, needs_selection=True,
        describe="extrude the selected faces along their normal by <distance>",
    ),
    _spec(
        "mesh.inset", "Inset Faces", EDIT, (("i",),), "Inset Faces",
        params=("thickness",), defaults={"thickness": 0.2}, effect={"faces": "+"},
        synonyms=("inset", "inset faces", "inset the faces"), typed="thickness", confirm=True,
        needs_selection=True, describe="inset the selected faces by <thickness>",
    ),
    _spec(
        "mesh.bevel", "Bevel", EDIT, (("ctrl", "b"),), "Bevel",
        params=("offset", "segments"), defaults={"offset": 0.1, "segments": 1},
        effect={"faces": "+"}, synonyms=("bevel", "round edges", "chamfer", "round off"),
        typed="offset", confirm=True, needs_selection=True,
        describe="bevel the selected edges by <offset>",
    ),
    _spec(
        "mesh.loopcut_slide", "Loop Cut and Slide", EDIT, (("ctrl", "r"),), "",
        params=("number_cuts",), defaults={"number_cuts": 1}, effect={"verts": "+"},
        synonyms=("loop cut", "loopcut", "edge loop", "add loop", "cut loop"), pointer=True,
        describe="add <number_cuts> edge loop(s) across the mesh",
    ),
    _spec(
        "mesh.subdivide", "Subdivide", EDIT, (), "Subdivide", params=("number_cuts",),
        defaults={"number_cuts": 1}, effect={"faces": "+"},
        synonyms=("subdivide", "subdivision cuts", "split faces"), needs_selection=True,
        describe="subdivide the selected faces",
    ),
    _spec(
        "transform.translate", "Move", ANY, (("g",),), "Move",
        params=("distance", "axis"), defaults={"distance": 1.0, "axis": "z"}, effect={"op": True},
        synonyms=("move", "translate", "grab", "slide", "move it", "nudge"), typed="distance",
        axis_param="axis", confirm=True, needs_selection=True,
        describe="move the selection <distance> along <axis>",
    ),
    _spec(
        "transform.resize", "Scale", ANY, (("s",),), "Scale",
        params=("factor", "axis"), defaults={"factor": 2.0, "axis": ""}, effect={"op": True},
        synonyms=("scale", "resize", "enlarge", "shrink", "bigger", "smaller", "grow", "stretch"),
        typed="factor", axis_param="axis", confirm=True, needs_selection=True,
        describe="scale the selection by <factor> (optionally only on <axis>)",
    ),
    _spec(
        "transform.rotate", "Rotate", ANY, (("r",),), "Rotate",
        params=("angle", "axis"), defaults={"angle": 45.0, "axis": "z"}, effect={"op": True},
        synonyms=("rotate", "turn", "spin", "tilt"), typed="angle", axis_param="axis",
        confirm=True, needs_selection=True, describe="rotate the selection <angle> degrees around <axis>",
    ),
    _spec(
        "mesh.select_mode", "Select Mode", EDIT, (), "", params=("type",), defaults={"type": "FACE"},
        effect={"select_mode": "param:type"},
        synonyms=("face select", "edge select", "vertex select", "select mode"),
        describe="switch to <type> select mode (1/2/3)",
    ),
    _spec(
        "mesh.select_all", "Select All", EDIT, (("a",),), "Select All", params=("action",),
        defaults={"action": "SELECT"}, effect={"op": True},
        synonyms=("select all", "select everything", "deselect all", "deselect", "select whole mesh"),
        describe="select or deselect everything (action SELECT/DESELECT)",
    ),
    _spec(
        "object.select_all", "Select All Objects", OBJECT, (("a",),), "Select All", params=("action",),
        defaults={"action": "SELECT"}, effect={"op": True}, synonyms=("select all objects",),
        describe="select or deselect all objects",
    ),
    _spec(
        "view3d.select", "Click Select", ANY, (), "", params=(), effect={},
        synonyms=("click", "pick", "select the face", "select a face", "select face"), pointer=True,
        describe="click in the viewport to select what is under the cursor",
    ),
    _spec(
        "mesh.delete", "Delete Geometry", EDIT, (), "Delete Faces", params=("type",), defaults={"type": "FACE"},
        effect={"faces": "-"}, synonyms=("delete faces", "delete face", "remove faces", "delete vertices"),
        needs_selection=True, describe="delete the selected <type>",
    ),
    _spec(
        "object.delete", "Delete Object", OBJECT, (("x",), ("delete",)), "Delete",
        effect={"objects": "-"}, synonyms=("delete", "delete object", "remove object", "get rid of"),
        confirm=True, needs_selection=True, describe="delete the selected object(s)",
    ),
    _spec(
        "object.duplicate_move", "Duplicate", OBJECT, (("shift", "d"),), "Duplicate Objects",
        params=("distance", "axis"), defaults={"distance": 0.0, "axis": "x"}, effect={"objects": "+"},
        synonyms=("duplicate", "copy", "clone", "another one"), typed="distance", axis_param="axis",
        confirm=True, needs_selection=True, describe="duplicate the selection and move it <distance> on <axis>",
    ),
    _spec(
        "mesh.duplicate_move", "Duplicate Mesh", EDIT, (("shift", "d"),), "Duplicate",
        params=("distance", "axis"), defaults={"distance": 0.0, "axis": "x"}, effect={"verts": "+"},
        synonyms=("duplicate faces", "copy faces"), typed="distance", axis_param="axis", confirm=True,
        needs_selection=True, describe="duplicate the selected geometry",
    ),
    _spec(
        "object.join", "Join", OBJECT, (("ctrl", "j"),), "Join", effect={"objects": "-"},
        synonyms=("join", "merge objects", "combine objects"), needs_selection=True,
        describe="join selected objects into the active one",
    ),
    _spec(
        "mesh.remove_doubles", "Merge by Distance", EDIT, (), "Merge by Distance", effect={"op": True},
        synonyms=("merge by distance", "remove doubles", "merge vertices", "weld"),
        describe="merge overlapping vertices",
    ),
    _spec(
        "mesh.edge_face_add", "Fill Face", EDIT, (("f",),), "New Edge/Face from Vertices",
        effect={"faces": "+"}, synonyms=("fill", "make face", "create face", "cap", "close hole"),
        needs_selection=True, describe="create a face from the selection",
    ),
    _spec(
        "mesh.normals_make_consistent", "Recalculate Normals", EDIT, (("shift", "n"),),
        "Recalculate Outside", effect={"op": True}, synonyms=("recalculate normals", "fix normals"),
        describe="recalculate normals to point outside",
    ),
    _spec(
        "object.shade_smooth", "Shade Smooth", OBJECT, (), "Shade Smooth", effect={"op": True},
        synonyms=("shade smooth", "smooth shading", "make it smooth", "smooth"),
        describe="smooth shading on selected objects",
    ),
    _spec(
        "object.shade_flat", "Shade Flat", OBJECT, (), "Shade Flat", effect={"op": True},
        synonyms=("shade flat", "flat shading"), describe="flat shading on selected objects",
    ),
    _spec(
        "object.subdivision_set", "Subdivision Surface", OBJECT, (("ctrl", "1"), ("ctrl", "2"), ("ctrl", "3")),
        "", params=("level",), defaults={"level": 2}, effect={"op": True},
        synonyms=("subdivision surface", "subsurf", "subdivision modifier", "smooth mesh", "high poly"),
        describe="add a Subdivision Surface modifier at <level>",
    ),
    _spec(
        "object.modifier_add", "Add Modifier", OBJECT, (), "Add Modifier", params=("type",),
        defaults={"type": "SUBSURF"}, effect={"op": True},
        synonyms=("modifier", "add modifier", "mirror modifier", "array modifier", "solidify"),
        describe="add a modifier of <type>",
    ),
    _spec(
        "object.origin_set", "Origin to Geometry", OBJECT, (), "Origin to Geometry", effect={"op": True},
        synonyms=("origin to geometry", "center origin", "set origin"), describe="move origin to geometry",
    ),
    _spec(
        "object.transform_apply", "Apply Transforms", OBJECT, (), "All Transforms", effect={"op": True},
        synonyms=("apply transforms", "apply scale", "apply rotation"), describe="apply all transforms",
    ),
    _spec(
        "ed.undo", "Undo", ANY, (("ctrl", "z"),), "Undo", effect={"op": True}, synonyms=("undo", "go back"),
        describe="undo the last step",
    ),
    _spec(
        "mesh.knife_tool", "Knife", EDIT, (("k",),), "Knife Topology Tool",
        effect={"verts": "+"}, synonyms=("knife", "knife cut", "cut with knife", "slice"),
        needs_selection=False, describe="cut new edges into the mesh with the knife tool",
    ),
    _spec(
        "mesh.bridge_edge_loops", "Bridge Edge Loops", EDIT, (), "Bridge Edge Loops",
        effect={"faces": "+"}, synonyms=("bridge", "bridge edge loops", "bridge loops", "connect loops"),
        needs_selection=True, describe="bridge two edge loops with faces",
    ),
    _spec(
        "mesh.bisect", "Bisect", EDIT, (), "Bisect", effect={"verts": "+"},
        synonyms=("bisect", "cut through", "plane cut"), needs_selection=True,
        describe="cut the mesh with a plane",
    ),
    _spec(
        "mesh.edge_split", "Edge Split", EDIT, (), "Edge Split", effect={"op": True},
        synonyms=("edge split", "split edges"), needs_selection=True, describe="split selected edges",
    ),
    _spec(
        "mesh.rip_move", "Rip", EDIT, (("v",),), "Rip", effect={"verts": "+"},
        synonyms=("rip", "rip vertices", "tear"), needs_selection=True, describe="rip vertices apart",
        confirm=True,  # V starts a modal move; Enter keeps the rip in place
    ),
    _spec(
        "mesh.edge_rotate", "Rotate Edge", EDIT, (), "Rotate Edge CW", effect={"op": True},
        synonyms=("rotate edge", "spin edge"), needs_selection=True, describe="rotate the selected edge",
    ),
    _spec(
        "mesh.fill_grid", "Grid Fill", EDIT, (), "Grid Fill", effect={"faces": "+"},
        synonyms=("grid fill", "fill grid", "quad fill"), needs_selection=True,
        describe="fill a hole with a grid of quads",
    ),
    _spec(
        "mesh.poke", "Poke Faces", EDIT, (), "Poke Faces", effect={"faces": "+"},
        synonyms=("poke", "poke faces"), needs_selection=True, describe="poke selected faces",
    ),
    _spec(
        "mesh.solidify", "Solidify", EDIT, (), "Solidify Faces", params=("thickness",),
        defaults={"thickness": 0.1}, effect={"faces": "+"},
        synonyms=("solidify", "add thickness", "shell"), typed="thickness", needs_selection=True,
        describe="solidify selected faces by <thickness>",
    ),
    _spec(
        "mesh.wireframe", "Wireframe", EDIT, (), "Wireframe", params=("thickness",),
        defaults={"thickness": 0.1}, effect={"faces": "+"},
        synonyms=("wireframe", "make wireframe"), needs_selection=True, describe="convert faces to wireframe",
    ),
    _spec(
        # P opens a menu and waits; the F3 label runs "Selection" directly.
        "mesh.separate", "Separate", EDIT, (), "Separate Selection", params=("type",),
        defaults={"type": "SELECTED"}, effect={"objects": "+"},
        synonyms=("separate", "separate selection", "split off", "p menu"), needs_selection=True,
        describe="separate selected geometry into a new object",
    ),
    _spec(
        "mesh.spin", "Spin", EDIT, (), "Spin", params=("steps",), defaults={"steps": 12},
        effect={"verts": "+"}, synonyms=("spin", "lathe", "revolve"), needs_selection=True,
        describe="spin the selection around the 3D cursor",
    ),
    _spec(
        "mesh.screw", "Screw", EDIT, (), "Screw", effect={"verts": "+"},
        synonyms=("screw", "screw tool"), needs_selection=True, describe="extrude and spin like a screw",
    ),
    _spec(
        "mesh.dissolve_edges", "Dissolve Edges", EDIT, (), "Dissolve Edges", effect={"op": True},
        synonyms=("dissolve", "dissolve edges", "dissolve faces", "limited dissolve"),
        needs_selection=True, describe="dissolve selected edges/faces",
    ),
    _spec(
        "wm.tool_set_by_id", "Set Tool", ANY, (), "", params=("name",), defaults={"name": ""},
        effect={"op": True}, synonyms=("polybuild", "poly build"), describe="activate a workspace tool",
    ),
    _spec(
        "material.set", "Set Material Color", OBJECT, (), "New Material",
        params=("color", "name"), defaults={"color": "red", "name": ""},
        effect={"op": True},
        synonyms=(
            "material", "add material", "assign material", "new material", "base color",
            "set color", "shader", "principled", "apply material",
        ),
        describe="assign a <color> material to the selected objects",
    ),
    _spec(
        "workspace.set", "Switch Workspace", ANY, (), "",
        params=("name",), defaults={"name": "Modeling"},
        effect={"op": True},
        synonyms=(
            "workspace", "modeling workspace", "uv editing", "texture paint",
            "sculpting workspace", "layout workspace", "shading workspace",
        ),
        describe="switch Blender to the <name> workspace (Modeling, UV Editing, Texture Paint…)",
    ),
    _spec(
        "object.mode_set", "Set Object Mode", ANY, (), "",
        params=("mode",), defaults={"mode": "OBJECT"},
        effect={"mode": "param:mode"},
        synonyms=(
            "sculpt mode", "texture paint mode", "vertex paint mode", "weight paint mode",
            "set mode", "object mode set",
        ),
        describe="set object interaction mode to <mode> (OBJECT, EDIT, SCULPT, TEXTURE_PAINT…)",
    ),
]

for _key, (_op, _label, _syn) in _PRIMITIVES.items():
    _SPECS.append(
        _spec(
            _op, _label, ANY, (), _label, params=("size",), defaults={},
            effect={"objects": "+", "verts": "+"}, synonyms=tuple(_syn) + (f"add {_syn[0]}",),
            describe=f"add a new {_syn[0]} primitive (scale afterwards for a specific size)",
        )
    )

SPECS: dict[str, OpSpec] = {s.op: s for s in _SPECS}

# Ops that are navigation / UI noise and never part of a skill.
IGNORED_OP_PREFIXES = ("view3d.", "view2d.", "screen.", "wm.", "localtext3d.", "outliner.", "ui.", "buttons.")
# object.mode_set is a first-class plan op (sculpt / paint / edit); Tab still covers EDIT↔OBJECT.
IGNORED_OPS: set[str] = set()
_HOTKEY_INDEX: dict[tuple[str, str], str] = {}
for _s in _SPECS:
    for _chord in _s.hotkeys:
        _HOTKEY_INDEX[("+".join(_chord), _s.mode)] = _s.op


def get(op: str) -> OpSpec | None:
    return SPECS.get(_normalize_idname(op))


normalize_idname = _normalize_idname


def is_noise_op(op: str) -> bool:
    idn = _normalize_idname(op)
    if idn in IGNORED_OPS:
        return True
    return any(idn.startswith(p) for p in IGNORED_OP_PREFIXES)


def op_for_hotkey(keys: list[str], mode: str) -> str | None:
    chord = "+".join(str(k).lower() for k in keys)
    mode = mode if mode in {EDIT, OBJECT} else OBJECT
    if chord in {"1", "2", "3"} and mode == EDIT:
        return "mesh.select_mode"
    if chord in {"ctrl+1", "ctrl+2", "ctrl+3", "ctrl+4", "ctrl+5"} and mode == OBJECT:
        return "object.subdivision_set"
    if chord == "alt+a":
        return "mesh.select_all" if mode == EDIT else "object.select_all"
    for cand_mode in (mode, ANY):
        found = _HOTKEY_INDEX.get((chord, cand_mode))
        if found:
            return found
    return None


def op_for_search_text(text: str) -> str | None:
    q = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not q:
        return None
    for s in _SPECS:
        if s.search and s.search.lower() == q:
            return s.op
    for s in _SPECS:
        if s.search and (q in s.search.lower() or s.search.lower() in q):
            return s.op
    return op_for_phrase(q)


_ADD_VERBS_RE = re.compile(r"\b(add|place|create|spawn|insert|make|new|another|put|drop|start with)\b")


def op_for_phrase(phrase: str, *, allow_primitives: bool | None = None) -> str | None:
    """Best knowledge-base op for a free-text phrase (synonym lookup).

    Ties/overlaps resolve to the synonym that appears EARLIEST in the phrase
    (the verb), then the longest — so "bevel the edges with offset 0.2" is a
    bevel, and "extrude the cylinder by 2" is an extrude, not "add cylinder".
    Primitive ops only win when the phrase has an add-style verb (or is a bare
    noun like "cube"), unless allow_primitives is forced.
    """
    raw = str(phrase or "").lower()
    p = " " + re.sub(r"[^a-z0-9 ]+", " ", raw) + " "
    p = re.sub(r"\s+", " ", p)
    if allow_primitives is None:
        content = [w for w in p.split() if w]
        allow_primitives = bool(_ADD_VERBS_RE.search(p)) or len(content) <= 2
    best: tuple[int, int, str] | None = None  # (position, -len, op)
    for s in _SPECS:
        if s.op.startswith("mesh.primitive_") and not allow_primitives:
            continue
        for syn in (s.label.lower(), *s.synonyms):
            token = " " + syn + " "
            pos = p.find(token)
            if pos < 0:
                continue
            key = (pos, -len(syn), s.op)
            if best is None or key < best:
                best = key
    return best[2] if best else None


def _fmt_num(v: Any) -> str:
    """Number as Blender's numeric input accepts it; never collapses a non-zero to '0'."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(f - round(f)) < 1e-6:
        return str(int(round(f)))
    text = f"{f:.6g}"
    if "e" in text or "E" in text:
        text = f"{f:.8f}".rstrip("0").rstrip(".")
    return text


def describe_op(op: str, params: dict[str, Any] | None = None) -> str:
    spec = get(op)
    params = params or {}
    if spec is None:
        return op
    if spec.op == "object.editmode_toggle":
        target = str(params.get("target") or "").upper()
        if target in {"EDIT", "OBJECT"}:
            return f"switch to {target.title()} mode (Tab)"
        return "toggle Object/Edit mode (Tab)"
    if spec.op in {"transform.resize", "transform.translate", "transform.rotate", "object.duplicate_move", "mesh.duplicate_move"}:
        verb = {"transform.resize": "scale", "transform.translate": "move", "transform.rotate": "rotate"}.get(spec.op, "duplicate")
        val = params.get(spec.typed, spec.defaults.get(spec.typed))
        axis = str(params.get("axis") or "").upper()
        unit = "°" if spec.op == "transform.rotate" else ""
        amount = f" by {_fmt_num(val)}{unit}" if val not in (None, "") else ""
        on = f" on {axis}" if axis else ""
        return f"{verb} the selection{amount}{on}".strip()
    if spec.op == "material.set":
        color = str(params.get("color") or spec.defaults.get("color") or "red")
        if isinstance(params.get("color"), (list, tuple)):
            color = color_name_in_text(" ".join(str(x) for x in params.get("color") or [])) or "custom"
        return f"assign a {color} material to the selected objects"
    if spec.op == "workspace.set":
        return f"switch to {params.get('name') or 'Modeling'} workspace"
    if spec.op == "object.mode_set":
        return f"enter {str(params.get('mode') or 'OBJECT').upper()} mode"
    text = spec.describe or spec.label.lower()
    for name in spec.params:
        val = params.get(name, spec.defaults.get(name, ""))
        text = text.replace(f"<{name}>", _fmt_num(val) if val not in ("", None) else "")
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s+(?:by|on|along|around|at|of)\s*$", "", text.strip())
    return re.sub(r"\s+", " ", text).strip()


def _step(action: str, **kw: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"action": action}
    out.update(kw)
    return out


def _typed_number_steps(value: Any, *, axis: str = "") -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    if axis and str(axis).lower() in {"x", "y", "z"}:
        steps.append(_step("key", keys=[str(axis).lower()], reason=f"constrain to {str(axis).upper()}"))
    try:
        f = float(value)
    except (TypeError, ValueError):
        return steps
    steps.append(_step("type", text=_fmt_num(f), reason=f"enter value {_fmt_num(f)}"))
    return steps


def expand_op(
    op: str,
    params: dict[str, Any] | None = None,
    *,
    variant: int = 0,
    pointer: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Turn one semantic op (+params) into keyboard/mouse primitives.

    variant 0 prefers the hotkey; variant 1 prefers F3 operator search.
    Returned steps have no mode handling — see solve_modes() in abstract.py.
    """
    spec = get(op)
    params = dict(params or {})
    if spec is None:
        return []
    steps: list[dict[str, Any]] = []
    why = describe_op(op, params)

    if spec.op == "view3d.select":
        if pointer:
            steps.append(_step("click", intent="viewport_point", reason="select in viewport", **pointer))
        return steps

    if spec.op == "material.set":
        rgb = parse_color(params.get("color") or spec.defaults.get("color"))
        label = color_name_in_text(str(params.get("color") or "")) or str(params.get("name") or "color")
        name = str(params.get("name") or "") or (label.title() if label else "AgentColor")
        return [
            _step(
                "blender_apply",
                apply={"op": "material.set", "color": list(rgb), "name": name[:48]},
                reason=f"assign {name} material",
            )
        ]

    if spec.op == "workspace.set":
        ws = str(params.get("name") or "Modeling").strip() or "Modeling"
        return [
            _step(
                "blender_apply",
                apply={"op": "workspace.set", "name": ws[:64]},
                reason=f"switch to {ws} workspace",
            )
        ]

    if spec.op == "object.mode_set":
        mode = str(params.get("mode") or "OBJECT").strip().upper() or "OBJECT"
        return [
            _step(
                "blender_apply",
                apply={"op": "object.mode_set", "mode": mode},
                reason=f"enter {mode} mode",
            )
        ]

    if spec.op == "mesh.select_mode":
        kind = str(params.get("type") or "FACE").upper()
        key = {"VERT": "1", "EDGE": "2", "FACE": "3"}.get(kind, "3")
        return [_step("key", keys=[key], reason=f"{kind.lower()} select mode")]

    if spec.op == "object.subdivision_set":
        level = int(max(1, min(5, int(float(params.get("level") or 2)))))
        return [_step("hotkey", keys=["ctrl", str(level)], reason=f"subdivision level {level}")]

    if spec.op in {"mesh.select_all", "object.select_all"}:
        action = str(params.get("action") or "SELECT").upper()
        if action == "DESELECT":
            return [_step("hotkey", keys=["alt", "a"], reason="deselect all")]
        return [_step("key", keys=["a"], reason="select all")]

    if spec.op == "mesh.loopcut_slide":
        cuts = int(max(1, min(20, int(float(params.get("number_cuts") or 1)))))
        steps.append(_step("hotkey", keys=["ctrl", "r"], reason="loop cut"))
        steps.append(_step("wait", seconds=0.35, reason="loop cut preview"))
        if cuts > 1:
            steps.append(_step("type", text=str(cuts), reason=f"{cuts} cuts"))
            steps.append(_step("wait", seconds=0.2, reason="update cut count"))
        where = dict(pointer) if pointer else {"x_norm": 0.45, "y_norm": 0.52}
        steps.append(_step("click", intent="viewport_center", reason="place loop cut", **where))
        steps.append(_step("wait", seconds=0.2, reason="start slide"))
        steps.append(_step("key", keys=["esc"], reason="keep cut centred"))
        return steps

    if spec.op == "mesh.delete":
        kind = str(params.get("type") or "FACE").upper()
        label = {"VERT": "Delete Vertices", "EDGE": "Delete Edges"}.get(kind, "Delete Faces")
        return _search_steps(label, why)

    if spec.op == "object.modifier_add":
        kind = str(params.get("type") or "SUBSURF").upper()
        label = {
            "SUBSURF": "Subdivision Surface", "MIRROR": "Mirror", "ARRAY": "Array",
            "SOLIDIFY": "Solidify", "BEVEL": "Bevel", "BOOLEAN": "Boolean",
        }.get(kind, kind.title())
        return _search_steps(label, why)

    use_search = bool(spec.search) and (variant == 1 or not spec.hotkeys)
    if use_search:
        steps.extend(_search_steps(spec.search, why))
    elif spec.hotkeys:
        chord = list(spec.hotkeys[0])
        steps.append(_step("hotkey" if len(chord) > 1 else "key", keys=chord, reason=why))
    else:
        return []

    # Only MODAL operators (confirm=True) accept typed numbers. For a non-modal op
    # run via F3 (e.g. Solidify Faces) the digits would land in the viewport,
    # where "1" switches select mode.
    if spec.typed and spec.confirm:
        val = params.get(spec.typed, spec.defaults.get(spec.typed))
        axis = ""
        if spec.axis_param:
            axis = str(params.get(spec.axis_param) or "")
            if not axis and spec.op == "transform.translate":
                # No explicit axis: bare numeric input fills X, so honour the
                # documented default axis instead of silently moving along X.
                axis = str(spec.defaults.get("axis") or "")
        if val is not None and val != "":
            steps.append(_step("wait", seconds=0.2, reason="modal ready"))
            steps.extend(_typed_number_steps(val, axis=axis))
        if spec.op == "mesh.bevel":
            segments = params.get("segments", spec.defaults.get("segments"))
            try:
                seg_n = int(float(segments)) if segments not in (None, "") else 1
            except (TypeError, ValueError):
                seg_n = 1
            if seg_n > 1:
                # Bevel modal: S switches numeric input to segment count.
                steps.append(_step("key", keys=["s"], reason="segments"))
                steps.append(_step("type", text=str(max(1, min(30, seg_n))), reason=f"{seg_n} segments"))
    if spec.confirm:
        steps.append(_step("key", keys=["enter"], reason="confirm"))
    return steps


def _search_steps(label: str, why: str) -> list[dict[str, Any]]:
    return [
        _step("key", keys=["f3"], reason="operator search"),
        _step("wait", seconds=0.45, reason="wait for search"),
        _step("type", text=label, reason=f"search {label}"),
        _step("wait", seconds=0.4, reason="results"),
        _step("key", keys=["enter"], reason=why or f"run {label}"),
    ]


def has_variant(op: str) -> bool:
    spec = get(op)
    return bool(spec and spec.search and spec.hotkeys)


def primitive_for_word(word: str) -> str | None:
    w = re.sub(r"\s+", " ", str(word or "").lower().strip())
    candidates = {w, w.rstrip("s"), w[:-2] if w.endswith("es") else w, w.replace("icosphere", "ico sphere")}
    if w in {"tori"}:
        candidates.add("torus")
    for _key, (op, _label, syns) in _PRIMITIVES.items():
        if candidates & set(syns):
            return op
    return None


_CATALOG_SKIP = {"view3d.select", "wm.tool_set_by_id"}


def tool_catalog(limit: int = 0) -> list[dict[str, Any]]:
    """Compact op list for the LLM planner prompt.

    Every runnable op is listed (primitives sit at the end of _SPECS and were
    being cut off by a positional limit, so the LLM had no cube/cylinder to start from).
    """
    out: list[dict[str, Any]] = []
    for s in _SPECS:
        if s.op in _CATALOG_SKIP:
            continue
        out.append(
            {
                "op": s.op,
                "mode": s.mode,
                "params": list(s.params),
                "does": s.describe or s.label,
            }
        )
        if limit and len(out) >= limit:
            break
    return out
