"""Goal → plan of semantic ops, composed from learned skills + Blender knowledge.

Two planners cooperate:
  1. rules: phrase parsing (numbers, axes, primitives, repeats) against learned
     skill names/aliases and knowledge-base synonyms. Fast, offline, predictable.
  2. llm:   a local text model (Ollama) that sees the same skills + op catalog
     and returns JSON. Used for phrasing the rules cannot resolve.
Both produce ops the executor can run in variation (any parameters, any mode)."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

from agent import knowledge as kb
from agent.abstract import new_op, summarize_ops
from agent.skills import _match_labels, skill_match_labels, split_goal_phrases

_NUMBER_RE = re.compile(r"(-?\d+(?:\.\d+)?)")
_AXIS_RE = re.compile(r"\b(?:on|along|around|in)?\s*(?:the\s+)?([xyz])(?:\s*[- ]?axis)?\b", re.I)
# "3 times" / "3x" / "twice" — but NOT "scale by 2 x" (x = axis) or "2 x axis".
_TIMES_RE = re.compile(r"\b(\d+)(?:\s*times\b|x\b(?!\s*[- ]?axis))|\b(twice|thrice)\b", re.I)
_WORD_NUM = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "twice": 2, "thrice": 3}
_PRIM_WORDS = (
    "cubes?|box(?:es)?|spheres?|balls?|uv spheres?|ico ?spheres?|cylinders?|tubes?|pipes?|cones?|"
    "planes?|floors?|tor(?:us|i)|donuts?|circles?|discs?|monkeys?|suzannes?"
)
_ADD_RE = re.compile(
    r"\b(?:(add|place|create|spawn|insert|make|new|another|put|start with|drop)\s+)?"
    r"(?:(a|an|\d+|two|three|four|five|six)\s+)?(?:(?:big|small|tall|new|simple|basic|default)\s+)?"
    r"(" + _PRIM_WORDS + r")"
    r"(?:\s+(?:of\s+|with\s+)?(?:size|sized|radius|scale(?:d)?)(?:\s+(?:of|to|by))?\s*(-?\d+(?:\.\d+)?))?",
    re.I,
)
_SKILL_MIN = 0.45

# Open-ended "make a car/house" goals must compose techniques — not replay a
# one-shot cube+extrude plan that was auto-saved under the same goal name.
_COMPLEX_OBJECT_RE = re.compile(
    r"\b(?:make|build|create|model|design|sculpt)\b[\w\s'\-]*?\b("
    r"car|truck|van|bus|bike|motorcycle|boat|ship|air\s?plane|aeroplane|airplane|aircraft|jet|"
    r"house|home|building|cabin|tower|castle|bridge|"
    r"chair|table|desk|sofa|bed|lamp|"
    r"robot|character|person|human|animal|tree|sword|gun|weapon"
    r")\b",
    re.I,
)
# Phrases where an object word is really a mesh-primitive or an edit operation.
_NOT_OBJECT_RE = re.compile(
    r"\b(?:mesh\s+plane|plane\s+(?:primitive|mesh)|ground\s+plane|"
    r"bridge\s+(?:the\s+)?(?:edge|loop|faces?|verts?|vertices)|edge\s+loops?\s+bridge|"
    r"table\s+(?:of|view))\b",
    re.I,
)
_SHALLOW_SKIP = {
    "object.editmode_toggle",
    "mesh.select_all",
    "object.select_all",
    "view3d.select",
}
_BOX_PRIMS = {
    "mesh.primitive_cube_add",
    "mesh.primitive_plane_add",
}

_NAMED_OBJECTS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"golden\s*gate", re.I),
        "golden_gate",
        "Golden Gate Bridge: TWO tall rectangular towers, a LONG thin deck along X, "
        "main cables between the tower tops, vertical hangers. International orange/red. Never a house or a single cube.",
    ),
    (
        re.compile(r"brooklyn\s+bridge|suspension\s+bridge", re.I),
        "bridge",
        "Suspension bridge: two towers, long thin deck along X, cables and hangers. Not a building.",
    ),
    (
        re.compile(r"eiffel", re.I),
        "eiffel",
        "Eiffel Tower: four spreading legs, tapering stacked platforms, a tall spire. Not a house.",
    ),
    (
        re.compile(r"statue of liberty|liberty statue", re.I),
        "liberty",
        "Statue of Liberty: box pedestal, robe cylinder, raised torch. Green-copper if no color is named.",
    ),
    (
        re.compile(r"\bpyramids?\b", re.I),
        "pyramid",
        "Pyramid: large square base, stacked shrinking boxes toward a point. Not a house with a roof slab.",
    ),
    (re.compile(r"\blighthouse\b", re.I), "lighthouse", "Lighthouse: tall tapered cylinder and a lantern room on top."),
    (re.compile(r"\bwindmill\b", re.I), "windmill", "Windmill: tower cylinder plus four sail boxes."),
    (re.compile(r"\btrees?\b", re.I), "tree", "Tree: trunk cylinder plus foliage (ico sphere or cone)."),
]

_MAKE_RE = re.compile(r"\b(?:make|build|create|model|design|sculpt)\b", re.I)
_TECHNIQUE_WORDS = {
    "extrude", "inset", "bevel", "scale", "resize", "move", "grab", "rotate",
    "loop", "cut", "loopcut", "subdivide", "delete", "duplicate", "select",
    "deselect", "join", "separate", "fill", "undo", "shade", "smooth",
    "thicken", "shrink", "grow",
}
_PRIM_NOUNS = {
    "cube", "box", "block", "sphere", "ball", "cylinder", "tube", "pipe", "cone",
    "plane", "floor", "torus", "donut", "circle", "disc", "disk", "monkey", "suzanne",
}
# Structural families inferred from language — used as a silhouette prior, not a
# closed list of objects the agent "knows". Unknown nouns become family "thing".
_FAMILY_CUES: list[tuple[str, re.Pattern[str]]] = [
    ("span", re.compile(r"\b(bridges?|span|aqueduct|overpass|viaduct|suspension)\b", re.I)),
    ("pyramid", re.compile(r"\bpyramids?\b", re.I)),
    ("plant", re.compile(r"\b(trees?|flowers?|plants?|cactus|bush(?:es)?|palm)\b", re.I)),
    ("tower", re.compile(r"\b(towers?|skyscrapers?|lighthouses?|obelisks?|minarets?|steeples?|spires?)\b", re.I)),
    ("aircraft", re.compile(r"\b(air\s?planes?|aeroplanes?|airplanes?|aircraft|jets?|helicopters?|drones?|spaceships?|rockets?|satellites?|ufos?)\b", re.I)),
    ("vessel", re.compile(r"\b(boats?|ships?|yachts?|canoes?|submarines?)\b", re.I)),
    ("vehicle", re.compile(r"\b(cars?|trucks?|vans?|buses|bus|bikes?|bicycles?|motorcycles?|wagons?|tanks?|jeeps?)\b", re.I)),
    ("furniture", re.compile(r"\b(chairs?|tables?|desks?|sofas?|beds?|stools?|benches?|shelves|cabinets?)\b", re.I)),
    ("building", re.compile(r"\b(houses?|homes?|cabins?|cottages?|buildings?|castles?|barns?|huts?)\b", re.I)),
    ("figure", re.compile(r"\b(people|person|human|character|robot|android|statue|man|woman|animals?|dogs?|cats?|horses?)\b", re.I)),
    ("instrument", re.compile(r"\b(pianos?|guitars?|violins?|drums?|trumpets?)\b", re.I)),
    ("round", re.compile(r"\b(spheres?|balls?|globes?|domes?)\b", re.I)),
]
_NAMED_TO_FAMILY = {
    "golden_gate": "span",
    "bridge": "span",
    "eiffel": "tower",
    "lighthouse": "tower",
    "windmill": "tower",
    "pyramid": "pyramid",
    "tree": "plant",
    "liberty": "figure",
}

PLANNER_SYSTEM = """You plan Blender 5.x modeling work as JSON only.
You get: the user's goal, current Blender state, learned SKILLS/CONCEPTS, an OP CATALOG, and a SILHOUETTE hint.
Output: {"plan":[{"skill":"<skill name>","params":{...}} | {"op":"<op id>","params":{...}}, ...],"why":"one sentence"}
Rules:
- Division of labor: SILHOUETTE / world knowledge = WHAT the object looks like. Learned SKILLS and CONCEPTS (from video/recording) = HOW to model in Blender (bevel, extrude, inset, loop cut, solidify, materials…). Use both. The user can ask for ANY object — build that object with the techniques they taught you.
- You already know what real-world objects look like. Do NOT collapse unknown names into a house or a single cube.
- Reproduce the SILHOUETTE with several Object-mode primitives (cube / cylinder / cone / ico sphere), then scale + move them into place. Distinctive parts must be separate objects (wheels, legs, towers, wings, neck, keys…).
- Different objects need DIFFERENT plans. A car is not a house. A bridge is not a house. A piano is not a cube.
- Prefer TECHNIQUE skills from the SKILLS list by name ({"skill":"bevel"}) so learned concepts are actually used. Do NOT replay a skill whose name is the whole object goal.
- Switch workspace only when needed: UV Editing / Texture Paint / Sculpting / Shading if the goal is about those. Stay in Layout or Modeling for normal builds; do not hop workspaces for every object.
- Decompose: primitive → axis scales (transform.resize with axis x/y/z) → Edit mode (object.editmode_toggle target EDIT) → select → detail ops. Pin editmode_toggle with target. Never invent op ids or pixel clicks.
- If the goal names a COLOR or material, end with material.set {color:"red"} after the geometry.
- Plan LENGTH: simple 1–6, normal object 24–56, detailed 64–120.
"""


def named_object(goal: str) -> str:
    """Optional extra-specific id (Golden Gate, Eiffel…). Empty for ordinary nouns."""
    g = goal or ""
    if _NOT_OBJECT_RE.search(g):
        return ""
    for pat, key, _hint in _NAMED_OBJECTS:
        if pat.search(g):
            return key
    if re.search(r"\bbridges?\b", g, re.I) and not _NOT_OBJECT_RE.search(g):
        return "bridge"
    return ""


def goal_subject(goal: str) -> str:
    """Noun phrase the user wants, minus make/color filler."""
    words = []
    for w in re.findall(r"[a-z]+", (goal or "").lower()):
        if w in _STOPWORDS or w in kb.COLORS or w in {"please", "look", "like", "want"}:
            continue
        words.append(w)
    return " ".join(words[:6]) or "object"


def structure_family(goal: str) -> str:
    """Open-ended silhouette family from language (not a whitelist of objects)."""
    g = goal or ""
    if _NOT_OBJECT_RE.search(g):
        return ""
    named = named_object(g)
    if named in _NAMED_TO_FAMILY:
        return _NAMED_TO_FAMILY[named]
    for fam, pat in _FAMILY_CUES:
        if pat.search(g):
            return fam
    if re.search(r"\b(tall|high|vertical)\b", g, re.I):
        return "tower"
    if re.search(r"\b(long|wide|horizontal)\b", g, re.I):
        return "span"
    if re.search(r"\b(round|spherical)\b", g, re.I):
        return "round"
    return "thing"


def silhouette_hint(goal: str) -> str:
    """World-knowledge brief for the LLM — always filled for object goals."""
    g = goal or ""
    for pat, _key, hint in _NAMED_OBJECTS:
        if pat.search(g):
            return hint
    subject = goal_subject(g)
    fam = structure_family(g)
    hints = {
        "span": f"{subject}: two supports + a LONG thin deck along X + optional cables. Not a house.",
        "tower": f"{subject}: tall vertical stack, tapering toward a spire. Not a house with a door.",
        "pyramid": f"{subject}: large square base, stacked shrinking levels. Not a house.",
        "plant": f"{subject}: trunk cylinder + foliage sphere/cone.",
        "vehicle": f"{subject}: long low body, cabin, wheels as cylinders.",
        "aircraft": f"{subject}: fuselage cylinder along X, wide thin wings, tail.",
        "vessel": f"{subject}: long low hull, a cabin block on top. No car wheels.",
        "furniture": f"{subject}: platform + legs (cylinders). Not a solid cube.",
        "building": f"{subject}: walls + roof slab + openings. Only if it is actually a building.",
        "figure": f"{subject}: torso, head, limbs as separate primitives.",
        "instrument": f"{subject}: recognizable body + distinctive parts (neck, keys, legs).",
        "round": f"{subject}: sphere/dome as the main volume, plus a stand if needed.",
        "thing": (
            f"You know what a '{subject}' looks like in the real world. "
            "Build that silhouette with 5–12 primitives (main body + the parts people recognize). "
            "NEVER a generic house, NEVER a single cube."
        ),
    }
    return hints.get(fam, hints["thing"])


def is_complex_object_goal(goal: str) -> bool:
    """True for 'make/build/model <a thing>' — any real-world object, not only a whitelist."""
    g = goal or ""
    if _NOT_OBJECT_RE.search(g):
        return False
    if not _MAKE_RE.search(g):
        return False
    nouns = []
    for w in re.findall(r"[a-z]+", g.lower()):
        if w in _STOPWORDS or w in kb.COLORS or w in {"please", "look", "like", "want", "size", "sized"}:
            continue
        nouns.append(w)
    if not nouns:
        return False
    allowed = _PRIM_NOUNS | _TECHNIQUE_WORDS
    if set(nouns) <= allowed:
        return False
    return True


def object_kind(goal: str) -> str:
    named = named_object(goal)
    if named:
        return named
    if not is_complex_object_goal(goal):
        return ""
    m = _COMPLEX_OBJECT_RE.search(goal or "")
    kind = (m.group(1).lower() if m else "")
    if kind in {"air plane", "airplane", "aeroplane", "jet"}:
        return "plane"
    if kind:
        return kind
    return structure_family(goal)


def goal_detail_level(goal: str) -> int:
    """0=simple phrase, 1=normal object, 2=advanced, 3=very detailed."""
    g = (goal or "").lower()
    if not is_complex_object_goal(g):
        return 0
    fam = structure_family(g)
    if fam in {"span", "tower", "figure", "aircraft"} or named_object(g):
        level = 2
    else:
        level = 1
    if re.search(r"\b(advanced|detailed|complex|realistic|professional|high[\s-]?poly)\b", g):
        level = 2
    if re.search(r"\b(very|ultra|highly|fully)\b.{0,24}\b(advanced|detailed|complex|realistic)\b", g) or re.search(
        r"\b(ultra|photoreal|production[\s-]?ready)\b", g
    ):
        level = 3
    # Explicit feature lists push detail up.
    features = _goal_features(g)
    if len(features) >= 3:
        level = max(level, 2)
    if len(features) >= 5:
        level = max(level, 3)
    return level


def _goal_features(goal: str) -> list[str]:
    g = (goal or "").lower()
    found: list[str] = []
    for name, pat in (
        ("windows", r"\bwindows?\b"),
        ("doors", r"\bdoors?\b"),
        ("roof", r"\broofs?\b"),
        ("chimney", r"\bchimneys?\b"),
        ("garage", r"\bgarages?\b"),
        ("wheels", r"\bwheels?\b"),
        ("cabin", r"\bcabins?\b"),
        ("hood", r"\bhoods?\b"),
        ("bumper", r"\bbumpers?\b"),
        ("legs", r"\blegs?\b"),
        ("arms", r"\barms?\b"),
        ("stairs", r"\bstairs?\b"),
        ("balcony", r"\bbalcon(?:y|ies)\b"),
        ("fence", r"\bfences?\b"),
        ("material", r"\b(?:materials?|shaders?|textures?|paint|color(?:ed|ed)?|colour(?:ed)?)\b"),
        ("towers", r"\btowers?\b"),
        ("cables", r"\bcables?\b"),
        ("deck", r"\bdecks?\b"),
    ):
        if re.search(pat, g):
            found.append(name)
    return found


def plan_budget(goal: str) -> int:
    """Max plan items for LLM validation / guidance — scales with complexity."""
    level = goal_detail_level(goal)
    if level == 0:
        # Phrase goals: still allow multi-step "then" chains.
        phrases = max(1, len(split_goal_phrases(goal or "")))
        return max(16, min(48, phrases * 8 + 8))
    base = {1: 72, 2: 120, 3: 180}.get(level, 72)
    return min(240, base + len(_goal_features(goal)) * 6)


def recommended_max_steps(goal: str, n_gui_steps: int) -> int:
    """GUI action budget so a full object plan can finish (not the N-panel Max steps alone)."""
    level = goal_detail_level(goal or "")
    floor = {0: 64, 1: 220, 2: 480, 3: 720}.get(level, 220)
    n = max(0, int(n_gui_steps))
    need = n + max(32, n // 3)
    return int(min(2500, max(floor, need)))


_WORKSPACE_CUES: list[tuple[str, re.Pattern[str]]] = [
    ("UV Editing", re.compile(r"\b(uv|unwrap|seams?|uv[\s-]?map)\b", re.I)),
    ("Texture Paint", re.compile(r"\b(texture\s*paint|paint\s+(?:the\s+)?texture|vertex\s*paint)\b", re.I)),
    ("Sculpting", re.compile(r"\bsculpt(?:ing|ed)?\b", re.I)),
    ("Shading", re.compile(r"\b(shading|shader\s*editor|material\s*nodes?)\b", re.I)),
    ("Layout", re.compile(r"\b(layout\s+workspace|scene\s+layout|animate\s+the\s+scene)\b", re.I)),
]
# Specialty screens that block normal mesh builds unless the goal asks for them.
_MESH_BLOCKING_WORKSPACES = {
    "uv editing",
    "texture paint",
    "sculpting",
    "shading",
    "compositing",
    "rendering",
    "scripting",
    "geometry nodes",
    "animation",
}


def desired_workspace(goal: str) -> str:
    """Specialty workspace only when the goal names it. Empty = no forced switch."""
    g = goal or ""
    for name, pat in _WORKSPACE_CUES:
        if pat.search(g):
            return name
    return ""


def desired_object_mode(goal: str) -> str:
    """Non-EDIT/OBJECT modes only when the goal asks for paint/sculpt."""
    g = (goal or "").lower()
    if re.search(r"\btexture\s*paint|paint\s+(?:the\s+)?texture\b", g):
        return "TEXTURE_PAINT"
    if re.search(r"\bvertex\s*paint\b", g):
        return "VERTEX_PAINT"
    if re.search(r"\bweight\s*paint\b", g):
        return "WEIGHT_PAINT"
    if re.search(r"\bsculpt(?:ing|ed)?\b", g):
        return "SCULPT"
    return ""


def _needs_mesh_workspace(goal: str) -> bool:
    """True when the goal is mesh modeling (not a specialty UV/paint/sculpt ask)."""
    if desired_workspace(goal):
        return False
    g = goal or ""
    return bool(
        is_complex_object_goal(g)
        or _MAKE_RE.search(g)
        or re.search(r"\b(extrude|bevel|inset|loop\s*cut|mesh|edit\s*mode)\b", g, re.I)
    )


def ensure_workspace_ops(plan: Plan, goal: str, context: dict[str, Any] | None = None) -> Plan:
    """Switch workspace/mode only when required — never hop Layout→Modeling by default."""
    if not plan.ops:
        return plan
    ctx = context or {}
    current_ws = str(ctx.get("workspace") or "").strip()
    current_l = current_ws.lower()
    want_ws = desired_workspace(goal)
    want_mode = desired_object_mode(goal)
    prefix: list[dict[str, Any]] = []
    have_ops = [str(o.get("op") or "") for o in plan.ops if isinstance(o, dict)]

    if "workspace.set" not in have_ops:
        if want_ws and current_l != want_ws.lower():
            # Goal explicitly needs UV / paint / sculpt / shading / layout.
            prefix.append(new_op("workspace.set", {"name": want_ws}))
        elif _needs_mesh_workspace(goal) and current_l in _MESH_BLOCKING_WORKSPACES:
            # Stuck in a specialty screen while trying to build mesh — leave for Modeling.
            prefix.append(new_op("workspace.set", {"name": "Modeling"}))
            want_ws = "Modeling"

    if want_mode and "object.mode_set" not in have_ops:
        raw = str(ctx.get("raw_mode") or ctx.get("mode") or "").upper()
        if want_mode not in raw and raw != want_mode:
            prefix.append(new_op("object.mode_set", {"mode": want_mode}))
    if not prefix:
        return plan
    plan.ops = prefix + list(plan.ops)
    tag = want_ws or want_mode
    if tag and tag.lower() not in (plan.why or "").lower():
        plan.why = (plan.why + f"; {tag}").strip("; ")
    return plan


def is_shallow_box_ops(ops: list[dict[str, Any]]) -> bool:
    """True for generic cube→scale→inset/extrude plans that look the same for any object."""
    rows = [o for o in (ops or []) if isinstance(o, dict) and o.get("op")]
    if not rows:
        return True
    ids = [str(o.get("op") or "") for o in rows]
    if ids.count("mesh.select_all") >= 3:
        return True
    interesting = [op for op in ids if op not in _SHALLOW_SKIP]
    if not interesting:
        return True
    # Multi-part assemblies (body + cabin + wheels…) are structure, not a box clone.
    prims = sum(1 for op in interesting if op.startswith("mesh.primitive_"))
    if prims >= 3 or len(interesting) >= 16:
        return False
    has_box = any(op in _BOX_PRIMS for op in interesting)
    detail = [
        op
        for op in interesting
        if op not in _BOX_PRIMS
        and not op.startswith("mesh.primitive_")
        and op not in {"transform.resize", "transform.translate", "transform.rotate"}
    ]
    # Cube + a couple of scales + at most two detail ops = shallow "box recipe".
    if has_box and len(detail) <= 2:
        return True
    return False


def is_click_spam_ops(ops: list[dict[str, Any]]) -> bool:
    """True when a 'skill' is mostly viewport clicks — useless for novel goals.

    Short recorded demos (click a face, press E) are legitimate; only long
    click-heavy recordings count as spam.
    """
    ids = [str(o.get("op") or "") for o in (ops or []) if isinstance(o, dict) and o.get("op")]
    if not ids:
        return True
    selects = sum(1 for op in ids if op == "view3d.select")
    if selects >= 6:
        return True
    return len(ids) >= 5 and (selects / len(ids) >= 0.35)


def is_search_only_ops(ops: list[dict[str, Any]]) -> bool:
    """True when ops are only F3 searches — typing a phrase is not modeling."""
    ids = [str(o.get("op") or "") for o in (ops or []) if isinstance(o, dict) and o.get("op")]
    if not ids:
        return True
    return all(op in {"search", "wm.search_operator"} for op in ids)


def is_weak_object_plan(ops: list[dict[str, Any]]) -> bool:
    """Plans that must not satisfy make-a-car/house goals."""
    ids = [str(o.get("op") or "") for o in (ops or []) if isinstance(o, dict) and o.get("op")]
    if not ids:
        return True
    if all(op in {"search", "wm.search_operator", "key", "view3d.select"} for op in ids):
        return True
    if not any(op.startswith("mesh.primitive_") for op in ids):
        return True
    return is_shallow_box_ops(ops) or is_click_spam_ops(ops)


def _wrong_silhouette(goal: str, ops: list[dict[str, Any]]) -> bool:
    """True when a plan is geometrically the wrong class of object (house for a piano…)."""
    fam = structure_family(goal)
    kind = object_kind(goal)
    ids = [str(o.get("op") or "") for o in (ops or []) if isinstance(o, dict) and o.get("op")]
    prims = sum(1 for op in ids if op.startswith("mesh.primitive_"))
    cyls = sum(1 for op in ids if op == "mesh.primitive_cylinder_add")
    spheres = sum(
        1
        for op in ids
        if op in {"mesh.primitive_cone_add", "mesh.primitive_ico_sphere_add", "mesh.primitive_uv_sphere_add"}
    )
    long_x = False
    tiny_y = False
    for row in ops or []:
        if str(row.get("op") or "") != "transform.resize":
            continue
        params = row.get("params") if isinstance(row.get("params"), dict) else {}
        try:
            factor = float(params.get("factor") or 0.0)
        except (TypeError, ValueError):
            continue
        axis = str(params.get("axis") or "").lower()
        if axis == "x" and factor >= 2.4:
            long_x = True
        if axis == "y" and factor <= 0.08:
            tiny_y = True
    looks_house = cyls == 0 and tiny_y and prims >= 2
    if fam not in {"building", ""} and looks_house:
        return True
    if fam == "span" or kind in {"golden_gate", "bridge"}:
        return prims < 4 or not (cyls >= 1 or long_x)
    if fam == "tower" or kind == "eiffel":
        return prims < 3
    if fam == "plant" or kind == "tree":
        return cyls < 1 and spheres < 1
    if fam == "pyramid":
        return prims < 2
    if fam in {"vehicle", "aircraft", "vessel", "figure", "furniture", "instrument"}:
        return prims < 3
    if fam == "thing":
        return prims < 3
    return False


def is_click_spam_skill(skill: dict[str, Any]) -> bool:
    return is_click_spam_ops(list(skill.get("ops") or []))


def is_whole_object_skill(skill: dict[str, Any]) -> bool:
    """True for cards named like 'make a house' — never replay these for object goals."""
    label = str(skill.get("goal") or skill.get("name") or "")
    return is_complex_object_goal(label)


def is_shallow_box_skill(skill: dict[str, Any]) -> bool:
    origin = str(skill.get("origin") or "")
    ops_rows = list(skill.get("ops") or [])
    if is_click_spam_skill(skill) or is_search_only_ops(ops_rows):
        return True
    taught = origin in {"recorded", "video", "concept"}
    if taught and not is_complex_object_goal(str(skill.get("goal") or skill.get("name") or "")):
        # Something the user deliberately taught ("make a window": cube → scale →
        # inset → extrude) is a real technique even if it starts from a cube.
        # The cube+≤2-detail heuristic only guards auto-saved *planned* clones.
        return False
    return is_shallow_box_ops(ops_rows)


@dataclass
class Plan:
    ops: list[dict[str, Any]] = field(default_factory=list)
    source: str = ""
    why: str = ""
    unresolved: list[str] = field(default_factory=list)
    used_skills: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return summarize_ops(self.ops, limit=10)

    def to_payload(self) -> dict[str, Any]:
        steps = [
            kb.describe_op(o["op"], o.get("params")) if kb.get(str(o.get("op") or "")) else str(o.get("op"))
            for o in self.ops
        ]
        return {
            "source": self.source,
            "why": self.why,
            "summary": self.summary,
            "used_skills": list(self.used_skills),
            "unresolved": list(self.unresolved),
            "n_ops": len(self.ops),
            "n_steps": len(steps),
            "steps": steps,
            "ops": copy.deepcopy(self.ops),
        }


# --------------------------------------------------------------------------- helpers


def _numbers(text: str) -> list[float]:
    return [float(n) for n in _NUMBER_RE.findall(text or "")]


def _axis(text: str) -> str:
    m = _AXIS_RE.search(text or "")
    return m.group(1).lower() if m else ""


def _repeat(text: str) -> int:
    m = _TIMES_RE.search(text or "")
    if not m:
        return 1
    if m.group(1):
        return max(1, min(48, int(m.group(1))))
    return _WORD_NUM.get(m.group(2).lower(), 1)


def _strip_repeat(text: str) -> str:
    return _TIMES_RE.sub(" ", text or "")


def instantiate_skill(skill: dict[str, Any], overrides: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Copy a skill's ops applying param overrides where the op accepts them."""
    ops = [copy.deepcopy(o) for o in (skill.get("ops") or []) if isinstance(o, dict) and o.get("op")]
    overrides = dict(overrides or {})
    # Recorded Tab presses are toggles; pin them to the mode they reached in the
    # demo so the skill works from either mode (the executor skips no-op switches).
    mode = str(skill.get("start_mode") or "").upper()
    if mode not in {"EDIT", "OBJECT"}:
        mode = "OBJECT"
        for row in ops:
            spec = kb.get(str(row.get("op") or ""))
            if spec and spec.mode in {"EDIT", "OBJECT"}:
                # First mode-specific op tells us where the demo was heading.
                mode = "OBJECT" if (spec.mode == "EDIT") else "EDIT"
                break
    for row in ops:
        if row.get("op") == "object.editmode_toggle":
            params = dict(row.get("params") or {})
            if str(params.get("target") or "").upper() not in {"EDIT", "OBJECT"}:
                mode = "EDIT" if mode == "OBJECT" else "OBJECT"
                params["target"] = mode
                row["params"] = params
            else:
                mode = str(params["target"]).upper()
    if not overrides:
        return ops
    for row in ops:
        spec = kb.get(str(row.get("op") or ""))
        if not spec:
            continue
        params = dict(row.get("params") or {})
        for name, val in overrides.items():
            if name in spec.params:
                params[name] = val
        row["params"] = params
    return ops


def overrides_from_phrase(phrase: str, skill: dict[str, Any]) -> dict[str, Any]:
    """Numbers/axis mentioned in the phrase become overrides for the skill's numeric params."""
    nums = _numbers(_strip_repeat(phrase))
    axis = _axis(phrase)
    out: dict[str, Any] = {}
    numeric_names: list[str] = []
    for row in skill.get("ops") or []:
        spec = kb.get(str(row.get("op") or ""))
        if not spec or not spec.typed:
            continue
        if spec.typed not in numeric_names:
            numeric_names.append(spec.typed)
    if nums and numeric_names:
        out[numeric_names[0]] = nums[0]
    if axis:
        out["axis"] = axis
    return out


def _skill_lookup(phrase: str, skills: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    best = None
    best_ov = 0.0
    complex_goal = is_complex_object_goal(phrase)
    # Open-ended "make a house/car" must be composed from techniques — never
    # replay a prior demo/video stub (those often collapse to F3-searching the goal text).
    if complex_goal:
        return (None, 0.0)
    for sk in skills:
        if not (sk.get("ops") or []):
            continue
        if is_click_spam_skill(sk) or is_search_only_ops(list(sk.get("ops") or [])):
            continue
        if is_whole_object_skill(sk):
            continue
        if is_shallow_box_skill(sk):
            continue
        ov = _match_labels(phrase, skill_match_labels(sk))
        if ov > best_ov:
            best, best_ov = sk, ov
    return (best, best_ov) if best_ov >= _SKILL_MIN else (None, best_ov)


def _technique_skill(skills: list[dict[str, Any]], *names: str) -> dict[str, Any] | None:
    want = {n.lower() for n in names}
    for sk in skills:
        label = str(sk.get("goal") or sk.get("name") or "").strip().lower()
        aliases = {str(a).lower() for a in (sk.get("aliases") or [])}
        if label in want or (want & aliases):
            if sk.get("ops"):
                return sk
    return None


def compose_object_plan(goal: str, skills: list[dict[str, Any]]) -> Plan:
    """Deterministic shape recipes that use atomic ops (not polluted skill cards).

    Geometry is derived from Blender's primitive sizes (cube spans ±1, cylinder
    r=1 depth 2, new objects land on the 3D cursor at the origin, S scales about
    the object origin) so parts sit ON the body instead of inside it. Only
    edit-mode ops that work with everything selected (loop cut, bevel) are used;
    inset/extrude on a fully selected closed mesh is a no-op or a shell copy.
    Length scales with detail words and requested features.
    """
    kind = object_kind(goal)
    fam = structure_family(goal)
    level = goal_detail_level(goal)
    features = set(_goal_features(goal))
    plan = Plan(source="compose", why=f"composed techniques for {kind or 'object'} (detail={level})")

    def use(name: str, op: str, params: dict[str, Any] | None = None) -> None:
        plan.ops.append(new_op(op, dict(params or {})))
        if name and name not in plan.used_skills:
            if _technique_skill(skills, name):
                plan.used_skills.append(name)

    def box(sx: float, sy: float, sz: float, *, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        """Cube with half-extents (sx, sy, sz) centred at (x, y, z)."""
        use("place a cube", "mesh.primitive_cube_add")
        for axis, f in (("x", sx), ("y", sy), ("z", sz)):
            if abs(f - 1.0) > 1e-6:
                use("scale", "transform.resize", {"factor": round(f, 3), "axis": axis})
        for axis, d in (("x", x), ("y", y), ("z", z)):
            if abs(d) > 1e-6:
                use("move", "transform.translate", {"distance": round(d, 3), "axis": axis})

    def cyl(radius: float, half_len: float, *, axis: str = "z", x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        """Cylinder of given radius / half-length along `axis`, centred at (x, y, z)."""
        use("place a cylinder", "mesh.primitive_cylinder_add")
        if axis == "y":
            use("rotate", "transform.rotate", {"angle": 90, "axis": "x"})
        elif axis == "x":
            use("rotate", "transform.rotate", {"angle": 90, "axis": "y"})
        if abs(radius - 1.0) > 1e-6:
            use("scale", "transform.resize", {"factor": round(radius, 3), "axis": ""})
        # After the uniform scale the half-length equals `radius`; fix it along the axis.
        along = round(half_len / max(radius, 1e-6), 3)
        if abs(along - 1.0) > 1e-6:
            use("scale", "transform.resize", {"factor": along, "axis": axis})
        for ax, d in (("x", x), ("y", y), ("z", z)):
            if abs(d) > 1e-6:
                use("move", "transform.translate", {"distance": round(d, 3), "axis": ax})

    def edit_detail(cuts: int, bevel_offset: float, segments: int) -> None:
        """Edit-mode pass that is valid with everything selected."""
        plan.ops.append(new_op("object.editmode_toggle", {"target": "EDIT"}))
        plan.ops.append(new_op("mesh.select_all", {"action": "SELECT"}))
        if cuts > 0:
            use("loop cut", "mesh.loopcut_slide", {"number_cuts": cuts})
        use("bevel", "mesh.bevel", {"offset": bevel_offset, "segments": segments})
        plan.ops.append(new_op("object.editmode_toggle", {"target": "OBJECT"}))

    vehicle = kind in {"car", "truck", "van", "bus", "bike", "motorcycle"} or fam == "vehicle"
    building = kind in {"house", "home", "building", "cabin", "castle"} or fam == "building"
    furniture = kind in {"chair", "table", "desk", "sofa", "bed", "lamp"} or fam == "furniture"

    if fam == "span" or kind in {"golden_gate", "bridge"}:
        # Long deck through two tall pylons; cables as thin X-cylinders; hangers down to the road.
        span = 4.6 if level < 3 else 5.2
        deck_z = 1.35
        box(span, 0.42, 0.09, z=deck_z)
        tower_x = span * 0.48
        tower_h = 2.5 if level < 2 else 2.8
        for tx in (-tower_x, tower_x):
            box(0.18, 0.48, tower_h, x=tx, z=tower_h)
            box(0.18, 0.55, 0.12, x=tx, z=2 * tower_h + 0.12)  # top strut
            if level >= 2:
                box(0.18, 0.5, 0.1, x=tx, z=tower_h)  # mid strut
        cable_z = 2 * tower_h + 0.05
        for cy in (-0.2, 0.2):
            cyl(0.05, span * 0.5, axis="x", y=cy, z=cable_z)
        hangers = (-span * 0.28, 0.0, span * 0.28) if level < 2 else (-span * 0.32, -span * 0.12, span * 0.12, span * 0.32)
        hang_half = (cable_z - deck_z) / 2.0
        hang_z = deck_z + hang_half
        for hx in hangers:
            cyl(0.035, hang_half, x=hx, z=hang_z)
        box(0.35, 0.5, 0.45, x=-(span + 0.2), z=0.45)  # anchorage
        box(0.35, 0.5, 0.45, x=(span + 0.2), z=0.45)
        plan.why = f"suspension bridge towers + deck + cables (L{level})"
    elif kind == "eiffel":
        box(1.6, 1.6, 0.12, z=0.12)  # base
        for lx, ly in [(-0.85, -0.85), (-0.85, 0.85), (0.85, -0.85), (0.85, 0.85)]:
            cyl(0.12, 0.7, x=lx, y=ly, z=0.7)
        box(0.9, 0.9, 0.1, z=1.5)
        box(0.45, 0.45, 0.8, z=2.4)
        cyl(0.08, 0.7, z=3.6)
        plan.why = "Eiffel-style tapering tower"
    elif kind == "pyramid":
        box(2.2, 2.2, 0.35, z=0.35)
        box(1.5, 1.5, 0.35, z=1.05)
        box(0.9, 0.9, 0.35, z=1.75)
        box(0.35, 0.35, 0.28, z=2.28)
        plan.why = "step pyramid"
    elif kind == "tree":
        cyl(0.18, 0.7, z=0.7)
        use("place an ico sphere", "mesh.primitive_ico_sphere_add")
        use("scale", "transform.resize", {"factor": 1.1, "axis": ""})
        use("move", "transform.translate", {"distance": 1.85, "axis": "z"})
        plan.why = "trunk + foliage"
    elif kind == "lighthouse":
        cyl(0.45, 1.3, z=1.3)
        cyl(0.28, 0.35, z=2.95)
        box(0.55, 0.55, 0.08, z=3.38)
        plan.why = "lighthouse tower + lantern"
    elif kind == "windmill":
        cyl(0.4, 1.1, z=1.1)
        for ang_axis, dist, ax in (("y", 1.7, "x"), ("y", -1.7, "x"), ("x", 1.7, "y"), ("x", -1.7, "y")):
            box(1.1, 0.08, 0.18, **{ax: dist}, z=2.2)
        plan.why = "windmill tower + sails"
    elif kind == "liberty":
        box(0.7, 0.7, 0.45, z=0.45)
        cyl(0.35, 0.7, z=1.6)
        box(0.12, 0.12, 0.55, x=0.45, z=2.55)
        use("place a cone", "mesh.primitive_cone_add")
        use("scale", "transform.resize", {"factor": 0.18, "axis": ""})
        use("move", "transform.translate", {"distance": 3.2, "axis": "z"})
        plan.why = "statue pedestal + figure + torch"
    elif fam == "tower":
        box(0.55, 0.55, 1.3, z=1.3)
        box(0.38, 0.38, 0.85, z=2.6 + 0.15)
        cyl(0.1, 0.55, z=3.7)
        plan.why = f"tall stacked {goal_subject(goal)}"
    elif fam == "aircraft" or kind == "plane":
        cyl(0.28, 1.5, axis="x")
        box(0.12, 1.9, 0.05, z=0.05)
        box(0.45, 0.08, 0.28, x=-1.45, z=0.35)
        plan.why = "fuselage + wings + tail"
    elif fam == "vessel" or kind in {"boat", "ship"}:
        box(2.3, 0.55, 0.28, z=0.28)
        box(0.7, 0.4, 0.32, x=-0.35, z=0.72)
        cyl(0.08, 0.45, x=0.6, z=0.95)
        plan.why = "hull + cabin"
    elif vehicle:
        tall = kind in {"truck", "van", "bus"}
        bx, by, bz = (2.4 if level < 3 else 2.8), 1.1, (0.55 if tall else 0.4)
        box(bx, by, bz)  # body centred at origin, bottom at -bz
        edit_detail(cuts=1 if level == 0 else min(4, level + 1), bevel_offset=0.05, segments=1 if level < 2 else 3)
        # Cabin sits ON the body (bottom of cabin = top of body).
        cab_h = 0.3 if tall else 0.35
        cab_len = bx * (0.7 if tall else 0.45)
        box(cab_len, by * 0.88, cab_h, x=(0.0 if tall else -bx * 0.12), z=bz + cab_h)
        if level >= 2 or "windows" in features:
            edit_detail(cuts=1, bevel_offset=0.08, segments=2)
        want_wheels = kind in {"car", "truck", "van", "bus", "bike", "motorcycle"} or "wheels" in features
        if want_wheels:
            r = 0.42 if not tall else 0.5
            positions = [(-bx * 0.62, -by), (-bx * 0.62, by), (bx * 0.62, -by), (bx * 0.62, by)]
            if kind in {"bike", "motorcycle"}:
                positions = [(-bx * 0.6, 0.0), (bx * 0.6, 0.0)]
            for wx, wy in positions:
                # Axle along Y, wheel centre at the body's bottom edge, poking out sideways.
                cyl(r, 0.15, axis="y", x=wx, y=(wy + (0.1 if wy > 0 else -0.1) if wy else 0.0), z=-bz - 0.05)
        if level >= 2 or "bumper" in features:
            for sign in (-1.0, 1.0):
                box(0.1, by * 1.02, 0.12, x=sign * (bx + 0.1), z=-bz * 0.5)
        if level >= 3 or "hood" in features:
            box(bx * 0.28, by * 0.9, 0.03, x=bx * 0.6, z=bz + 0.03)  # hood plate
        plan.why = f"vehicle body + cabin (L{level})" + (" + wheels" if want_wheels else "")
    elif building:
        hx, hy = (1.4 if level < 3 else 1.8), (1.2 if level < 3 else 1.5)
        hz = {0: 1.2, 1: 1.5, 2: 2.0, 3: 2.6}.get(level, 1.5)  # half-height
        box(hx, hy, hz, z=hz)  # walls, bottom on the ground
        cuts = 1 if level <= 1 else (2 if level == 2 else 3)
        if "windows" in features or "doors" in features:
            cuts = max(cuts, 2)
        edit_detail(cuts=cuts, bevel_offset=0.02 if level < 2 else 0.03, segments=1 if level < 2 else 2)
        roof_t = 0.2
        top = 2 * hz
        # Roof slab resting on the walls.
        box(hx + 0.2, hy + 0.2, roof_t, z=top + roof_t)
        if level >= 1 or "doors" in features:
            box(0.32, 0.05, 0.6, y=-(hy + 0.05), z=0.6)  # front door on -Y face
        if level >= 2 or "windows" in features:
            for wx in (-hx * 0.55, hx * 0.55):
                box(0.3, 0.05, 0.3, x=wx, y=-(hy + 0.05), z=hz + 0.3)
        if level >= 2 or "chimney" in features:
            box(0.2, 0.2, 0.45, x=hx * 0.5, y=hy * 0.3, z=top + 2 * roof_t + 0.3)
        if level >= 2 or "garage" in features:
            gz = 0.85
            box(1.0, hy * 0.8, gz, x=hx + 1.0, z=gz)
            box(1.2, hy * 0.8 + 0.15, 0.12, x=hx + 1.0, z=2 * gz + 0.12)  # garage roof
        if level >= 3 or "balcony" in features:
            box(0.9, 0.35, 0.05, y=-(hy + 0.35), z=hz + 0.05)
            box(0.9, 0.03, 0.3, y=-(hy + 0.7), z=hz + 0.35)  # railing
        if level >= 3 or "stairs" in features:
            for i in range(3):
                box(0.5, 0.15, 0.08, y=-(hy + 0.15 + i * 0.3), z=0.24 - i * 0.08)
        plan.why = f"walls + roof + parts (L{level}, features={sorted(features) or ['base']})"
    elif furniture:
        if kind in {"table", "desk"}:
            top_z, leg_r = 0.9, 0.06
            box(1.4, 0.8, 0.05, z=top_z)  # top: 0.85..0.95
            legs = [(-1.25, -0.65), (-1.25, 0.65), (1.25, -0.65), (1.25, 0.65)]
            for lx, ly in legs:
                cyl(leg_r, (top_z - 0.05) / 2, x=lx, y=ly, z=(top_z - 0.05) / 2)
            if kind == "desk" and level >= 1:
                box(0.4, 0.7, 0.35, x=0.95, z=0.35)  # drawer block
        elif kind in {"sofa", "bed"}:
            w = 1.6 if kind == "sofa" else 1.0
            d = 0.8 if kind == "sofa" else 2.0
            box(w, d, 0.25, z=0.25)  # base
            box(w, 0.15, 0.4, y=d - 0.15, z=0.5 + 0.4)  # back / headboard
            if kind == "sofa":
                for sign in (-1.0, 1.0):
                    box(0.15, d, 0.2, x=sign * (w - 0.15), z=0.5 + 0.2)  # armrests
        elif kind == "lamp":
            box(0.3, 0.3, 0.03, z=0.03)  # base
            cyl(0.04, 0.6, z=0.66)  # pole
            use("place a cone", "mesh.primitive_cone_add")
            use("scale", "transform.resize", {"factor": 0.35, "axis": ""})
            use("move", "transform.translate", {"distance": 1.55, "axis": "z"})
        else:  # chair
            seat_z = 0.5
            box(0.45, 0.45, 0.05, z=seat_z)  # seat 0.45..0.55
            box(0.45, 0.05, 0.45, y=-0.4, z=seat_z + 0.5)  # back
            for lx, ly in [(-0.38, -0.38), (-0.38, 0.38), (0.38, -0.38), (0.38, 0.38)]:
                cyl(0.04, (seat_z - 0.05) / 2, x=lx, y=ly, z=(seat_z - 0.05) / 2)
        if level >= 2:
            edit_detail(cuts=0, bevel_offset=0.02, segments=2)
        plan.why = f"furniture parts for {kind} (L{level})"
    elif fam == "figure" or kind in {"robot", "character", "person", "human", "animal"}:
        box(0.32, 0.2, 0.5, z=1.05)  # torso
        use("place an ico sphere", "mesh.primitive_ico_sphere_add")
        use("scale", "transform.resize", {"factor": 0.32, "axis": ""})
        use("move", "transform.translate", {"distance": 1.72, "axis": "z"})
        for lx in (-0.16, 0.16):
            cyl(0.07, 0.38, x=lx, z=0.38)
        cyl(0.06, 0.35, x=0.42, z=1.15)
        cyl(0.06, 0.35, x=-0.42, z=1.15)
        plan.why = f"figure silhouette for {goal_subject(goal)}"
    elif fam == "instrument":
        box(1.4, 0.5, 0.65, z=0.85)
        for lx, ly in [(-1.15, -0.35), (-1.15, 0.35), (1.15, -0.35), (1.15, 0.35)]:
            cyl(0.06, 0.28, x=lx, y=ly, z=0.28)
        box(1.4, 0.22, 0.08, y=-0.62, z=0.85)
        plan.why = f"instrument body + parts ({goal_subject(goal)})"
    elif fam == "round":
        use("place a UV sphere", "mesh.primitive_uv_sphere_add")
        use("scale", "transform.resize", {"factor": 1.1, "axis": ""})
        use("move", "transform.translate", {"distance": 1.1, "axis": "z"})
        cyl(0.25, 0.12, z=0.12)
        plan.why = f"round volume for {goal_subject(goal)}"
    else:
        g = (goal or "").lower()
        tall = bool(re.search(r"\b(tall|high|vertical)\b", g))
        long = bool(re.search(r"\b(long|wide|horizontal)\b", g))
        sx, sy, sz = 1.15, 0.75, 0.7
        if tall:
            sx, sy, sz = 0.5, 0.5, 1.8
        if long:
            sx, sy, sz = 2.4, 0.55, 0.4
        box(sx, sy, sz, z=sz)
        box(max(0.25, sx * 0.35), max(0.2, sy * 0.4), max(0.15, sz * 0.28), z=2 * sz + max(0.15, sz * 0.28))
        cyl(0.1 if not tall else 0.08, 0.35, x=sx + 0.12, z=0.45)
        plan.why = f"assembled silhouette for {goal_subject(goal)}"

    return plan


_STOPWORDS = {
    "and", "then", "it", "its", "the", "a", "an", "please", "now", "also", "with", "out", "of", "from",
    "this", "that", "them", "there", "so", "just", "one", "to", "on", "in", "at", "up", "all", "some",
    "me", "my", "make", "build", "create", "model", "design", "sculpt",
}

_KIND_HINTS: dict[str, tuple[str, ...]] = {
    # Structural parts + at most one common finish op. Extra techniques only if the goal names them.
    "house": ("roof", "window", "door", "wall", "brick", "chimney", "bevel"),
    "home": ("roof", "window", "door", "wall", "bevel"),
    "building": ("roof", "window", "door", "wall", "bevel"),
    "cabin": ("roof", "door", "log", "bevel"),
    "car": ("wheel", "cabin", "body", "mirror", "bevel"),
    "truck": ("wheel", "cabin", "body", "bevel"),
    "chair": ("leg", "seat", "bevel"),
    "table": ("leg", "top", "bevel"),
    "golden_gate": ("tower", "cable", "deck", "span", "bevel", "material"),
    "bridge": ("tower", "cable", "deck", "span", "bevel"),
    "eiffel": ("tower", "leg", "spire"),
    "tree": ("trunk", "leaf", "cylinder"),
    "instrument": ("leg", "body", "keys", "bevel"),
    "figure": ("arm", "leg", "head", "bevel"),
    "aircraft": ("wing", "body", "tail", "bevel"),
    "vessel": ("hull", "cabin", "bevel"),
    "span": ("tower", "cable", "deck", "bevel"),
    "tower": ("spire", "bevel"),
    "vehicle": ("wheel", "cabin", "body", "bevel"),
    "furniture": ("leg", "seat", "bevel"),
    "plant": ("trunk", "leaf"),
    "pyramid": ("step", "bevel"),
    "round": ("sphere", "bevel"),
    # Unknown nouns: silhouette only — do NOT hint every modeling op.
    "thing": ("body",),
}

# Modeling vocabulary that applies when the goal itself names the technique.
_UNIVERSAL_TECHNIQUE_HINTS: tuple[str, ...] = (
    "bevel",
    "extrude",
    "inset",
    "loop cut",
    "solidify",
    "subdivide",
    "shade smooth",
    "mirror",
    "array",
    "boolean",
    "knife",
    "bridge edge",
    "bridge loop",
    "fill",
    "duplicate",
    "scale",
    "move",
    "rotate",
)

# Nouns that name the object being built — must not score as technique overlap
# (e.g. "make a bridge" must not match skill "Bridge edge loops").
_OBJECT_NOUNS = {
    "car", "truck", "van", "bus", "bike", "boat", "ship", "plane", "airplane", "aircraft", "jet",
    "house", "home", "building", "cabin", "tower", "castle", "bridge", "chair", "table", "desk",
    "sofa", "bed", "lamp", "robot", "character", "person", "human", "animal", "tree", "sword",
    "gun", "weapon", "piano", "guitar", "spaceship", "rocket", "lighthouse", "windmill", "pyramid",
    "gate", "span", "viaduct", "overpass",
}


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", (text or "").lower()) if w not in _STOPWORDS]


def _goal_color(goal: str) -> str:
    return kb.color_name_in_text(goal or "")


def is_technique_skill(skill: dict[str, Any]) -> bool:
    """True for transferable Blender how-to cards (video/record concepts), not whole objects."""
    if is_whole_object_skill(skill) or is_click_spam_skill(skill):
        return False
    if is_search_only_ops(list(skill.get("ops") or [])):
        return False
    origin = str(skill.get("origin") or "").lower()
    if origin in {"video", "concept", "recorded"}:
        return True
    blob = " ".join(skill_match_labels(skill)).lower()
    if any(h in blob for h in _UNIVERSAL_TECHNIQUE_HINTS):
        return True
    toks = set(_content_words(blob))
    return bool(toks & _TECHNIQUE_WORDS)


def _skill_score(goal: str, skill: dict[str, Any]) -> float:
    """Rank skills for a goal. Require real overlap — do not boost every technique onto every object."""
    labels = skill_match_labels(skill)
    blob = " ".join(labels).lower()
    gtoks = set(_content_words(goal))
    stoks = set(_content_words(blob))
    shared = gtoks & stoks
    # Homonym trap: object noun in the goal matching a technique card name.
    if is_complex_object_goal(goal) and is_technique_skill(skill):
        shared -= _OBJECT_NOUNS
        kind = object_kind(goal)
        fam = structure_family(goal)
        if kind:
            shared -= set(_content_words(kind.replace("_", " ")))
        if fam:
            shared -= set(_content_words(fam))
    overlap = float(len(shared))
    score = overlap
    kind = object_kind(goal)
    fam = structure_family(goal)
    hints: set[str] = set()
    for key in (kind, fam):
        for word in _KIND_HINTS.get(key, ()):
            hints.update(_content_words(word))
    # Hints are finish/part words — never the object noun itself.
    hints -= _OBJECT_NOUNS
    # Only add technique hints the goal actually names — not the whole universal set.
    g_low = (goal or "").lower()
    _named_tech = (
        (r"\bbevel\b", ("bevel",)),
        (r"\bextrude\b", ("extrude",)),
        (r"\binset\b", ("inset",)),
        (r"\bloop\s*cuts?\b", ("loop", "cut", "loopcut")),
        (r"\bsolidify\b", ("solidify",)),
        (r"\bsubdivide\b", ("subdivide",)),
        (r"\bshade\s*smooth\b|\bsmooth\b", ("shade", "smooth")),
        (r"\bmirror\b", ("mirror",)),
        (r"\barray\b", ("array",)),
        (r"\bboolean\b", ("boolean",)),
        (r"\bknife\b", ("knife",)),
        (r"\bbridge\s+(?:edge|loop)", ("bridge", "edge", "loop")),
        (r"\bmaterial\b|\bshader\b", ("material", "shader")),
    )
    for pat, words in _named_tech:
        if re.search(pat, g_low):
            for word in words:
                hints.update(_content_words(word))
    if _goal_color(goal):
        hints.update({"material", "color", "shader", "paint"})
    hint_hits = float(len(hints & stoks))
    score += 0.45 * hint_hits
    label_hit = _match_labels(goal, labels)
    # Technique cards must not win on object-noun label overlap alone
    # ("make a bridge" must not match "Bridge edge loops" / skill "bridge").
    if is_complex_object_goal(goal) and is_technique_skill(skill):
        skill_sig = stoks - _OBJECT_NOUNS
        goal_sig = gtoks - _OBJECT_NOUNS
        if not (skill_sig & (goal_sig | hints)):
            label_hit = 0.0
    score += label_hit
    # Small boost only when the card already shares words with the goal/hints.
    if is_complex_object_goal(goal) and is_technique_skill(skill) and (overlap > 0 or hint_hits > 0):
        score += 0.35
        origin = str(skill.get("origin") or "").lower()
        if origin in {"video", "concept"}:
            score += 0.15
    return score


def _rank_skills(goal: str, skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = [(_skill_score(goal, sk), i, sk) for i, sk in enumerate(skills)]
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [sk for _sc, _i, sk in scored]


def _graftable_ops(ops: list[dict[str, Any]], have: set[str]) -> list[dict[str, Any]]:
    extra: list[dict[str, Any]] = []
    skip = _SHALLOW_SKIP | {"key", "search", "wm.search_operator", "view3d.select"}
    for row in ops:
        op = str(row.get("op") or "")
        if not op or op in have or op in skip:
            continue
        extra.append(row)
    useful = [
        o
        for o in extra
        if str(o.get("op") or "")
        not in {"transform.resize", "transform.translate", "transform.rotate"}
        and not str(o.get("op") or "").startswith("mesh.primitive_")
    ]
    return extra if useful else []


def graft_learned_techniques(plan: Plan, goal: str, skills: list[dict[str, Any]]) -> Plan:
    """Apply only goal-relevant taught techniques (not every video card)."""
    if not plan.ops:
        return plan
    color = _goal_color(goal)
    if not color and object_kind(goal) == "golden_gate":
        color = "orange"
    have = {str(o.get("op") or "") for o in plan.ops}
    if color and "material.set" not in have:
        if "object.editmode_toggle" in have:
            plan.ops.append(new_op("object.editmode_toggle", {"target": "OBJECT"}))
        plan.ops.append(new_op("object.select_all", {"action": "SELECT"}))
        plan.ops.append(new_op("material.set", {"color": color, "name": color.title()}))
        have.add("material.set")
        tag = f"{color} material"
        if tag not in plan.used_skills:
            plan.used_skills.append(tag)
        if "material" not in plan.why.lower() and color not in plan.why.lower():
            plan.why = (plan.why + f"; {color} material").strip("; ")

    complex_goal = is_complex_object_goal(goal)
    # Keep grafts sparse — stacking 8 unrelated techniques made Runs look hallucinated.
    graft_limit = 3 if complex_goal else 2
    grafted = 0
    used_l = {str(n).strip().lower() for n in plan.used_skills}
    for sk in _rank_skills(goal, skills):
        if grafted >= graft_limit:
            break
        if not (sk.get("ops") or []):
            continue
        if is_whole_object_skill(sk) or is_click_spam_skill(sk) or is_search_only_ops(list(sk.get("ops") or [])):
            continue
        if is_shallow_box_skill(sk):
            continue
        name = str(sk.get("goal") or sk.get("name") or "").strip()
        if not name or name.lower() in used_l:
            continue
        # Require real relevance (word/hint overlap), not a free technique boost.
        min_score = 0.75 if complex_goal else 0.85
        if _skill_score(goal, sk) < min_score:
            continue
        extra = _graftable_ops(instantiate_skill(sk), have)
        if not color:
            # Don't force a painted material onto colorless goals.
            extra = [r for r in extra if str(r.get("op") or "") != "material.set"]
        if not extra:
            continue
        # Keep edit-mode technique packs coherent: ensure EDIT before mesh edits.
        first_op = str(extra[0].get("op") or "")
        if first_op.startswith("mesh.") and first_op not in {
            "mesh.primitive_cube_add",
            "mesh.primitive_cylinder_add",
            "mesh.primitive_cone_add",
            "mesh.primitive_uv_sphere_add",
            "mesh.primitive_ico_sphere_add",
            "mesh.primitive_plane_add",
            "mesh.primitive_torus_add",
        }:
            if "object.editmode_toggle" not in have:
                plan.ops.append(new_op("object.editmode_toggle", {"target": "EDIT"}))
                plan.ops.append(new_op("mesh.select_all", {"action": "SELECT"}))
                have.add("object.editmode_toggle")
                have.add("mesh.select_all")
        plan.ops.extend(extra[:4])
        for row in extra[:4]:
            have.add(str(row.get("op") or ""))
        plan.used_skills.append(name)
        used_l.add(name.lower())
        grafted += 1
    if grafted and "learned" not in (plan.why or "").lower():
        plan.why = (plan.why + f"; applied {grafted} learned techniques").strip("; ")
    return plan


def _rule_ops_for_phrase(phrase: str) -> tuple[list[dict[str, Any]], str]:
    """Knowledge-base parse of one phrase (no skills). Returns (ops, unresolved_text)."""
    p = phrase.strip().lower()
    if not p:
        return [], ""
    out: list[dict[str, Any]] = []
    nums = _numbers(_strip_repeat(p))
    axis = _axis(p)

    # Primitives: "add a cube", "3 cubes", "a sphere of size 2", "cube"
    m_add = _ADD_RE.search(p)
    if m_add:
        verb, count_raw, prim_word, size_raw = m_add.group(1), m_add.group(2), m_add.group(3), m_add.group(4)
        prim = kb.primitive_for_word(prim_word) or ("mesh.primitive_uv_sphere_add" if "sphere" in prim_word else None)
        rest_words = (p[: m_add.start()] + " " + p[m_add.end():]).split()
        bare = all(w in {"the", "a", "an", "one", "please"} for w in rest_words)
        counted = bool(count_raw) and count_raw.lower() not in {"a", "an"}
        if prim and (verb or bare or counted):
            count = 1
            if count_raw:
                count = int(count_raw) if count_raw.isdigit() else _WORD_NUM.get(count_raw.lower(), 1)
            for _ in range(max(1, min(48, count))):
                out.append(new_op(prim))
                if size_raw:
                    size = float(size_raw)
                    if size > 0 and abs(size - 1.0) > 1e-6:
                        out.append(new_op("transform.resize", {"factor": size, "axis": ""}))
            p = (p[: m_add.start()] + " " + p[m_add.end():]).strip()
            nums = _numbers(_strip_repeat(p))
            axis = _axis(p)
            if not _content_words(p):
                return out, ""

    op = kb.op_for_phrase(p)
    spec = kb.get(op) if op else None
    if spec is None or op.startswith("mesh.primitive_"):
        # Nothing actionable in the remainder: report it so the LLM can interpret it.
        leftover = " ".join(_content_words(p))
        return out, leftover
    params: dict[str, Any] = {}
    if spec.typed:
        if nums:
            params[spec.typed] = nums[0]
        elif op == "transform.resize" and re.search(r"\b(half|smaller|shrink)\b", p):
            params["factor"] = 0.5
        elif op == "transform.resize" and re.search(r"\b(double|bigger|larger|grow)\b", p):
            params["factor"] = 2.0
    if spec.axis_param and axis:
        params[spec.axis_param] = axis
    if op == "mesh.select_mode":
        params["type"] = "VERT" if "vert" in p else "EDGE" if "edge" in p else "FACE"
    if op in {"mesh.select_all", "object.select_all"}:
        params["action"] = "DESELECT" if "deselect" in p else "SELECT"
    if op == "object.subdivision_set" and nums:
        params["level"] = int(nums[0])
    if op == "mesh.loopcut_slide" and nums:
        params["number_cuts"] = int(nums[0])
    if op == "object.editmode_toggle":
        params["target"] = "EDIT" if "edit" in p else "OBJECT" if "object" in p else ""
    if op == "mesh.delete":
        params["type"] = "VERT" if "vert" in p else "EDGE" if "edge" in p else "FACE"
    if op == "object.modifier_add":
        for key in ("mirror", "array", "solidify", "bevel", "boolean", "subsurf"):
            if key in p:
                params["type"] = key.upper()
    out.append(new_op(op, params))
    return out, ""


def plan_with_rules(goal: str, skills: list[dict[str, Any]]) -> Plan:
    plan = Plan(source="rules")
    for phrase in split_goal_phrases(goal):
        times = _repeat(phrase)
        skill, _ov = _skill_lookup(_strip_repeat(phrase), skills)
        ops: list[dict[str, Any]] = []
        if skill is not None:
            ops = instantiate_skill(skill, overrides_from_phrase(phrase, skill))
            name = str(skill.get("goal") or skill.get("name") or "")
            if name and name not in plan.used_skills:
                plan.used_skills.append(name)
        leftover = ""
        if not ops:
            # "add a cube 3 times": the repeat words are handled above, not left over.
            ops, leftover = _rule_ops_for_phrase(_strip_repeat(phrase))
        if not ops:
            plan.unresolved.append(phrase)
            continue
        if leftover:
            plan.unresolved.append(leftover)
        for _ in range(times):
            plan.ops.extend(copy.deepcopy(ops))
    plan.why = "matched learned skills + Blender knowledge" if plan.ops else "no rule matched"
    return plan


# --------------------------------------------------------------------------- llm


def _skills_block(skills: list[dict[str, Any]], limit: int = 48, *, goal: str = "") -> str:
    ranked = _rank_skills(goal, skills) if goal else skills
    lines: list[str] = []
    for sk in ranked[:limit]:
        name = str(sk.get("goal") or sk.get("name") or "").strip()
        if not name or not sk.get("ops"):
            continue
        aliases = ", ".join(str(a) for a in (sk.get("aliases") or [])[:4])
        lines.append(
            json.dumps(
                {
                    "skill": name,
                    "aliases": aliases,
                    "summary": str(sk.get("summary") or "")[:160],
                    "params": sk.get("params") or {},
                    "ops": [{"op": o.get("op"), "params": o.get("params") or {}} for o in (sk.get("ops") or [])[:10]],
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines) if lines else "(none yet)"


def _validate_llm_plan(
    data: dict[str, Any],
    skills: list[dict[str, Any]],
    *,
    max_items: int = 48,
) -> Plan:
    plan = Plan(source="llm", why=str(data.get("why") or "")[:160])
    items = data.get("plan") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return plan
    by_name = {str(s.get("goal") or s.get("name") or "").lower(): s for s in skills if s.get("ops")}
    for item in items[: max(1, int(max_items))]:
        if not isinstance(item, dict):
            continue
        params = item.get("params") if isinstance(item.get("params"), dict) else {}
        if item.get("skill"):
            key = str(item["skill"]).strip().lower()
            sk = by_name.get(key)
            if sk is None:
                sk, _ = _skill_lookup(key, skills)
            if sk is None:
                continue
            plan.ops.extend(instantiate_skill(sk, params))
            name = str(sk.get("goal") or sk.get("name") or "")
            if name not in plan.used_skills:
                plan.used_skills.append(name)
            continue
        op = kb.normalize_idname(str(item.get("op") or ""))
        spec = kb.get(op)
        if spec is None:
            guess = kb.op_for_phrase(str(item.get("op") or "").replace("_", " ").replace(".", " "))
            spec = kb.get(guess) if guess else None
            op = guess or ""
        if spec is None:
            continue
        clean: dict[str, Any] = {}
        for name in spec.params:
            if name in params and params[name] not in (None, ""):
                val = _clean_param(name, params[name])
                if val is not None:
                    clean[name] = val
        plan.ops.append(new_op(op, clean))
    return plan


# Sanity ranges for LLM-supplied parameters (Blender units / degrees / counts).
_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "factor": (0.05, 20.0),
    "distance": (-50.0, 50.0),
    "thickness": (0.0, 5.0),
    "offset": (0.0, 5.0),
    "angle": (-360.0, 360.0),
    "segments": (1, 30),
    "number_cuts": (1, 30),
    "level": (1, 5),
    "steps": (3, 128),
}
_ENUM_PARAMS = {
    "axis": {"x", "y", "z", ""},
    "target": {"EDIT", "OBJECT"},
    "action": {"SELECT", "DESELECT", "INVERT", "TOGGLE"},
}


def _clean_param(name: str, raw: Any) -> Any:
    """Validate one LLM parameter; None means 'drop it'."""
    if name == "axis":
        val = str(raw).strip().lower()
        return val if val in _ENUM_PARAMS["axis"] else None
    if name in {"target", "action"}:
        val = str(raw).strip().upper()
        if name == "target" and val in {"EDITMODE", "EDIT_MODE", "EDIT MODE"}:
            val = "EDIT"
        if name == "target" and val in {"OBJECTMODE", "OBJECT_MODE", "OBJECT MODE"}:
            val = "OBJECT"
        return val if val in _ENUM_PARAMS[name] else None
    if name in {"type", "name", "text", "color"}:
        if name == "color" and isinstance(raw, (list, tuple)):
            try:
                return [max(0.0, min(1.0, float(x))) for x in list(raw)[:3]]
            except (TypeError, ValueError):
                return None
        return str(raw).strip().upper() if name == "type" else str(raw).strip()
    try:
        num = float(raw)
    except (TypeError, ValueError):
        return None
    if num != num:  # NaN
        return None
    lo, hi = _PARAM_RANGES.get(name, (-1e6, 1e6))
    num = max(lo, min(hi, num))
    if name in {"segments", "number_cuts", "level", "steps"}:
        return int(round(num))
    if name == "factor" and abs(num) < 1e-6:
        return None  # scaling by 0 collapses the mesh
    return num


def _salvage_plan_items(raw: str) -> dict[str, Any]:
    """Recover complete {"op"/"skill": ...} items from truncated model output."""
    items: list[dict[str, Any]] = []
    for m in re.finditer(r'\{\s*"(?:op|skill)"\s*:\s*"[^"]*"(?:\s*,\s*"params"\s*:\s*\{[^{}]*\})?\s*\}', raw or ""):
        try:
            items.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
    return {"plan": items, "why": "recovered from truncated response"} if items else {}


def plan_with_llm(
    goal: str,
    skills: list[dict[str, Any]],
    *,
    model: str,
    context: dict[str, Any] | None = None,
    rules_hint: Plan | None = None,
) -> Plan:
    from agent.policy import PolicyError, chat_text_json

    budget = plan_budget(goal)
    ctx = context or {}
    state = (
        f"mode={ctx.get('mode') or 'unknown'} active={ctx.get('active') or '-'} "
        f"objects={ctx.get('objects', '?')} faces={ctx.get('faces', '?')} sel_faces={ctx.get('sel_faces', '?')}"
    )
    hint = ""
    if rules_hint and rules_hint.ops:
        hint = "Partial plan from rules (keep what is right, fix the rest): " + json.dumps(
            [{"op": o.get("op"), "params": o.get("params") or {}} for o in rules_hint.ops][: budget]
        )
    if rules_hint and rules_hint.unresolved:
        hint += "\nUnresolved phrases: " + "; ".join(rules_hint.unresolved[:8])
    sil = silhouette_hint(goal)
    if sil:
        hint = f"SILHOUETTE you must match (use several primitives; do not emit a house):\n{sil}\n" + hint
    user = (
        f"GOAL: {goal}\nBLENDER STATE: {state}\n"
        f"TARGET PLAN LENGTH: about {budget} items (more for advanced/detailed goals; fewer for simple ones).\n\n"
        f"SKILLS:\n{_skills_block(skills, limit=48, goal=goal)}\n\n"
        f"OP CATALOG:\n{json.dumps(kb.tool_catalog(), ensure_ascii=False)}\n\n{hint}\n"
        "Return the JSON plan now."
    )
    # ~50 tokens per plan item plus headroom; longer budgets need more tokens.
    predict = int(min(8192, 320 + 50 * budget))
    try:
        raw = chat_text_json(model, system=PLANNER_SYSTEM, user_text=user, num_predict=predict, temperature=0.15)
    except PolicyError as exc:
        return Plan(source="llm", why=f"planner model unavailable: {str(exc)[:120]}")
    data: Any = {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw or "")
        try:
            data = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            data = _salvage_plan_items(raw)
    if not isinstance(data, dict) or not isinstance(data.get("plan"), list):
        data = _salvage_plan_items(raw) or {}
    return _validate_llm_plan(data if isinstance(data, dict) else {}, skills, max_items=budget)


# --------------------------------------------------------------------------- entry


def make_plan(
    goal: str,
    *,
    skills: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
    model: str = "",
    allow_llm: bool = True,
) -> Plan:
    goal = (goal or "").strip()
    if not goal:
        return Plan(why="empty goal")
    rules = plan_with_rules(goal, skills)
    complex_goal = is_complex_object_goal(goal)

    def _usable(ops: list[dict[str, Any]]) -> bool:
        if not ops:
            return False
        if complex_goal:
            return not is_weak_object_plan(ops)
        return not is_search_only_ops(ops)

    def _finish(plan: Plan) -> Plan:
        return ensure_workspace_ops(graft_learned_techniques(plan, goal, skills), goal, context)

    # Open-ended objects: compose a real silhouette, then let the LLM refine it.
    # Never keep an LLM house/cube when it does not match the requested object.
    if complex_goal and (not _usable(rules.ops) or rules.unresolved):
        composed = compose_object_plan(goal, skills)
        if allow_llm and model and model != "none":
            llm = plan_with_llm(goal, skills, model=model, context=context, rules_hint=composed)
            if _usable(llm.ops) and not _wrong_silhouette(goal, llm.ops):
                return _finish(llm)
        if composed.ops:
            return _finish(composed)

    if _usable(rules.ops) and not rules.unresolved:
        return _finish(rules)
    if allow_llm and model and model != "none":
        llm = plan_with_llm(goal, skills, model=model, context=context, rules_hint=rules)
        if llm.ops:
            if complex_goal and (not _usable(llm.ops) or _wrong_silhouette(goal, llm.ops)):
                composed = compose_object_plan(goal, skills)
                if composed.ops:
                    return _finish(composed)
            if _usable(llm.ops) or not complex_goal:
                llm.used_skills = list(dict.fromkeys(rules.used_skills + llm.used_skills))
                return _finish(llm)
        if _usable(rules.ops):
            rules.why += f"; LLM added nothing ({llm.why})" if llm.why else "; LLM added nothing"
            return _finish(rules)
        if complex_goal:
            return _finish(compose_object_plan(goal, skills))
    if complex_goal:
        return _finish(compose_object_plan(goal, skills))
    return _finish(rules) if rules.ops else rules


def enrich_plan_for_start(plan: Plan, context: dict[str, Any] | None) -> Plan:
    """Make a plan runnable from the current state: select-all before extrude-like ops on fresh geometry."""
    ctx = context or {}
    ops = plan.ops
    out: list[dict[str, Any]] = []
    have_selection = int(ctx.get("sel_faces") or 0) + int(ctx.get("sel_verts") or 0) > 0
    added_primitive = False
    for row in ops:
        op = str(row.get("op") or "")
        spec = kb.get(op)
        if op.startswith("mesh.primitive_"):
            added_primitive = True
            have_selection = False
        if op in {"mesh.select_all", "object.select_all"}:
            # The plan already handles selection here — don't insert a duplicate.
            have_selection = str((row.get("params") or {}).get("action") or "SELECT").upper() != "DESELECT"
            out.append(row)
            continue
        if spec and spec.needs_selection and spec.mode == "EDIT" and not have_selection and added_primitive:
            out.append(new_op("mesh.select_all", {"action": "SELECT"}))
            have_selection = True
        if op == "view3d.select":
            have_selection = True
        out.append(row)
    plan.ops = out
    return plan
