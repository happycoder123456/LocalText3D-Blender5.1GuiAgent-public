"""Distill demos into reusable skills, retarget steps, and compose compound goals."""

from __future__ import annotations

import re
from typing import Any

from agent.window import WindowRect

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SPLIT_RE = re.compile(
    r"\b(?:then|after\s+that|and\s+then|afterwards|,?\s+then)\b|,",
    re.IGNORECASE,
)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _hard_step_count(actions: list[dict[str, Any]]) -> int:
    n = 0
    for a in actions:
        if not isinstance(a, dict):
            continue
        if str(a.get("action") or "") not in {"wait", "stop", ""}:
            n += 1
    return n


def _goals_match(a: str, b: str, *, min_overlap: float = 0.34) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    al = (a or "").strip().lower()
    bl = (b or "").strip().lower()
    if al and bl and (al in bl or bl in al) and min(len(al), len(bl)) >= 4:
        return True
    return len(ta & tb) / float(len(ta | tb)) >= min_overlap


def _action_signature(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    kind = str(action.get("action") or "").lower()
    if kind in {"click", "drag"}:
        x = int(float(action.get("x") or 0))
        y = int(float(action.get("y") or 0))
        return f"{kind}:{x // 8}:{y // 8}:{action.get('target') or ''}"
    if kind in {"key", "hotkey"}:
        keys = action.get("keys") or []
        if isinstance(keys, str):
            keys = [keys]
        norm = ",".join(str(k).lower() for k in keys[:3])
        return f"{kind}:{norm}"
    if kind == "type":
        return f"type:{(action.get('text') or '')[:32].lower()}"
    if kind == "wait":
        return "wait"
    if kind == "stop":
        return "stop"
    return kind


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def estimate_window_size(
    actions: list[dict[str, Any]],
    *,
    fallback: tuple[int, int] = (1600, 900),
) -> tuple[int, int]:
    """Infer a reference window size from absolute click coords if unknown."""
    max_x = 0.0
    max_y = 0.0
    for a in actions:
        if not isinstance(a, dict):
            continue
        if str(a.get("action") or "") not in {"click", "drag"}:
            continue
        max_x = max(max_x, float(a.get("x") or 0), float(a.get("x2") or 0))
        max_y = max(max_y, float(a.get("y") or 0), float(a.get("y2") or 0))
    w = int(max(fallback[0], max_x * 1.05 + 40))
    h = int(max(fallback[1], max_y * 1.05 + 40))
    return w, h


def skill_step_signature(step: dict[str, Any] | None) -> str:
    """Fingerprint for distilled skill steps (normalized clicks)."""
    if not isinstance(step, dict):
        return ""
    kind = str(step.get("action") or "").lower()
    if kind in {"click", "drag"} and step.get("x_norm") is not None:
        xn = int(round(_clamp01(float(step.get("x_norm") or 0)) * 100))
        yn = int(round(_clamp01(float(step.get("y_norm") or 0)) * 100))
        return f"{kind}_n:{xn}:{yn}:{step.get('target') or ''}"
    return _action_signature(step)


def distill_actions(
    actions: list[dict[str, Any]],
    *,
    window_w: int = 0,
    window_h: int = 0,
) -> list[dict[str, Any]]:
    """Turn concrete demo actions into abstract skill steps."""
    if window_w <= 0 or window_h <= 0:
        window_w, window_h = estimate_window_size(actions)
    window_w = max(1, int(window_w))
    window_h = max(1, int(window_h))
    steps: list[dict[str, Any]] = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        kind = str(a.get("action") or "")
        if kind in {"", "stop"}:
            continue
        if kind == "wait":
            seconds = float(max(0.0, min(2.0, float(a.get("seconds") or 0.0))))
            if seconds < 0.05:
                continue
            steps.append(
                {
                    "action": "wait",
                    "seconds": seconds,
                    "reason": str(a.get("reason") or "skill pause"),
                    "from_teacher": True,
                    "stats": {"ok": 0, "fail": 0, "last_reward": 0.0},
                }
            )
            continue
        if kind in {"key", "hotkey", "type"}:
            step = dict(a)
            step["from_teacher"] = True
            step["stats"] = {"ok": 0, "fail": 0, "last_reward": 0.0}
            step["sig"] = skill_step_signature(step)
            steps.append(step)
            continue
        if kind == "click":
            step = {
                "action": "click",
                "intent": "viewport_point",
                "x_norm": _clamp01(float(a.get("x") or 0) / float(window_w)),
                "y_norm": _clamp01(float(a.get("y") or 0) / float(window_h)),
                "target": str(a.get("target") or ""),
                "from_teacher": True,
                "reason": str(a.get("reason") or "skill click"),
                "stats": {"ok": 0, "fail": 0, "last_reward": 0.0},
            }
            step["sig"] = skill_step_signature(step)
            steps.append(step)
            continue
        if kind == "drag":
            step = {
                "action": "drag",
                "intent": "viewport_point",
                "x_norm": _clamp01(float(a.get("x") or 0) / float(window_w)),
                "y_norm": _clamp01(float(a.get("y") or 0) / float(window_h)),
                "x2_norm": _clamp01(float(a.get("x2") or a.get("x") or 0) / float(window_w)),
                "y2_norm": _clamp01(float(a.get("y2") or a.get("y") or 0) / float(window_h)),
                "target": str(a.get("target") or ""),
                "from_teacher": True,
                "reason": str(a.get("reason") or "skill drag"),
                "stats": {"ok": 0, "fail": 0, "last_reward": 0.0},
            }
            duration = float(a.get("duration") or 0.0)
            if duration > 0:
                step["duration"] = min(2.0, duration)
            pts = a.get("points")
            if isinstance(pts, list):
                norms: list[list[float]] = []
                for p in pts:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        norms.append(
                            [
                                _clamp01(float(p[0]) / float(window_w)),
                                _clamp01(float(p[1]) / float(window_h)),
                            ]
                        )
                if len(norms) >= 2:
                    step["points_norm"] = norms[:48]
            step["sig"] = skill_step_signature(step)
            steps.append(step)
            continue
        step = dict(a)
        step["from_teacher"] = True
        step["stats"] = {"ok": 0, "fail": 0, "last_reward": 0.0}
        step["sig"] = skill_step_signature(step)
        steps.append(step)
    return steps


def materialize_step(step: dict[str, Any], rect: WindowRect) -> dict[str, Any]:
    """Map a distilled skill step onto the current Blender window."""
    out = dict(step)
    kind = str(out.get("action") or "")
    w = max(1, int(rect.width))
    h = max(1, int(rect.height))
    if kind == "click" and out.get("x_norm") is not None:
        out["x"] = float(out["x_norm"]) * w
        out["y"] = float(out["y_norm"]) * h
        out["reason"] = str(out.get("reason") or "skill click") + " (retargeted)"
    elif kind == "drag" and out.get("x_norm") is not None:
        out["x"] = float(out["x_norm"]) * w
        out["y"] = float(out["y_norm"]) * h
        out["x2"] = float(out.get("x2_norm") if out.get("x2_norm") is not None else out["x_norm"]) * w
        out["y2"] = float(out.get("y2_norm") if out.get("y2_norm") is not None else out["y_norm"]) * h
        pts = out.get("points_norm")
        if isinstance(pts, list) and len(pts) >= 2:
            out["points"] = [
                [float(p[0]) * w, float(p[1]) * h]
                for p in pts
                if isinstance(p, (list, tuple)) and len(p) >= 2
            ]
        out["reason"] = str(out.get("reason") or "skill drag") + " (retargeted)"
    return out


def snap_click_to_detections(
    action: dict[str, Any],
    detections: list[dict[str, Any]],
    *,
    max_dist: float = 120.0,
) -> dict[str, Any]:
    """If a detection is near the planned click, snap to its center."""
    kind = str(action.get("action") or "")
    if kind not in {"click", "drag"} or not detections:
        return action
    x = float(action.get("x") or 0)
    y = float(action.get("y") or 0)
    best = None
    best_d = max_dist
    for d in detections:
        bbox = d.get("bbox") or [0, 0, 0, 0]
        cx = float(bbox[0]) + float(bbox[2]) / 2.0
        cy = float(bbox[1]) + float(bbox[3]) / 2.0
        dist = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
        if dist < best_d:
            best_d = dist
            best = (cx, cy, str(d.get("class") or ""))
    if best is None:
        return action
    out = dict(action)
    out["x"], out["y"] = best[0], best[1]
    if best[2]:
        out["target"] = best[2]
    out["reason"] = str(out.get("reason") or "click") + " (snapped)"
    return out


def allowed_kinds_for_skill(steps: list[dict[str, Any]]) -> set[str]:
    kinds = {str(s.get("action") or "") for s in steps if isinstance(s, dict)}
    kinds.discard("")
    kinds |= {"wait", "stop"}
    return kinds


def successful_skill_alts(
    steps: list[dict[str, Any]],
    *,
    banned_sigs: set[str] | None = None,
    prefer_kind: str = "",
) -> list[dict[str, Any]]:
    """Steps in this skill with ok > fail, optionally filtered by action kind."""
    banned = banned_sigs or set()
    out: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        kind = str(step.get("action") or "")
        if kind in {"wait", "stop"}:
            continue
        if prefer_kind and kind != prefer_kind:
            continue
        stats = step.get("stats") if isinstance(step.get("stats"), dict) else {}
        ok = int(stats.get("ok") or 0)
        fail = int(stats.get("fail") or 0)
        if ok <= fail:
            continue
        sig = str(step.get("sig") or skill_step_signature(step))
        if sig and sig in banned:
            continue
        out.append(dict(step))
    return out


def bump_step_stats(step: dict[str, Any], reward: float) -> dict[str, Any]:
    out = dict(step)
    stats = dict(out.get("stats") or {"ok": 0, "fail": 0, "last_reward": 0.0})
    if reward >= 0:
        stats["ok"] = int(stats.get("ok") or 0) + 1
    else:
        stats["fail"] = int(stats.get("fail") or 0) + 1
    stats["last_reward"] = float(reward)
    out["stats"] = stats
    return out


def split_goal_phrases(goal: str) -> list[str]:
    parts = [p.strip() for p in _SPLIT_RE.split(goal or "") if p and p.strip()]
    return parts if parts else ([goal.strip()] if (goal or "").strip() else [])


def _alias_list(raw: Any) -> list[str]:
    if isinstance(raw, str):
        return [p.strip() for p in raw.split(",") if p.strip()]
    if isinstance(raw, list):
        return [str(a).strip() for a in raw if str(a).strip()]
    return []


def _match_labels(query: str, labels: list[str]) -> float:
    """Best overlap of query against name/aliases (0..1).

    Short hotkey aliases (X, G, R, E…) must NOT substring-match inside words
    like \"extrude\" / \"bridge\" — that hijacks plans with delete/rotate/grab.
    """
    q = (query or "").strip().lower()
    if not q:
        return 0.0
    qt = _tokens(q)
    best = 0.0
    for label in labels:
        lab = (label or "").strip().lower()
        if not lab:
            continue
        if lab == q:
            best = max(best, 0.95)
            continue
        # Hotkeys / 1–2 char aliases: only exact token match, never \"x\" in \"extrude\".
        if len(lab) <= 2:
            if lab in qt:
                best = max(best, 0.9)
            continue
        if lab in q or q in lab:
            # Require word-ish boundaries for multi-char substring hits.
            if lab in q:
                # Avoid \"extrude\" matching a skill named \"ex\".
                best = max(best, 0.8 if len(lab) >= 4 or lab in qt else 0.0)
            else:
                best = max(best, 0.8 if len(q) >= 4 else 0.0)
            continue
        lt = _tokens(lab)
        if not lt:
            continue
        lt_substantive = {t for t in lt if len(t) > 2}
        if not lt_substantive:
            if lt <= qt:
                best = max(best, 0.85)
            continue
        ov = len(qt & lt) / float(len(qt | lt))
        if lt_substantive <= qt:
            ov = max(ov, 0.75)
        best = max(best, ov)
    return best


def skill_match_labels(skill: dict[str, Any]) -> list[str]:
    """Names a skill/concept can be referred to by."""
    labels: list[str] = []
    for key in ("goal", "name"):
        val = str(skill.get(key) or "").strip()
        if val:
            labels.append(val)
    # Drop 1–2 char hotkey aliases from matching — they substring-hijack goals.
    labels.extend(a for a in _alias_list(skill.get("aliases")) if len(a.strip()) > 2)
    seen: set[str] = set()
    out: list[str] = []
    for lab in labels:
        key = lab.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(lab)
    return out


def compose_skills(
    goal: str,
    known_skills: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Match a compound goal against known skill/concept records (newest-first).

    Matches concept names and aliases (e.g. Goal 'thicken this' → inset/extrude).
    """
    goal = (goal or "").strip()
    if not goal or not known_skills:
        return []

    by_key: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []
    labels_by_key: dict[str, list[str]] = {}
    for sk in known_skills:
        name = str(sk.get("goal") or sk.get("name") or "").strip()
        if not name:
            continue
        key = " ".join(sorted(_tokens(name)))
        if key in by_key:
            continue
        by_key[key] = sk
        ordered_keys.append(key)
        labels_by_key[key] = skill_match_labels(sk)

    phrases = split_goal_phrases(goal)
    composed: list[dict[str, Any]] = []
    used: set[str] = set()
    if len(phrases) >= 2:
        for phrase in phrases:
            best_key = None
            best_ov = 0.0
            for key in ordered_keys:
                if key in used:
                    continue
                ov = _match_labels(phrase, labels_by_key[key])
                if ov > best_ov:
                    best_ov = ov
                    best_key = key
            if best_key and best_ov >= 0.34:
                used.add(best_key)
                composed.append(dict(by_key[best_key]))

    if composed:
        return composed

    best_key = None
    best_score = -1.0
    for key in ordered_keys:
        ov = _match_labels(goal, labels_by_key[key])
        if ov < 0.34:
            continue
        hard = _hard_step_count(list(by_key[key].get("steps") or []))
        score = ov + hard * 0.01
        if score > best_score:
            best_score = score
            best_key = key
    if best_key:
        return [dict(by_key[best_key])]
    return []


def skill_success_rate(skill: dict[str, Any]) -> float:
    steps = skill.get("steps") or []
    ok = fail = 0
    for s in steps:
        if not isinstance(s, dict):
            continue
        stats = s.get("stats") if isinstance(s.get("stats"), dict) else {}
        ok += int(stats.get("ok") or 0)
        fail += int(stats.get("fail") or 0)
    total = ok + fail
    if total <= 0:
        return 0.0
    return float(ok) / float(total)


def precondition_steps(preconditions: Any) -> list[dict[str, Any]]:
    """Turn concept preconditions into Hands steps for the user's mesh."""
    texts: list[str] = []
    if isinstance(preconditions, str) and preconditions.strip():
        texts = [preconditions.strip()]
    elif isinstance(preconditions, list):
        texts = [str(p).strip() for p in preconditions if str(p).strip()]
    if not texts:
        return []

    joined = " ".join(texts).lower()
    steps: list[dict[str, Any]] = []
    needs_edit = any(
        phrase in joined
        for phrase in (
            "edit mode",
            "editmode",
            "face selected",
            "edge selected",
            "vertex selected",
            "faces selected",
            "edges selected",
            "vertices selected",
            "mesh selected",
        )
    )
    if needs_edit:
        steps.append(
            {
                "action": "key",
                "keys": ["tab"],
                "reason": "enter Edit Mode (concept precondition)",
                "from_teacher": True,
                "stats": {"ok": 0, "fail": 0, "last_reward": 0.0},
            }
        )
        steps.append(
            {
                "action": "wait",
                "seconds": 0.25,
                "reason": "wait for Edit Mode",
                "from_teacher": True,
                "stats": {"ok": 0, "fail": 0, "last_reward": 0.0},
            }
        )
    return steps


def materialize_concept_skill(concept: dict[str, Any]) -> dict[str, Any]:
    """Convert a concept card into a skill record the agent loop can run."""
    name = str(concept.get("name") or concept.get("goal") or "").strip()
    steps = [dict(s) for s in (concept.get("steps") or []) if isinstance(s, dict)]
    pre = precondition_steps(concept.get("preconditions"))
    if pre and steps:
        first_keys = [str(k).lower() for k in (steps[0].get("keys") or [])]
        if str(steps[0].get("action")) == "key" and first_keys == ["tab"]:
            pre = []
    return {
        "goal": name,
        "name": name,
        "summary": str(concept.get("summary") or ""),
        "when_to_use": str(concept.get("when_to_use") or ""),
        "preconditions": concept.get("preconditions") or [],
        "aliases": _alias_list(concept.get("aliases")),
        "steps": pre + steps,
        "ops": [o for o in (concept.get("ops") or []) if isinstance(o, dict)],
        "params": concept.get("params") or {},
        "start_mode": str(concept.get("start_mode") or ""),
        "origin": "video",
        "source": "concept",
        "video": str(concept.get("video") or ""),
        "t_start": float(concept.get("t_start") or 0.0),
        "t_end": float(concept.get("t_end") or 0.0),
        "from_video": True,
        "t": float(concept.get("t") or 0.0),
    }
