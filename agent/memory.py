"""Cross-session multi-skill memory: plans, mistakes, and refined recoveries."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from agent.paths import ensure_dataset, memory_path
from agent.skills import _alias_list, distill_actions, materialize_concept_skill


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PLAN_SOURCES = ("refined_plan", "teacher_plan", "agent_plan")
_KEEP_SOURCES = frozenset({*_PLAN_SOURCES, "skill", "concept", "op_stats"})
MAX_JSONL_LINE_BYTES = 1_048_576
MAX_TRANSIENT_ROWS = 8000
MAX_MEMORY_FILE_BYTES = 24_000_000
# Newest teacher demos are ground truth. Refined/agent only win if richer + newer.
_PLAN_SOURCE_BOOST = {
    "teacher_plan": 0.45,
    "refined_plan": 0.2,
    "agent_plan": 0.05,
}


def _hard_step_count(actions: list[dict[str, Any]]) -> int:
    n = 0
    for a in actions:
        if not isinstance(a, dict):
            continue
        if str(a.get("action") or "") not in {"wait", "stop", ""}:
            n += 1
    return n


def _plan_has_pointer(actions: list[dict[str, Any]]) -> bool:
    for a in actions:
        if not isinstance(a, dict):
            continue
        if str(a.get("action") or "") in {"click", "drag"}:
            return True
    return False


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


_JUNK_NAME_BITS = (
    "navigating in blender",
    "selecting and moving",
    "resolution scale",
    "interface tab",
    "system tab",
    "untick clipping",
    "delete key",
    "scale tool",
    "g and z",
    "tab to",
    "tab button",
    "option + click",
    "ctrl r click",
    "click select",
    "right click",
    "material tab",
    "left click",
    "use left click",
    "click on icon",
    "click on drop-down",
    "right click to cancel",
    "right click to add",
)


def _is_junk_skill_name(name: str) -> bool:
    """UI / hotkey tutorial stubs that poison Run plans."""
    low = (name or "").strip().lower()
    if not low or len(low) <= 2:
        return True
    if any(junk in low for junk in _JUNK_NAME_BITS):
        return True
    if re.fullmatch(r"[a-z](?:\s*(?:and|\+|\,)\s*[a-z]){0,3}", low):
        return True
    if re.search(r"\b(left|right)\s+click\b", low) and "loop" not in low:
        return True
    return False


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def goals_match(a: str, b: str, *, min_overlap: float = 0.34) -> bool:
    """True if two goal strings refer to the same skill."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    # Substring either way for short labels ("extrude" vs "extrude the face").
    al = (a or "").strip().lower()
    bl = (b or "").strip().lower()
    if al and bl and (al in bl or bl in al) and min(len(al), len(bl)) >= 4:
        return True
    return _overlap(ta, tb) >= min_overlap


def _concept_key(text: str) -> str:
    """Exact technique identity for video concepts (no fuzzy substring merge)."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def action_signature(action: dict[str, Any] | None) -> str:
    """Compact fingerprint for anti-repeat / ban lists."""
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


def same_action(a: dict[str, Any] | None, b: dict[str, Any] | None, *, eps: float = 12.0) -> bool:
    """True if two actions are effectively the same (for anti-repeat)."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if action_signature(a) == action_signature(b) and action_signature(a):
        return True
    ka = str(a.get("action") or "")
    kb = str(b.get("action") or "")
    if ka != kb:
        return False
    if ka in {"click", "drag"}:
        return (
            abs(float(a.get("x") or 0) - float(b.get("x") or 0)) <= eps
            and abs(float(a.get("y") or 0) - float(b.get("y") or 0)) <= eps
        )
    if ka in {"key", "hotkey"}:
        return [str(x).lower() for x in (a.get("keys") or [])] == [
            str(x).lower() for x in (b.get("keys") or [])
        ]
    if ka == "type":
        return str(a.get("text") or "") == str(b.get("text") or "")
    return False


def _clean_plan_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        kind = str(a.get("action") or "")
        if kind == "stop":
            continue
        if kind == "wait":
            sec = float(a.get("seconds") or 0.0)
            if sec < 0.05:
                continue
            cleaned.append({**dict(a), "seconds": float(max(0.0, min(2.0, sec)))})
        else:
            cleaned.append(dict(a))
    return cleaned


def _with_stop(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [dict(a) for a in actions if isinstance(a, dict)]
    if out and str(out[-1].get("action") or "") != "stop":
        out.append({"action": "stop", "reason": "end of plan"})
    return out


_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve()) if path.exists() else str(path)
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


def _locked(fn):
    """Serialize memory access: the agent thread appends while HTTP threads rewrite."""

    def wrapper(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


class AgentMemory:
    def __init__(self, dataset_root: Path | None = None):
        self.root = ensure_dataset(dataset_root)
        self.path = memory_path(self.root)
        self._lock = _lock_for(self.path)
        # load_all() cache keyed by file (mtime_ns, size); status polls read the
        # file several times per second, so re-parsing on every call is wasteful.
        self._cache_key: tuple[int, int] | None = None
        self._cache_rows: list[dict[str, Any]] = []

    def _file_key(self) -> tuple[int, int] | None:
        try:
            st = self.path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    @_locked
    def count(self) -> int:
        return len(self.load_all())

    def append(
        self,
        *,
        goal: str,
        action: dict[str, Any] | None = None,
        reward: float,
        screenshot_id: str = "",
        reason: str = "",
        source: str = "agent",
        session_id: str = "",
        actions: list[dict[str, Any]] | None = None,
        count: int = 1,
    ) -> None:
        row: dict[str, Any] = {
            "t": time.time(),
            "goal": goal,
            "reward": float(reward),
            "screenshot_id": screenshot_id,
            "reason": reason,
            "source": source,
            "session_id": session_id,
            "count": int(count),
        }
        if actions is not None:
            row["actions"] = actions
            row["sig"] = f"plan:{len(actions)}"
        else:
            row["action"] = action or {}
            row["sig"] = action_signature(action)
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if self._cache_key is not None:
                self._cache_rows.append(row)
                self._cache_key = self._file_key()
            try:
                size = self.path.stat().st_size
            except OSError:
                size = 0
            if size > MAX_MEMORY_FILE_BYTES:
                self._cache_key = None
                compacted = self.load_all()
                self._rewrite(compacted)

    @_locked
    def clear(self) -> None:
        if self.path.is_file():
            self.path.unlink()
        self._cache_key = None
        self._cache_rows = []

    @_locked
    def _rewrite(self, rows: list[dict[str, Any]]) -> None:
        """Atomic rewrite: readers never see a half-written file."""
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)
        self._cache_rows = [dict(r) for r in rows]
        self._cache_key = self._file_key()

    @_locked
    def forget_learning(self, goal: str) -> int:
        """Drop mistakes + auto-saved plans for a goal so a new demo can teach cleanly."""
        rows = self.load_all()
        kept: list[dict[str, Any]] = []
        dropped = 0
        for row in rows:
            source = str(row.get("source") or "")
            if source in {"mistake", "refined_plan", "agent_plan"} and goals_match(
                goal, str(row.get("goal") or "")
            ):
                dropped += 1
                continue
            kept.append(row)
        if dropped:
            self._rewrite(kept)
        return dropped

    def newest_teacher_sigs(self, goal: str) -> set[str]:
        """Signatures from the newest teacher plan for this goal (protected from bans)."""
        newest: tuple[float, list[dict[str, Any]]] | None = None
        for row in self.load_all():
            if str(row.get("source") or "") != "teacher_plan":
                continue
            if not goals_match(goal, str(row.get("goal") or "")):
                continue
            actions = row.get("actions")
            if not isinstance(actions, list) or not actions:
                continue
            t = float(row.get("t") or 0.0)
            if newest is None or t >= newest[0]:
                newest = (t, [a for a in actions if isinstance(a, dict)])
        if newest is None:
            return set()
        out: set[str] = set()
        for action in newest[1]:
            sig = action_signature(action)
            if sig and sig not in {"wait", "stop"}:
                out.add(sig)
        return out

    @_locked
    def load_all(self) -> list[dict[str, Any]]:
        key = self._file_key()
        if key is None:
            self._cache_key = None
            self._cache_rows = []
            return []
        if key == self._cache_key:
            return [dict(r) for r in self._cache_rows]
        rows: list[dict[str, Any]] = []
        transient: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                if len(raw) > MAX_JSONL_LINE_BYTES:
                    continue
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                if str(row.get("source") or "") in _KEEP_SOURCES:
                    rows.append(row)
                else:
                    transient.append(row)
                    if len(transient) > MAX_TRANSIENT_ROWS * 2:
                        transient = transient[-MAX_TRANSIENT_ROWS:]
        if len(transient) > MAX_TRANSIENT_ROWS:
            transient = transient[-MAX_TRANSIENT_ROWS:]
        rows.extend(transient)
        self._cache_rows = rows
        self._cache_key = key
        return [dict(r) for r in rows]

    def list_skills(self, limit: int = 80) -> list[str]:
        """Distinct goals the agent has a skill or plan for (newest first)."""
        seen: set[str] = set()
        out: list[str] = []
        rows = [
            r
            for r in self.load_all()
            if str(r.get("source") or "") in (*_PLAN_SOURCES, "skill", "concept")
        ]
        rows.sort(key=lambda r: float(r.get("t") or 0.0), reverse=True)
        for row in rows:
            goal = str(row.get("goal") or "").strip()
            if not goal:
                continue
            key = " ".join(sorted(_tokens(goal)))
            if key in seen:
                continue
            seen.add(key)
            out.append(goal)
            if len(out) >= limit:
                break
        return out

    def list_skill_records(self, limit: int = 240) -> list[dict[str, Any]]:
        """Newest skill/concept schemas first (full records with steps/stats)."""
        rows = [
            r
            for r in self.load_all()
            if str(r.get("source") or "") in {"skill", "concept"}
        ]
        rows.sort(key=lambda r: float(r.get("t") or 0.0), reverse=True)
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for row in rows:
            goal = str(row.get("goal") or row.get("name") or "").strip()
            key = " ".join(sorted(_tokens(goal)))
            if not key or key in seen:
                continue
            seen.add(key)
            # Concepts become runnable skill-shaped records for compose_skills.
            if str(row.get("source") or "") == "concept":
                out.append(materialize_concept_skill(row))
            else:
                out.append(row)
            if len(out) >= limit:
                break
        return out

    def list_concepts(self, limit: int = 400) -> list[dict[str, Any]]:
        """Concept cards for the N-panel (newest first, deduped by name)."""
        rows = [r for r in self.load_all() if str(r.get("source") or "") == "concept"]
        rows.sort(key=lambda r: float(r.get("t") or 0.0), reverse=True)
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for row in rows:
            name = str(row.get("name") or row.get("goal") or "").strip()
            key = " ".join(sorted(_tokens(name)))
            if not key or key in seen:
                continue
            if _is_junk_skill_name(name):
                continue
            goal_toks = set(_TOKEN_RE.findall(name.lower()))
            try:
                from agent.planner import _OBJECT_NOUNS
            except Exception:
                _OBJECT_NOUNS = set()  # type: ignore
            if goal_toks and goal_toks <= _OBJECT_NOUNS:
                continue
            seen.add(key)
            out.append(
                {
                    "name": name,
                    "summary": str(row.get("summary") or ""),
                    "when_to_use": str(row.get("when_to_use") or ""),
                    "preconditions": row.get("preconditions") or [],
                    "aliases": _alias_list(row.get("aliases")),
                    "steps": row.get("steps") or [],
                    "video": str(row.get("video") or ""),
                    "t_start": float(row.get("t_start") or 0.0),
                    "t_end": float(row.get("t_end") or 0.0),
                    "from_video": True,
                }
            )
            if len(out) >= limit:
                break
        return out

    def concepts_summary(self, limit: int = 200) -> str:
        cards = self.list_concepts(limit=limit)
        if not cards:
            return ""
        bits = []
        for c in cards:
            name = c["name"]
            summary = str(c.get("summary") or "").strip()
            if summary:
                bits.append(f"{name}: {summary}")
            else:
                bits.append(name)
        return " | ".join(bits)

    @_locked
    def save_concept(self, concept: dict[str, Any], *, session_id: str = "") -> bool:
        """Persist one concept card and a runnable skill with the same name."""
        name = str(concept.get("name") or concept.get("goal") or "").strip()
        steps = [s for s in (concept.get("steps") or []) if isinstance(s, dict)]
        if not name or not steps:
            return False
        # Reject object-goal names and search-only stubs — they poison Run plans.
        try:
            from agent.planner import is_complex_object_goal, is_search_only_ops
        except Exception:
            is_complex_object_goal = lambda _t: False  # type: ignore
            is_search_only_ops = lambda _o: False  # type: ignore
        if is_complex_object_goal(name):
            return False
        aliases = _alias_list(concept.get("aliases"))
        summary = str(concept.get("summary") or "").strip()
        when = str(concept.get("when_to_use") or "").strip()
        preconditions = concept.get("preconditions") or []
        video = str(concept.get("video") or "")
        t_start = float(concept.get("t_start") or 0.0)
        t_end = float(concept.get("t_end") or 0.0)
        ops = [o for o in (concept.get("ops") or []) if isinstance(o, dict) and o.get("op")]
        if not ops:
            from agent.abstract import infer_ops_from_actions

            ops = infer_ops_from_actions(steps, start_mode="EDIT" if any("edit" in str(p).lower() for p in preconditions) else "OBJECT")
        if is_search_only_ops(ops):
            return False
        params = concept.get("params") or {}
        if not params and ops:
            from agent.abstract import collect_params

            params = collect_params(ops)

        rows = self.load_all()
        kept: list[dict[str, Any]] = []
        for row in rows:
            src = str(row.get("source") or "")
            g = str(row.get("goal") or row.get("name") or "")
            if src in {"concept", "skill"} and _concept_key(name) == _concept_key(g):
                # Exact technique name only — do NOT fuzzy-merge "extrude roof" into "extrude".
                if src == "concept":
                    old_aliases = _alias_list(row.get("aliases"))
                    for a in old_aliases:
                        if a.lower() not in {x.lower() for x in aliases}:
                            aliases.append(a)
                    if not summary:
                        summary = str(row.get("summary") or "")
                    if not when:
                        when = str(row.get("when_to_use") or "")
                continue
            kept.append(row)

        kept.append(
            {
                "t": time.time(),
                "goal": name,
                "name": name,
                "reward": 0.9,
                "reason": "concept from video" if video else "concept",
                "source": "concept",
                "session_id": session_id,
                "sig": f"concept:{len(steps)}",
                "summary": summary,
                "when_to_use": when,
                "preconditions": preconditions,
                "aliases": aliases,
                "steps": steps,
                "ops": ops,
                "params": params,
                "origin": "video",
                "video": video,
                "t_start": t_start,
                "t_end": t_end,
                "from_video": bool(video),
                "count": 1,
                "screenshot_id": "",
            }
        )
        self._rewrite(kept)
        # Also expose as a skill so older list_skills / teacher_plan paths see it.
        runnable = materialize_concept_skill(
            {
                "name": name,
                "summary": summary,
                "when_to_use": when,
                "preconditions": preconditions,
                "aliases": aliases,
                "steps": steps,
                "video": video,
                "t_start": t_start,
                "t_end": t_end,
            }
        )
        self.save_skill(
            goal=name,
            steps=list(runnable.get("steps") or []),
            session_id=session_id,
        )
        # Attach aliases onto the skill row for compose matching.
        rows2 = self.load_all()
        for i in range(len(rows2) - 1, -1, -1):
            row = rows2[i]
            if str(row.get("source") or "") != "skill":
                continue
            if _concept_key(name) != _concept_key(str(row.get("goal") or "")):
                continue
            row = dict(row)
            row["aliases"] = aliases
            row["summary"] = summary
            row["when_to_use"] = when
            row["preconditions"] = preconditions
            row["from_video"] = bool(video)
            row["ops"] = ops
            row["params"] = params
            row["origin"] = "video"
            rows2[i] = row
            break
        self._rewrite(rows2)
        return True

    def save_concepts(
        self,
        concepts: list[dict[str, Any]],
        *,
        session_id: str = "",
    ) -> int:
        """Save many concept cards; returns how many were written."""
        written = 0
        for raw in concepts:
            if not isinstance(raw, dict):
                continue
            if self.save_concept(raw, session_id=session_id):
                written += 1
        return written

    @_locked
    def save_skill(
        self,
        *,
        goal: str,
        steps: list[dict[str, Any]],
        session_id: str = "",
        window_w: int = 0,
        window_h: int = 0,
    ) -> None:
        """Persist a distilled skill schema (replaces older skill for same goal)."""
        goal = (goal or "").strip()
        if not goal or not steps:
            return
        rows = self.load_all()
        kept: list[dict[str, Any]] = []
        goal_key = _concept_key(goal)
        for row in rows:
            if str(row.get("source") or "") == "skill" and _concept_key(str(row.get("goal") or "")) == goal_key:
                continue
            kept.append(row)
        kept.append(
            {
                "t": time.time(),
                "goal": goal,
                "reward": 0.9,
                "reason": "distilled skill",
                "source": "skill",
                "session_id": session_id,
                "sig": f"skill:{len(steps)}",
                "steps": steps,
                "window_w": int(window_w or 0),
                "window_h": int(window_h or 0),
                "stats": {"ok": 0, "fail": 0, "runs": 0},
                "count": 1,
                "screenshot_id": "",
            }
        )
        self._rewrite(kept)

    # ------------------------------------------------------------ semantics
    _SEM_KEYS = ("ops", "summary", "start_mode", "end_mode", "effects", "params", "origin", "grounded")

    @_locked
    def attach_semantics(self, goal: str, card: dict[str, Any], *, protect_recorded: bool = False) -> bool:
        """Store a semantic skill card (ops + params + modes) on the newest skill for goal.

        Creates the skill row when no distilled skill exists (e.g. planned goals).
        protect_recorded: never overwrite a demo/video skill with a planned variation;
        a differently-worded goal becomes its own skill instead.
        """
        goal = (goal or "").strip()
        ops = [o for o in (card.get("ops") or []) if isinstance(o, dict) and o.get("op")]
        if not goal or not ops:
            return False
        def _skill_key(text: str) -> str:
            return " ".join(sorted(_tokens(text)))

        goal_key = _skill_key(goal)
        rows = self.load_all()
        for i in range(len(rows) - 1, -1, -1):
            row = rows[i]
            if str(row.get("source") or "") != "skill":
                continue
            # Exact skill identity only — fuzzy goals_match would merge
            # "place a cube" into "extrude a cube".
            if _skill_key(str(row.get("goal") or "")) != goal_key:
                continue
            if protect_recorded and str(row.get("origin") or "recorded") in {"recorded", "video"}:
                return False  # the taught skill stays authoritative
            row = dict(row)
            for key in self._SEM_KEYS:
                if key in card:
                    row[key] = card[key]
            aliases = _alias_list(row.get("aliases"))
            for a in _alias_list(card.get("aliases")):
                if a.lower() not in {x.lower() for x in aliases}:
                    aliases.append(a)
            row["aliases"] = aliases
            row["t"] = time.time()
            rows[i] = row
            self._rewrite(rows)
            return True
        rows.append(
            {
                "t": time.time(),
                "goal": goal,
                "reward": 0.9,
                "reason": f"semantic skill ({card.get('origin') or 'planned'})",
                "source": "skill",
                "session_id": "",
                "sig": f"skill:{len(ops)}",
                "steps": list(card.get("steps") or []),
                "aliases": _alias_list(card.get("aliases")),
                "stats": {"ok": 0, "fail": 0, "runs": 0},
                "count": 1,
                "screenshot_id": "",
                **{k: card[k] for k in self._SEM_KEYS if k in card},
            }
        )
        self._rewrite(rows)
        return True

    def semantic_skills(self, limit: int = 240) -> list[dict[str, Any]]:
        """Skill cards with ops (newest first). Legacy rows get ops inferred from their steps."""
        from agent.abstract import infer_ops_from_actions, strip_trailing_ui, summarize_ops
        from agent.planner import is_complex_object_goal, is_search_only_ops

        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in self.list_skill_records(limit=limit * 3):
            goal = str(row.get("goal") or row.get("name") or "").strip()
            key = " ".join(sorted(_tokens(goal)))
            if not key or key in seen:
                continue
            # Poison stubs: video seed named like the Run goal with only F3 search.
            if is_complex_object_goal(goal):
                continue
            low = goal.lower()
            if _is_junk_skill_name(goal):
                continue
            # Bare object nouns ("bridge", "house") are not transferable techniques.
            goal_toks = set(_TOKEN_RE.findall(low))
            try:
                from agent.planner import _OBJECT_NOUNS
            except Exception:
                _OBJECT_NOUNS = set()  # type: ignore
            if goal_toks and goal_toks <= _OBJECT_NOUNS:
                continue
            # Mis-grounded tutorial UI junk that used to inject Tab/E/K mid-plan.
            ops = [o for o in (row.get("ops") or []) if isinstance(o, dict) and o.get("op")]
            if not ops:
                steps = [s for s in (row.get("steps") or []) if isinstance(s, dict)]
                ops = strip_trailing_ui(
                    infer_ops_from_actions(
                        steps,
                        start_mode=str(row.get("start_mode") or "OBJECT"),
                        window_w=int(row.get("window_w") or 0),
                        window_h=int(row.get("window_h") or 0),
                    )
                )
                if not ops:
                    continue
            if is_search_only_ops(ops):
                continue
            # Recorded "extrude" demos that start with Delete are corrupted.
            if str(ops[0].get("op") or "") in {"mesh.delete", "object.delete"} and "delete" not in low:
                continue
            seen.add(key)
            out.append(
                {
                    "goal": goal,
                    "name": goal,
                    "aliases": [a for a in _alias_list(row.get("aliases")) if len(str(a).strip()) > 2],
                    "summary": str(row.get("summary") or "") or summarize_ops(ops),
                    "ops": ops,
                    "params": row.get("params") or {},
                    "start_mode": str(row.get("start_mode") or ""),
                    "origin": str(row.get("origin") or ("video" if row.get("from_video") else "recorded")),
                    "grounded": bool(row.get("grounded")),
                    "steps": row.get("steps") or [],
                    "t": float(row.get("t") or 0.0),
                }
            )
            if len(out) >= limit:
                break
        return out

    def op_stats(self) -> dict[str, dict[str, Any]]:
        for row in reversed(self.load_all()):
            if str(row.get("source") or "") == "op_stats":
                data = row.get("stats")
                return dict(data) if isinstance(data, dict) else {}
        return {}

    @_locked
    def bump_op_stat(self, op: str, variant: int, ok: bool) -> None:
        """Remember whether hotkey (0) or search (1) worked for an op on this machine."""
        if not op:
            return
        rows = self.load_all()
        stats: dict[str, dict[str, Any]] = {}
        idx = None
        for i in range(len(rows) - 1, -1, -1):
            if str(rows[i].get("source") or "") == "op_stats":
                idx = i
                data = rows[i].get("stats")
                stats = dict(data) if isinstance(data, dict) else {}
                break
        entry = dict(stats.get(op) or {})
        # Decay old evidence so an early streak of failures does not pin a
        # variant forever; recent outcomes dominate.
        for k in ("v0_ok", "v0_fail", "v1_ok", "v1_fail"):
            if k in entry:
                entry[k] = round(float(entry.get(k) or 0.0) * 0.85, 3)
        key = f"v{int(variant)}_{'ok' if ok else 'fail'}"
        entry[key] = float(entry.get(key) or 0.0) + 1.0
        stats[op] = entry
        new_row = {"t": time.time(), "goal": "", "reward": 0.0, "source": "op_stats", "sig": "op_stats", "stats": stats}
        if idx is None:
            rows.append(new_row)
        else:
            rows[idx] = new_row
        self._rewrite(rows)

    def preferred_variant(self, op: str) -> int:
        entry = self.op_stats().get(op) or {}
        v0 = float(entry.get("v0_ok") or 0) - float(entry.get("v0_fail") or 0)
        v1 = float(entry.get("v1_ok") or 0) - float(entry.get("v1_fail") or 0)
        return 1 if v1 > v0 else 0

    @_locked
    def update_skill_steps(self, goal: str, steps: list[dict[str, Any]]) -> None:
        """Rewrite the newest matching skill's steps/stats after a Learn run."""
        rows = self.load_all()
        changed = False
        # Exact skill identity (same key as attach_semantics) — fuzzy matching
        # let a run for "extrude" overwrite the "extrude roof" skill.
        goal_key = " ".join(sorted(_tokens(goal)))
        for i in range(len(rows) - 1, -1, -1):
            row = rows[i]
            if str(row.get("source") or "") != "skill":
                continue
            if " ".join(sorted(_tokens(str(row.get("goal") or "")))) != goal_key:
                continue
            row = dict(row)
            row["steps"] = steps
            row["t"] = time.time()
            stats = dict(row.get("stats") or {})
            stats["runs"] = int(stats.get("runs") or 0) + 1
            row["stats"] = stats
            rows[i] = row
            changed = True
            break
        if changed:
            self._rewrite(rows)

    @_locked
    def remember_success(
        self,
        *,
        goal: str,
        action: dict[str, Any],
        reward: float = 0.8,
        reason: str = "",
    ) -> None:
        """Persist a successful action so future runs prefer it."""
        sig = action_signature(action)
        if not sig or sig in {"wait", "stop"}:
            return
        self.append(
            goal=goal,
            action=action,
            reward=float(reward),
            reason=reason or "learned success",
            source="success",
        )

    @_locked
    def ingest_demo(
        self,
        *,
        goal: str,
        actions: list[dict[str, Any]],
        session_id: str = "",
        reward: float = 0.85,
        window_w: int = 0,
        window_h: int = 0,
    ) -> int:
        """Store teacher demo as an ordered plan + distilled skill + success steps.

        Multiple goals/skills are kept side-by-side. Same session_id is not
        re-ingested. Same goal with a new session adds another plan (retrieve
        prefers newest / refined).
        """
        goal = (goal or "").strip() or "teacher demo"
        useful = _clean_plan_actions(actions)
        if not useful:
            return 0

        existing_plans = {
            str(r.get("session_id") or "")
            for r in self.load_all()
            if str(r.get("source") or "") == "teacher_plan" and r.get("session_id")
        }
        is_new_session = bool(session_id) and session_id not in existing_plans
        # A brand-new recording replaces prior auto-learning for this skill.
        if is_new_session or not session_id:
            self.forget_learning(goal)

        written = 0
        if not session_id or session_id not in existing_plans:
            self.append(
                goal=goal,
                reward=float(reward),
                reason="teacher plan",
                source="teacher_plan",
                session_id=session_id,
                actions=useful,
            )
            written += 1
            steps = distill_actions(useful, window_w=window_w, window_h=window_h)
            if steps:
                self.save_skill(
                    goal=goal,
                    steps=steps,
                    session_id=session_id,
                    window_w=window_w,
                    window_h=window_h,
                )
                written += 1

        existing = {
            (str(r.get("session_id") or ""), str(r.get("sig") or ""))
            for r in self.load_all()
            if str(r.get("source") or "") == "teacher"
        }
        for action in useful:
            if str(action.get("action") or "") == "wait":
                continue
            sig = action_signature(action)
            key = (session_id, sig)
            if session_id and key in existing:
                continue
            self.append(
                goal=goal,
                action=action,
                reward=float(reward),
                reason="teacher demo",
                source="teacher",
                session_id=session_id,
            )
            existing.add(key)
            written += 1
        return written

    def retrieve_plan(
        self,
        goal: str,
        *,
        banned_sigs: set[str] | None = None,
    ) -> list[dict[str, Any]] | None:
        """Best matching plan for this goal.

        Newest full teacher demos beat hollow auto-saved [tab]-only plans.
        Teacher steps are never stripped by soft visual mistakes.
        """
        goal_tok = _tokens(goal)
        banned = set(banned_sigs if banned_sigs is not None else self.banned_for_goal(goal))
        protected = self.newest_teacher_sigs(goal)
        banned -= protected
        best: tuple[float, list[dict[str, Any]], str] | None = None
        now = time.time()
        # Track newest teacher richness so we can reject junk refined/agent plans.
        newest_teacher_hard = 0
        newest_teacher_t = -1.0
        for row in self.load_all():
            if str(row.get("source") or "") != "teacher_plan":
                continue
            if not goals_match(goal, str(row.get("goal") or "")):
                continue
            actions = row.get("actions")
            if not isinstance(actions, list):
                continue
            t = float(row.get("t") or 0.0)
            hard = _hard_step_count([a for a in actions if isinstance(a, dict)])
            if t >= newest_teacher_t:
                newest_teacher_t = t
                newest_teacher_hard = hard

        for row in self.load_all():
            source = str(row.get("source") or "")
            if source not in _PLAN_SOURCES:
                continue
            if not goals_match(goal, str(row.get("goal") or "")):
                continue
            actions = row.get("actions")
            if not isinstance(actions, list) or not actions:
                continue
            cleaned = [dict(a) for a in actions if isinstance(a, dict)]
            hard = _hard_step_count(cleaned)
            # Reject auto plans that threw away the demo's clicks/drags.
            if source in {"refined_plan", "agent_plan"}:
                if hard < 2:
                    continue
                if newest_teacher_hard >= 4 and hard < max(3, newest_teacher_hard // 2):
                    continue
                if newest_teacher_hard >= 3 and not _plan_has_pointer(cleaned):
                    continue
            ov = _overlap(goal_tok, _tokens(str(row.get("goal") or "")))
            age = max(0.0, now - float(row.get("t") or 0.0))
            # Strong recency: a demo from the last hour dominates older junk.
            recency = 1.0 / (1.0 + age / 3600.0)
            boost = _PLAN_SOURCE_BOOST.get(source, 0.0)
            richness = min(0.35, hard * 0.04)
            score = ov * 0.4 + recency * 0.35 + boost + richness
            if source == "teacher_plan" and float(row.get("t") or 0.0) >= newest_teacher_t - 1.0:
                score += 0.25
            filtered: list[dict[str, Any]] = []
            for a in cleaned:
                kind = str(a.get("action") or "")
                if kind == "wait":
                    filtered.append(a)
                    continue
                sig = action_signature(a)
                if sig and sig in banned:
                    continue
                filtered.append(a)
            if not any(str(a.get("action") or "") not in {"wait", "stop"} for a in filtered):
                continue
            # Prefer plans that kept pointer actions when the teacher had them.
            if newest_teacher_hard >= 3 and _plan_has_pointer(cleaned) and not _plan_has_pointer(filtered):
                score -= 0.4
            if best is None or score > best[0]:
                best = (score, filtered, source)
        if best is None:
            return None
        return _with_stop(best[1])

    @_locked
    def remember_mistake(
        self,
        *,
        goal: str,
        action: dict[str, Any],
        reason: str = "",
    ) -> None:
        """Persist a failed action so future runs for this goal skip it."""
        sig = action_signature(action)
        if not sig or sig in {"wait", "stop"}:
            return
        # Bump count if we already have this mistake for a matching goal.
        for row in reversed(self.load_all()):
            if str(row.get("source") or "") != "mistake":
                continue
            if not goals_match(goal, str(row.get("goal") or "")):
                continue
            if str(row.get("sig") or "") != sig:
                continue
            prev = int(row.get("count") or 1)
            self.append(
                goal=goal,
                action=action,
                reward=-1.0,
                reason=reason or str(row.get("reason") or "repeated mistake"),
                source="mistake",
                count=prev + 1,
            )
            return
        self.append(
            goal=goal,
            action=action,
            reward=-1.0,
            reason=reason or "failed step",
            source="mistake",
            count=1,
        )

    def banned_for_goal(self, goal: str, *, min_count: int = 1) -> set[str]:
        """Signatures that failed for this goal (persistent across runs)."""
        protected = self.newest_teacher_sigs(goal)
        bans: dict[str, int] = {}
        for row in self.load_all():
            if str(row.get("source") or "") != "mistake":
                continue
            if not goals_match(goal, str(row.get("goal") or "")):
                continue
            action = row.get("action") if isinstance(row.get("action"), dict) else {}
            kind = str((action or {}).get("action") or "").lower()
            reason = str(row.get("reason") or "").lower()
            # Keys often look like "no change" when the cursor was over the N-panel;
            # never permanently ban teacher hotkeys for weak SSIM alone.
            if kind in {"key", "hotkey", "type"} and (
                "almost no change" in reason or "weak or ambiguous" in reason
            ):
                continue
            # First soft click miss: don't ban until it fails twice.
            if kind in {"click", "drag"} and (
                "almost no change" in reason or "weak or ambiguous" in reason
            ):
                if int(row.get("count") or 1) < 2:
                    continue
            sig = str(row.get("sig") or action_signature(action or {}))
            if not sig or sig in protected:
                continue
            bans[sig] = max(bans.get(sig, 0), int(row.get("count") or 1))
        return {sig for sig, n in bans.items() if n >= min_count}

    @_locked
    def save_refined_plan(
        self,
        *,
        goal: str,
        actions: list[dict[str, Any]],
        reason: str = "refined after mistake",
    ) -> None:
        """Save a plan that already incorporates recovery (preferred next time)."""
        useful = _clean_plan_actions(actions)
        if _hard_step_count(useful) < 2:
            return
        # Never overwrite a richer teacher demo with a hollow recovery stub.
        teacher_hard = 0
        for row in self.load_all():
            if str(row.get("source") or "") != "teacher_plan":
                continue
            if not goals_match(goal, str(row.get("goal") or "")):
                continue
            teacher_hard = max(
                teacher_hard,
                _hard_step_count([a for a in (row.get("actions") or []) if isinstance(a, dict)]),
            )
        if teacher_hard >= 4 and _hard_step_count(useful) < max(3, teacher_hard // 2):
            return
        if not useful:
            return
        self.append(
            goal=goal,
            reward=0.95,
            reason=reason,
            source="refined_plan",
            actions=useful,
        )

    @_locked
    def save_successful_run(
        self,
        *,
        goal: str,
        actions: list[dict[str, Any]],
    ) -> None:
        useful = _clean_plan_actions(actions)
        if _hard_step_count(useful) < 3:
            return
        if not useful:
            return
        self.append(
            goal=goal,
            reward=0.9,
            reason="successful guided run",
            source="agent_plan",
            actions=useful,
        )

    def teacher_alternatives(
        self,
        goal: str,
        *,
        banned_sigs: set[str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Teacher (or strong) actions for recovery — never invent new ones."""
        banned = set(banned_sigs or set()) | self.banned_for_goal(goal)
        hints = self.retrieve(
            goal,
            n_success=max(limit, 8),
            n_fail=0,
            success_threshold=0.5,
            banned_sigs=banned,
        )
        out: list[dict[str, Any]] = []
        for row in hints.get("successes") or []:
            action = row.get("action")
            if not isinstance(action, dict):
                continue
            if str(action.get("action") or "") in {"wait", "stop"}:
                continue
            out.append(dict(action))
            if len(out) >= limit:
                break
        return out

    def retrieve(
        self,
        goal: str,
        *,
        n_success: int = 3,
        n_fail: int = 3,
        success_threshold: float = 0.2,
        banned_sigs: set[str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return similar past successes and mistakes by goal overlap + recency."""
        rows = self.load_all()
        if not rows:
            return {"successes": [], "mistakes": []}

        goal_tok = _tokens(goal)
        now = time.time()
        banned = banned_sigs or set()
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            source = str(row.get("source") or "agent")
            if source in _PLAN_SOURCES:
                continue
            if source == "mistake":
                # Mistakes still retrieved as mistakes below via reward.
                pass
            ov = _overlap(goal_tok, _tokens(str(row.get("goal") or "")))
            if goal_tok and ov <= 0.0 and not goals_match(goal, str(row.get("goal") or "")):
                continue
            age = max(0.0, now - float(row.get("t") or 0.0))
            recency = 1.0 / (1.0 + age / 86400.0)
            boost = 0.25 if source in {"teacher", "success"} else 0.0
            if source == "mistake":
                boost = 0.1
            if source == "skill":
                continue
            score = ov * 0.75 + recency * 0.15 + boost
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)

        successes: list[dict[str, Any]] = []
        mistakes: list[dict[str, Any]] = []
        for _score, row in scored:
            reward = float(row.get("reward") or 0.0)
            source = str(row.get("source") or "")
            sig = str(row.get("sig") or action_signature(row.get("action") or {}))
            if (reward >= success_threshold or source in {"teacher", "success"}) and len(successes) < n_success:
                if source == "mistake":
                    continue
                if sig and sig in banned:
                    continue
                successes.append(row)
            elif (reward < 0.0 or source == "mistake") and len(mistakes) < n_fail:
                mistakes.append(row)
            if len(successes) >= n_success and len(mistakes) >= n_fail:
                break
        return {"successes": successes, "mistakes": mistakes}
