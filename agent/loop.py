"""Main agent loop: adaptive skill mastery + replay + recipes."""

from __future__ import annotations

import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from agent.abstract import abstract_recording, make_skill_card, ops_to_steps
from agent.context import ContextTracker
from agent.critic import score_transition
from agent.hands import Hands
from agent.judge import judge_op
from agent.knowledge import expand_op, has_variant
from agent.memory import AgentMemory, action_signature, goals_match
from agent.paths import ensure_dataset, prune_old_files, screenshots_dir
from agent.planner import Plan, enrich_plan_for_start, make_plan
from agent.policy import VisionPolicy, resolve_planner_model
from agent.recipes import recipe_for_goal
from agent.recorder import Recorder, summarize_recorded_inputs
from agent.skills import (
    allowed_kinds_for_skill,
    bump_step_stats,
    compose_skills,
    distill_actions,
    materialize_step,
    skill_step_signature,
    skill_success_rate,
    snap_click_to_detections,
    successful_skill_alts,
)
from agent.vision import BlenderVision
from agent.window import capture_window, encode_png_bytes, find_blender_window


def _normalize_run_mode(raw: Any) -> str:
    mode = str(raw or "learn").strip().lower()
    if mode in {"replay", "copy", "imitate"}:
        return "replay"
    return "learn"


def _with_stop(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [dict(a) for a in actions if isinstance(a, dict)]
    if out and str(out[-1].get("action") or "") != "stop":
        out.append({"action": "stop", "reason": "end of demo plan"})
    return out


class AgentLoop:
    def __init__(
        self,
        dataset_root: Path | None = None,
        templates_dir: Path | None = None,
    ):
        self.root = ensure_dataset(dataset_root)
        self.memory = AgentMemory(self.root)
        self.vision = BlenderVision(templates_dir=templates_dir)
        self.recorder = Recorder(dataset_root=self.root)
        self.hands = Hands(on_stop=self._stop_from_esc)
        self.policy = VisionPolicy()
        self._blender_apply_lock = threading.Lock()
        self._blender_apply_q: list[dict[str, Any]] = []
        self.hands.on_blender_apply = self.queue_blender_apply
        self._thread: threading.Thread | None = None
        self._video_thread: threading.Thread | None = None
        # start/stop are called from HTTP threads while the loop thread runs;
        # the lock keeps a second Play from racing a still-finishing run.
        self._lifecycle = threading.Lock()
        self._stop = threading.Event()
        # Hands.execute("wait") polls this so Stop/Esc interrupt long recorded pauses.
        self.hands.stop_event = self._stop
        self._running = False
        self._plan_start_ctx: dict[str, Any] | None = None
        self._sem_exec_t = 0.0
        self._abstracted_session_id = ""
        self._status = "Idle"
        self._detail = ""
        self._last_reward = 0.0
        self._last_action: dict[str, Any] = {}
        self._step = 0
        self._goal = ""
        self._error = ""
        self._recipe: list[dict[str, Any]] = []
        self._recipe_i = 0
        self._mode = "idle"  # idle | replay | recipe | master | vlm
        self._run_mode = "learn"
        self._banned_sigs: set[str] = set()
        self._recent_failures: list[dict[str, Any]] = []
        self._guide_retries = 0
        self._think_sec = 0.2
        self._settle_sec = 0.12
        self._executed_plan: list[dict[str, Any]] = []
        self._plan_dirty = False
        self._positive_hard = 0
        self._attempted_hard = 0
        self._skill_queue: list[dict[str, Any]] = []
        self._skill_i = 0
        self._skill_steps: list[dict[str, Any]] = []
        self._skill_step_i = 0
        self._current_skill_goal = ""
        self._skill_steps_live: list[dict[str, Any]] = []
        self._learning_video = False
        self._concepts_summary = ""
        self._learn_phase = ""
        self._learn_percent = 0.0
        self._learn_started_t = 0.0
        self._learn_beat_t = 0.0
        self._learn_pulse = 0
        self._learn_history: list[str] = []
        # Semantic planning / Blender context
        self.context = ContextTracker()
        self._plan: Plan | None = None
        self._planner_model = ""
        self._plan_preview: dict[str, Any] = {}
        self._sem_cur = -1
        self._sem_t0 = 0.0
        self._sem_before_ctx: dict[str, Any] | None = None
        self._sem_before_frame: Any = None
        self._sem_results: list[dict[str, Any]] = []
        self._sem_retried: set[int] = set()
        self._learned_note = ""
        self._deferred: tuple[Any, ...] = ()

    def status(self) -> dict[str, Any]:
        rec = self.recorder.status()
        skills = self.memory.list_skills(limit=240)
        try:
            cards = self.memory.semantic_skills(limit=240)
        except Exception:
            cards = []
        try:
            concept_cards = self.memory.list_concepts(limit=400)
            concept_n = len(concept_cards)
        except Exception:
            concept_cards = []
            concept_n = 0
        return {
            "context": self.context.summary(),
            "context_ok": self.context.available(max_age=3.0),
            "plan_preview": self._plan_preview,
            "planner_model": self._planner_model,
            "learned_note": self._learned_note,
            "skills_detail": [
                {
                    "name": c.get("goal"),
                    "summary": str(c.get("summary") or "")[:160],
                    "origin": c.get("origin"),
                    "params": c.get("params") or {},
                }
                for c in cards
            ],
            "running": self._running,
            "recording": self.recorder.running,
            "status": self._status,
            "detail": self._detail,
            "step": self._step,
            "goal": self._goal,
            "last_reward": self._last_reward,
            "last_action": self._last_action,
            "memory_count": self.memory.count(),
            "skills": skills,
            "skills_count": len(skills),
            "concepts_count": concept_n,
            "templates": self.vision.class_names,
            "error": (self._error or rec.get("error", ""))[:300],
            "dataset": str(self.root),
            "recorder": rec,
            "model": self.policy.model,
            "mode": self._mode,
            "run_mode": self._run_mode,
            "think_sec": self._think_sec,
            "settle_sec": self._settle_sec,
            "last_session_events": int(rec.get("last_session_events") or 0),
            "replay_available": bool(rec.get("replay_available")),
            "last_session_goal": str(rec.get("last_session_goal") or ""),
            "learning_video": self._learning_video,
            "concepts": [
                {"name": c.get("name"), "summary": str(c.get("summary") or "")[:80]}
                for c in concept_cards[:200]
            ],
            "concepts_summary": self._concepts_summary or self.memory.concepts_summary(limit=200),
            "learning_progress": self._learning_progress_payload(),
        }

    def queue_blender_apply(self, payload: dict[str, Any]) -> None:
        """Hands queued a bpy-side action (materials) for the addon timer to run."""
        if not isinstance(payload, dict) or not payload.get("op"):
            return
        with self._blender_apply_lock:
            self._blender_apply_q.append(dict(payload))

    def take_blender_apply(self) -> list[dict[str, Any]]:
        with self._blender_apply_lock:
            out = list(self._blender_apply_q)
            self._blender_apply_q.clear()
        return out

    def _learning_progress_payload(self) -> dict[str, Any]:
        now = time.time()
        started = float(self._learn_started_t or 0.0)
        beat = float(self._learn_beat_t or 0.0)
        elapsed = max(0.0, now - started) if started else 0.0
        since_beat = max(0.0, now - beat) if beat else elapsed
        alive = bool(self._learning_video) and since_beat < 45.0
        return {
            "active": bool(self._learning_video),
            "phase": self._learn_phase or ("learning" if self._learning_video else ""),
            "percent": float(max(0.0, min(100.0, self._learn_percent))),
            "elapsed_sec": float(elapsed),
            "since_update_sec": float(since_beat),
            "pulse": int(self._learn_pulse),
            "alive": alive,
            "message": self._detail if self._learning_video else "",
            "recent": list(self._learn_history[-5:]),
        }

    @staticmethod
    def _phase_from_progress(msg: str) -> tuple[str, float]:
        """Map teacher progress text → (phase label, rough percent 0–100)."""
        low = (msg or "").lower()
        m = re.search(r"watching\s+(\d+)\s*/\s*(\d+)", low)
        if m:
            cur, total = int(m.group(1)), max(1, int(m.group(2)))
            # Teaching concepts is most of the wall time (55–92%).
            return "teaching", 55.0 + 37.0 * (cur / total)
        if "teaching steps" in low or low.startswith("teaching "):
            return "teaching", 60.0
        if "scanning segment" in low:
            mseg = re.search(r"segment\s+(\d+)\s*/\s*(\d+)", low)
            if mseg:
                cur, total = int(mseg.group(1)), max(1, int(mseg.group(2)))
                return "concepts", 20.0 + 30.0 * (cur / total)
            return "concepts", 25.0
        if "downloading" in low or "retrying download" in low:
            return "download", 12.0
        if "youtube unavailable" in low or "learning from your last" in low:
            return "download", 18.0
        if "using local video" in low:
            return "download", 20.0
        if "sampling keyframe" in low:
            return "keyframes", 30.0
        if "finding concepts" in low or "learning from narration" in low:
            return "concepts", 45.0
        if "vision failed" in low:
            return "concepts", 42.0
        if "learned " in low and "concept" in low:
            return "saving", 95.0
        if "starting" in low:
            return "starting", 3.0
        return "working", max(5.0, 0.0)

    def _note_learn_progress(self, msg: str, *, percent: float | None = None, phase: str = "") -> None:
        text = str(msg or "").strip()[:240]
        if not text:
            return
        inferred_phase, inferred_pct = self._phase_from_progress(text)
        self._learn_phase = phase or inferred_phase
        if percent is not None:
            self._learn_percent = float(percent)
        else:
            # Never move the bar backwards on noisy retries.
            self._learn_percent = max(float(self._learn_percent), float(inferred_pct))
        self._detail = text
        self._status = "Learning"
        self._learn_beat_t = time.time()
        self._learn_pulse = int(self._learn_pulse) + 1
        if not self._learn_history or self._learn_history[-1] != text:
            self._learn_history.append(text)
            self._learn_history = self._learn_history[-12:]

    def busy(self) -> str:
        """Which long-running activity owns the machine right now ('' when idle)."""
        if self._running or (self._thread is not None and self._thread.is_alive() and self._status == "Running"):
            return "agent"
        if self.recorder.running:
            return "recording"
        if self._learning_video:
            return "video"
        return ""

    def start_record(
        self,
        interval_ms: float = 80.0,
        capture_mode: str = "video",
        goal: str = "",
    ) -> None:
        label = (goal or "").strip()
        other = self.busy()
        if other == "agent":
            raise RuntimeError("Stop the agent before recording — it would record its own keystrokes")
        if other == "video":
            raise RuntimeError("Wait for Learn from Video to finish before recording")
        if other == "recording":
            return  # already recording; Stop Recording ends it
        self.context.mark_session_start()
        self._learned_note = ""
        self.recorder.start(interval_ms=interval_ms, capture_mode=capture_mode, goal=label)
        mode = self.recorder.capture_mode
        if label:
            self._status = "Recording"
            self._detail = f"Recording skill '{label}' ({mode})."
        elif mode == "screenshots":
            self._detail = "Recording screenshots + input (set Goal before Stop!)."
        else:
            self._detail = "Recording video + input (set Goal before Stop!)."

    def stop_record(self, goal: str = "") -> None:
        session_id = str(getattr(self.recorder, "_session_id", "") or "")
        if not self.recorder.running and session_id and session_id == self._abstracted_session_id:
            # Second Stop click / server shutdown: this session was already saved;
            # re-running abstraction would overwrite its skill card.
            return
        if session_id:
            self._abstracted_session_id = session_id
        session_goal = str(getattr(self.recorder, "_session_goal", "") or "").strip()
        label = (goal or "").strip() or session_goal
        if not label or label.lower() in {"teacher demo", "demo"}:
            self.recorder.stop(goal="")
            rec = self.recorder.status()
            n = int(rec.get("last_session_events") or 0)
            self._status = "Need goal"
            self._detail = (
                f"Recording stopped ({n} events) but NOT saved — "
                "type a Goal (e.g. Extrude face) and Record again."
            )
            return
        self.recorder.stop(goal=label)
        rec = self.recorder.status()
        n = int(rec.get("last_session_events") or 0)
        learned = 0
        input_summary = ""
        blender_ops = len(self.context.session_ops())
        if n > 0 or blender_ops > 0:
            actions = self.recorder.last_session_actions() if n > 0 else []
            session_id = str(rec.get("session_id") or "")
            ww = wh = 0
            rect = find_blender_window()
            if rect is not None:
                ww, wh = int(rect.width), int(rect.height)
            if actions:
                learned = self.memory.ingest_demo(
                    goal=label,
                    actions=actions,
                    session_id=session_id,
                    window_w=ww,
                    window_h=wh,
                )
                input_summary = summarize_recorded_inputs(
                    self.recorder.last_session_events_raw()
                )
            # Understand the demo as operations + parameters (technique, not replay).
            # Works even when the input hook saw nothing: Blender's operator log is enough.
            try:
                card = abstract_recording(
                    label,
                    actions,
                    tracker=self.context,
                    input_events=self.recorder.last_session_events_raw(),
                    window_w=ww,
                    window_h=wh,
                )
                if card.get("ops"):
                    if not actions:
                        card["steps"] = ops_to_steps(card["ops"], start_mode=str(card.get("start_mode") or ""))
                    if self.memory.attach_semantics(label, card):
                        learned += 1
                    how = "from Blender's operator log" if card.get("grounded") else "from hotkeys (start the addon context for exact params)"
                    self._learned_note = f"{card.get('summary')} [{how}]"
                else:
                    self._learned_note = "No modeling operations recognised in that recording."
            except Exception as exc:  # never lose the raw demo because abstraction failed
                self._learned_note = f"Semantic analysis failed: {str(exc)[:120]}"
        if not self._running:
            self._status = "Idle"
            if n > 0 or blender_ops > 0:
                mode = str(rec.get("capture_mode") or "video")
                skills = self.memory.list_skills()
                detail = f"Learned '{label}' ({mode}, {learned} rows, {len(skills)} skills)."
                if self._learned_note:
                    detail = f"{detail} Technique: {self._learned_note}"
                elif input_summary:
                    detail = f"{detail} Inputs: {input_summary}."
                self._detail = detail
            else:
                self._detail = (
                    "Recording stopped but nothing was captured: no mouse/keys and no Blender "
                    "operators. Do the steps in the 3D viewport while recording."
                )

    def start_agent(
        self,
        goal: str,
        model: str,
        max_steps: int = 160,
        templates_dir: str = "",
        run_mode: str = "learn",
        think_sec: float | None = None,
        settle_sec: float | None = None,
        planner_model: str = "",
    ) -> None:
        with self._lifecycle:
            if self._running:
                return
            prev = self._thread
            if prev is not None and prev.is_alive() and prev is not threading.current_thread():
                # Previous run is still unwinding (capture/hands); give it a moment.
                prev.join(timeout=3.0)
                if prev.is_alive():
                    raise RuntimeError("Agent is still finishing the previous run — try again in a second")
            other = self.busy()
            if other == "recording":
                raise RuntimeError("Stop Recording before running the agent")
            if other == "video":
                raise RuntimeError("Wait for Learn from Video to finish before running the agent")
            # Claim the slot before the (slow) planning below so a second Play is rejected.
            self._running = True
            self._stop.clear()
        try:
            self._start_agent_locked(
                goal,
                model,
                max_steps=max_steps,
                templates_dir=templates_dir,
                run_mode=run_mode,
                think_sec=think_sec,
                settle_sec=settle_sec,
                planner_model=planner_model,
            )
        except Exception:
            self._running = False
            raise

    def _start_agent_locked(
        self,
        goal: str,
        model: str,
        *,
        max_steps: int,
        templates_dir: str,
        run_mode: str,
        think_sec: float | None,
        settle_sec: float | None,
        planner_model: str,
    ) -> None:
        goal = (goal or "").strip()
        self._run_mode = _normalize_run_mode(run_mode)
        self._plan = None
        self._sem_cur = -1
        self._sem_results = []
        self._sem_retried = set()
        if think_sec is not None:
            self._think_sec = float(max(0.0, min(8.0, think_sec)))
        else:
            self._think_sec = 0.0 if self._run_mode == "replay" else 0.2
        if settle_sec is not None:
            self._settle_sec = float(max(0.0, min(8.0, settle_sec)))
        else:
            self._settle_sec = 0.05 if self._run_mode == "replay" else 0.12

        last_demo = self.recorder.last_session_actions()
        last_meta = self.recorder._read_last_session_meta() or {}
        last_goal = str(last_meta.get("goal") or "")
        recipe = recipe_for_goal(goal) if goal else None

        self._skill_queue = []
        self._skill_i = 0
        self._skill_steps = []
        self._skill_step_i = 0
        self._skill_steps_live = []
        self._current_skill_goal = ""

        if self._run_mode == "replay":
            if not last_demo:
                raise ValueError("No recording to replay. Record a demo first.")
            self._mode = "replay"
            self._recipe = _with_stop(last_demo)
            self._goal = goal or last_goal or "Replay last recording"
            self._banned_sigs = set()
        else:
            if not goal:
                raise ValueError("Enter a goal for Learn mode")
            if last_demo and last_goal and goals_match(goal, last_goal):
                session_id = str(last_meta.get("session_id") or "")
                ww = wh = 0
                rect = find_blender_window()
                if rect is not None:
                    ww, wh = int(rect.width), int(rect.height)
                self.memory.ingest_demo(
                    goal=last_goal,
                    actions=last_demo,
                    session_id=session_id,
                    window_w=ww,
                    window_h=wh,
                )

            self._banned_sigs = self.memory.banned_for_goal(goal)
            self._goal = goal
            # Instant: rules planner over learned skills + Blender knowledge.
            plan = self._build_plan(goal, model, planner_model, allow_llm=False)
            if plan:
                self._mode = "master"
                self._load_skill(0)
            elif self._planner_model:
                # Slow: local text model composes a plan — done on the agent thread.
                self._mode = "planning"
                self._deferred = (goal, model, planner_model, recipe)
            else:
                self._decide_legacy(goal, model, recipe)

        self.policy.model = model or "llava"
        if templates_dir:
            try:
                resolved = Path(templates_dir).expanduser().resolve()
            except OSError:
                resolved = None
            if resolved is not None and resolved.is_dir():
                self.vision.templates_dir = resolved
                self.vision.reload()
        self._recipe_i = 0
        self._recent_failures = []
        self._guide_retries = 0
        self._executed_plan = []
        self._plan_dirty = False
        self._positive_hard = 0
        self._attempted_hard = 0
        self._stop.clear()
        self._running = True
        self._step = 0
        self._error = ""
        self._status = "Running"
        self._plan_start_ctx = self.context.latest(max_age=15.0)
        self._set_start_detail()
        target = self._plan_then_run if self._mode == "planning" else self._run
        self._thread = threading.Thread(
            target=target,
            args=(max_steps,),
            name="agent-loop",
            daemon=True,
        )
        self._thread.start()

    def _set_start_detail(self) -> None:
        if self._mode == "planning":
            self._status = "Planning"
            self._detail = f"Asking {self._planner_model or 'planner'} how to '{self._goal}' with known skills…"
        elif self._mode == "replay":
            self._detail = f"Replaying last recording ({len(self._recipe)} steps)"
        elif self._mode == "recipe":
            self._detail = f"Goal: {self._goal} (known recipe, paced)"
        elif self._mode == "vlm":
            self._detail = f"Goal: {self._goal} (freeform vision — no skill yet)"
        elif self._plan is not None:
            used = ", ".join(self._plan.used_skills[:3])
            self._detail = (
                f"Plan ({self._plan.source}{' · skills: ' + used if used else ''}): "
                f"{self._plan.summary}"
            )
        else:
            n_skills = len(self._skill_queue)
            rate = skill_success_rate(self._skill_queue[0]) if self._skill_queue else 0.0
            self._detail = (
                f"Mastering '{self._current_skill_goal}' "
                f"(skill 1/{n_skills}, {len(self._skill_steps)} steps, "
                f"prior success {rate:.0%})"
            )

    def _decide_legacy(self, goal: str, model: str, recipe: list[dict[str, Any]] | None) -> None:
        """Pre-semantic fallbacks: composed raw skills, recipes, teacher plans, freeform vision."""
        known = self.memory.list_skill_records()
        composed = compose_skills(goal, known)
        if composed:
            self._mode = "master"
            self._skill_queue = composed
            self._load_skill(0)
            return
        if recipe:
            self._mode = "recipe"
            self._recipe = recipe
            return
        plan = self.memory.retrieve_plan(goal, banned_sigs=self._banned_sigs)
        if plan:
            steps = distill_actions([a for a in plan if str(a.get("action")) != "stop"])
            self._mode = "master"
            self._skill_queue = [{"goal": goal, "steps": steps, "source": "skill"}]
            self._load_skill(0)
            return
        # Freeform vision loop — no prior skill/demo required.
        if not (model or "").strip() or model == "none":
            skills = self.memory.list_skills()
            hint = ", ".join(skills[:5]) if skills else "(none yet)"
            raise ValueError(
                f"No skill for '{goal}' and no vision model selected. "
                f"Pick llama3.2-vision (or Record a demo). Known: {hint}"
            )
        self._mode = "vlm"
        self._recipe = []

    def _plan_then_run(self, max_steps: int) -> None:
        goal, model, planner_model, recipe = self._deferred
        try:
            plan = self._build_plan(goal, model, planner_model, allow_llm=True)
            if plan and not self._stop.is_set():
                self._mode = "master"
                self._load_skill(0)
            elif not self._stop.is_set():
                self._decide_legacy(goal, model, recipe)
        except ValueError as exc:
            self._error = str(exc)[:300]
            self._status = "Error"
            self._detail = self._error
            self._running = False
            return
        except Exception as exc:
            self._error = f"Planning failed: {str(exc)[:200]}"
            self._status = "Error"
            self._detail = self._error
            self._running = False
            return
        if self._stop.is_set():
            self._running = False
            self._status = "Stopped"
            return
        self._status = "Running"
        self._set_start_detail()
        self._run(max_steps)

    def _build_plan(self, goal: str, model: str, planner_model: str, *, allow_llm: bool = True) -> Plan | None:
        """Compose learned skills + Blender knowledge into a runnable plan (or None).

        allow_llm=False is the instant rules-only pass; a plan with unresolved
        phrases is rejected there when a planner model exists (the LLM pass will
        handle it on the agent thread).
        """
        ctx = self.context.latest(max_age=15.0)
        try:
            skills = self.memory.semantic_skills()
        except Exception:
            skills = []
        pm = ""
        try:
            pm = resolve_planner_model(planner_model, model if model and model != "none" else "")
        except Exception:
            pm = ""
        self._planner_model = pm
        try:
            plan = make_plan(goal, skills=skills, context=ctx, model=pm, allow_llm=bool(pm) and allow_llm)
        except Exception as exc:
            # Recorded as a note, not as the run's error: the legacy fallback may
            # still run fine, and a set _error blocks saving that success.
            self._learned_note = f"planner: {str(exc)[:160]}"
            return None
        if not plan.ops:
            return None
        # Safety net: never run a click-spam / search-stub as a "house/car" plan.
        from agent.planner import (
            compose_object_plan,
            ensure_workspace_ops,
            graft_learned_techniques,
            is_complex_object_goal,
            is_weak_object_plan,
        )

        if is_complex_object_goal(goal) and is_weak_object_plan(plan.ops):
            plan = compose_object_plan(goal, skills)
            plan = graft_learned_techniques(plan, goal, skills)
            plan = ensure_workspace_ops(plan, goal, ctx)
        # Drop bare viewport clicks — they don't transfer and burn the step budget.
        plan.ops = [
            o
            for o in plan.ops
            if str(o.get("op") or "") != "view3d.select"
            or (isinstance(o.get("pointer"), dict) and o.get("pointer"))
        ]
        if not plan.ops:
            return None
        if plan.unresolved and not allow_llm and pm:
            return None  # defer to the LLM pass
        if plan.unresolved:
            plan.why = (plan.why + f"; could not interpret: {'; '.join(plan.unresolved[:3])}").strip("; ")
        plan = enrich_plan_for_start(plan, ctx)
        start_mode = str((ctx or {}).get("mode") or "")
        steps = ops_to_steps(
            plan.ops,
            start_mode=start_mode,
            variant_for=self.memory.preferred_variant,
            default_pointer=self._default_pointer(),
        )
        if not steps:
            return None
        self._plan = plan
        self._plan_preview = plan.to_payload()
        self._skill_queue = [{"goal": goal, "steps": steps, "source": "plan", "ops": plan.ops}]
        return plan

    def _default_pointer(self) -> dict[str, Any]:
        center = self.context.viewport_center_norm()
        if center:
            return {"x_norm": center[0], "y_norm": center[1]}
        return {"x_norm": 0.45, "y_norm": 0.52}

    def preview_plan(self, goal: str, model: str = "", planner_model: str = "") -> dict[str, Any]:
        """Explain what the agent would do for goal without running it (rules now, LLM in background)."""
        goal = (goal or "").strip()
        if not goal:
            raise ValueError("Enter a goal to preview")
        ctx = self.context.latest(max_age=15.0)
        skills = self.memory.semantic_skills()
        rules = make_plan(goal, skills=skills, context=ctx, model="", allow_llm=False)
        self._plan_preview = {**rules.to_payload(), "pending": bool(rules.unresolved or not rules.ops)}
        pm = resolve_planner_model(planner_model, model if model and model != "none" else "")
        self._planner_model = pm
        if self._plan_preview["pending"] and pm:

            def _bg() -> None:
                try:
                    plan = make_plan(goal, skills=skills, context=ctx, model=pm, allow_llm=True)
                    self._plan_preview = {**plan.to_payload(), "pending": False}
                except Exception as exc:
                    self._plan_preview = {**rules.to_payload(), "pending": False, "why": f"LLM failed: {str(exc)[:100]}"}

            threading.Thread(target=_bg, name="plan-preview", daemon=True).start()
        return self._plan_preview

    def _load_skill(self, index: int) -> bool:
        if index < 0 or index >= len(self._skill_queue):
            return False
        skill = self._skill_queue[index]
        self._skill_i = index
        self._current_skill_goal = str(skill.get("goal") or self._goal)
        steps = [dict(s) for s in (skill.get("steps") or []) if isinstance(s, dict)]
        self._skill_steps = steps
        self._skill_steps_live = [dict(s) for s in steps]
        self._skill_step_i = 0
        return True

    def stop(self) -> None:
        self._stop.set()
        self.hands.disarm()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            # Let the loop thread leave capture/hands before the UI is told it's idle,
            # otherwise a quick second Play drives two agents at once.
            t.join(timeout=4.0)
        self._running = False
        if not self.recorder.running:
            self._status = "Stopped"
            self._detail = "Agent stopped."

    def _stop_from_esc(self) -> None:
        """Keyboard Esc / mouse-corner failsafe pressed by the user."""
        if not self._running:
            return
        step, op_i = self._step, self._sem_cur + 1
        self.stop()
        self._detail = f"Stopped by Esc at step {step} (op {op_i}). Press Play to run again."

    def clear_memory(self) -> None:
        self.memory.clear()
        self._concepts_summary = ""

    def learn_from_video(
        self,
        *,
        url: str = "",
        path: str = "",
        goal: str = "",
        model: str = "",
        max_minutes: float = 6.0,
    ) -> None:
        """Background: extract modeling concepts from YouTube/local video into memory."""
        if self.busy():
            raise RuntimeError("Stop recording/agent before learning from video")
        goal = (goal or "").strip()
        if not (url or "").strip() and not (path or "").strip():
            raise ValueError("Paste a YouTube URL or choose a local video file")
        if (path or "").strip():
            from agent.video_teacher import ensure_video_path_in_dataset

            # Reject arbitrary on-disk paths before the background thread starts
            # so the HTTP handler can return 400.
            ensure_video_path_in_dataset(path, self.root)

        self._learning_video = True
        self._error = ""
        self._status = "Learning"
        self._learn_started_t = time.time()
        self._learn_beat_t = self._learn_started_t
        self._learn_pulse = 0
        self._learn_history = []
        self._learn_percent = 1.0
        self._note_learn_progress("Starting video teacher…", percent=2.0, phase="starting")
        self.policy.model = model or self.policy.model or "llava"

        def _run() -> None:
            from agent.video_teacher import learn_concepts_from_video

            def on_progress(msg: str) -> None:
                self._note_learn_progress(msg)

            try:
                result = learn_concepts_from_video(
                    url=url,
                    path=path,
                    goal=goal,
                    model=self.policy.model,
                    max_minutes=max_minutes,
                    dataset_root=self.root,
                    on_progress=on_progress,
                )
                concepts = result.get("concepts") or []
                self._note_learn_progress("Saving concepts into memory…", percent=96.0, phase="saving")
                session_id = f"video_{uuid.uuid4().hex[:10]}"
                written = self.memory.save_concepts(
                    list(concepts) if isinstance(concepts, list) else [],
                    session_id=session_id,
                )
                names = [str(c.get("name") or "") for c in concepts if isinstance(c, dict)]
                names = [n for n in names if n][:12]
                self._concepts_summary = self.memory.concepts_summary(limit=200)
                self._learn_phase = "done"
                self._learn_percent = 100.0
                self._learn_beat_t = time.time()
                self._learn_pulse = int(self._learn_pulse) + 1
                self._status = "Idle"
                self._detail = (
                    f"Saved {written} concept(s) from video"
                    + (f": {', '.join(names)}" if names else "")
                    + ". Select your mesh, type a Goal, then Run Agent."
                )
                self._learn_history.append(self._detail[:240])
            except Exception as exc:
                self._error = str(exc)[:300]
                self._learn_phase = "error"
                self._status = "Error"
                self._detail = self._error
                self._learn_beat_t = time.time()
            finally:
                self._learning_video = False

        # Separate handle: stop()/start_agent() join self._thread (the agent loop).
        self._video_thread = threading.Thread(target=_run, name="video-teacher", daemon=True)
        self._video_thread.start()

    # ------------------------------------------------------------ semantic run
    def _begin_semantic_step(self, step: dict[str, Any], frame: Any) -> None:
        sem_i = step.get("sem_i")
        if sem_i is None or int(sem_i) == self._sem_cur:
            return
        self._sem_cur = int(sem_i)
        self._sem_t0 = time.time()
        self._sem_before_ctx = self.context.latest(max_age=6.0)
        self._sem_before_frame = frame
        op = self._current_op()
        if op:
            from agent.knowledge import describe_op

            self._detail = f"Op {self._sem_cur + 1}/{len(self._plan.ops) if self._plan else '?'}: {describe_op(op.get('op'), op.get('params'))}"

    def _current_op(self) -> dict[str, Any] | None:
        if self._plan is None or not (0 <= self._sem_cur < len(self._plan.ops)):
            return None
        return self._plan.ops[self._sem_cur]

    def _plan_len_hint(self) -> int:
        """Primitive count of the plan as originally expanded (before retries were inserted)."""
        if self._skill_queue and 0 <= self._skill_i < len(self._skill_queue):
            return len(self._skill_queue[self._skill_i].get("steps") or [])
        return len(self._skill_steps)

    def _finish_semantic_step(self, step: dict[str, Any], after_frame: Any, prefix: str) -> None:
        op_row = self._current_op() or {}
        op = str(op_row.get("op") or "")
        params = dict(op_row.get("params") or {})
        # Judge against a snapshot taken AFTER the confirming keystroke (plus one
        # addon context tick), not one posted mid-sequence — a stale snapshot
        # read as "no effect" and made the agent redo the op.
        t_exec = self._sem_exec_t or time.time()
        after_ctx = self.context.wait_for_fresh(t_exec + 0.2, timeout=1.4)
        observed = self.context.ops_between(self._sem_t0 - 0.05)
        reward, note = judge_op(op, params, self._sem_before_ctx, after_ctx, observed)
        visual = None
        if self._sem_before_frame is not None and after_frame is not None:
            try:
                visual = score_transition(self._sem_before_frame, after_frame)
            except Exception:
                visual = None
        if reward is None:
            if visual is not None:
                reward = float(visual["reward"])
                note = f"visual: {visual.get('note')}"
            else:
                reward = 0.0
                note = "no feedback"
        reward = float(max(-1.0, min(1.0, reward)))
        self._last_reward = reward
        self._attempted_hard += 1
        ok = reward >= 0
        if ok:
            self._positive_hard += 1
            self._executed_plan.append({"action": "op", **op_row})
        variant = int(step.get("variant") or 0)
        if has_variant(op):
            self.memory.bump_op_stat(op, variant, ok)
        self.memory.append(
            goal=self._goal,
            action={"action": "op", "op": op, "params": params, "variant": variant},
            reward=reward,
            reason=note,
            source="agent",
        )
        self._sem_results.append({"op": op, "params": params, "reward": reward, "note": note})
        from agent.knowledge import describe_op

        label = describe_op(op, params) if op else "step"
        self._detail = f"{prefix}{label} → {'ok' if ok else 'miss'} ({note})"
        if ok or self._sem_cur in self._sem_retried:
            return
        if has_variant(op):
            # Same operation, other route (hotkey ⇄ F3 search); learned per machine.
            self._sem_retried.add(self._sem_cur)
            alt = 1 - variant
            prims = expand_op(op, params, variant=alt, pointer=op_row.get("pointer") or self._default_pointer())
            if prims:
                for p in prims:
                    p.update({"sem_i": self._sem_cur, "sem_last": False, "variant": alt, "from_teacher": True})
                    p.setdefault("stats", {"ok": 0, "fail": 0, "last_reward": 0.0})
                prims[-1]["sem_last"] = True
                prims.append({"action": "wait", "seconds": 0.25, "reason": "let Blender apply", "sem_i": self._sem_cur, "sem_last": False})
                self._skill_steps[self._skill_step_i:self._skill_step_i] = prims
                self._skill_steps_live[self._skill_step_i:self._skill_step_i] = [dict(p) for p in prims]
                self._sem_t0 = time.time()
                self._sem_before_ctx = after_ctx
                self._sem_before_frame = after_frame
                self._plan_dirty = True
                self._detail = f"{label} missed via {'hotkey' if variant == 0 else 'search'}; retrying via {'search' if alt == 1 else 'hotkey'}"

    def _save_plan_outcome(self, success_ratio: float) -> None:
        plan = self._plan
        if plan is None or not self._goal:
            return
        if self._status != "Done" or self._error or self._attempted_hard == 0:
            return
        if self._positive_hard >= 1 and success_ratio >= 0.6:
            from agent.planner import is_complex_object_goal, is_weak_object_plan

            # Don't memorize shallow/search stubs as "make a car/house" — that
            # poisons the next run into replaying the same useless plan.
            if is_complex_object_goal(self._goal) and is_weak_object_plan(plan.ops):
                self._detail = (
                    f"Done: {self._positive_hard}/{self._attempted_hard} operations confirmed. "
                    f"Not saved as a skill — plan was too weak for '{self._goal}'. "
                    "Record a better demo or learn more techniques from video."
                )
                return
            # Mode the plan STARTED in (not the mode before its last op) — otherwise a
            # plan that ended in Edit mode is saved as an Edit-mode skill.
            ctx = self._plan_start_ctx or {}
            card = make_skill_card(
                self._goal,
                plan.ops,
                start_mode=str(ctx.get("mode") or ""),
                origin="planned",
                grounded=self.context.available(max_age=30.0),
            )
            card["steps"] = ops_to_steps(plan.ops, start_mode=str(ctx.get("mode") or ""))
            try:
                self.memory.attach_semantics(self._goal, card, protect_recorded=True)
            except Exception:
                pass
            self._detail = (
                f"Done: {self._positive_hard}/{self._attempted_hard} operations confirmed. "
                f"'{self._goal}' is now a reusable skill."
            )
        else:
            failed = [r for r in self._sem_results if r["reward"] < 0]
            why = "; ".join(f"{r['op'].split('.')[-1]}: {r['note']}" for r in failed[:3])
            self._detail = f"Finished with misses ({self._positive_hard}/{self._attempted_hard} ok). {why}"[:300]

    def _pause(self, seconds: float) -> bool:
        if seconds <= 0:
            return not self._stop.is_set()
        return not self._stop.wait(seconds)

    def _describe_upcoming(self, action: dict[str, Any]) -> str:
        kind = str(action.get("action") or "")
        if kind == "click":
            return f"click at ({int(action.get('x') or 0)}, {int(action.get('y') or 0)})"
        if kind == "drag":
            return "drag"
        if kind in {"key", "hotkey"}:
            return f"{kind} {action.get('keys') or []}"
        if kind == "type":
            return f'type "{action.get("text") or ""}"'
        if kind == "wait":
            return f"wait {float(action.get('seconds') or 0):.1f}s"
        return kind

    def _next_guide_action(self) -> dict[str, Any] | None:
        if self._recipe_i >= len(self._recipe):
            return None
        action = dict(self._recipe[self._recipe_i])
        self._recipe_i += 1
        return action

    def _next_master_step(self) -> dict[str, Any] | None:
        while self._skill_step_i >= len(self._skill_steps):
            if self._skill_steps_live and self._current_skill_goal and self._plan is None:
                self.memory.update_skill_steps(
                    self._current_skill_goal, self._skill_steps_live
                )
                # Consumed: the run's finally must not write the same skill again.
                self._skill_steps_live = []
            if not self._load_skill(self._skill_i + 1):
                return {"action": "stop", "reason": "All skills finished."}
            n = len(self._skill_queue)
            self._detail = (
                f"Skill {self._skill_i + 1}/{n}: '{self._current_skill_goal}'"
            )
        step = dict(self._skill_steps[self._skill_step_i])
        self._skill_step_i += 1
        return step

    def _adapt_step(self, step: dict[str, Any], rect: Any, frame: Any) -> dict[str, Any]:
        action = materialize_step(step, rect)
        kind = str(action.get("action") or "")
        detections: list[dict[str, Any]] = []
        try:
            detections = self.vision.detect(frame) if frame is not None else []
        except Exception:
            detections = []
        if kind in {"click", "drag"} and detections:
            action = snap_click_to_detections(action, detections)
        sig = action_signature(action) or skill_step_signature(step)
        if sig and sig in self._banned_sigs and self._plan is None:
            alts = successful_skill_alts(
                self._skill_steps,
                banned_sigs=self._banned_sigs,
                prefer_kind=kind,
            )
            if alts:
                action = materialize_step(alts[0], rect)
                if kind in {"click", "drag"} and detections:
                    action = snap_click_to_detections(action, detections)
                action["reason"] = "preferred successful skill alt"
        return action

    def _constrained_recovery(
        self,
        failed: dict[str, Any],
        rect: Any,
        frame: Any,
    ) -> dict[str, Any] | None:
        kind = str(failed.get("action") or "")
        sig = action_signature(failed)
        if sig:
            self._banned_sigs.add(sig)
        alts = successful_skill_alts(
            self._skill_steps,
            banned_sigs=self._banned_sigs,
            prefer_kind=kind,
        )
        if alts:
            action = materialize_step(alts[0], rect)
            action["reason"] = "recovery from successful skill memory"
            return action
        allowed = allowed_kinds_for_skill(self._skill_steps)
        for alt in self.memory.teacher_alternatives(
            self._current_skill_goal or self._goal,
            banned_sigs=self._banned_sigs,
            limit=6,
        ):
            alt_kind = str(alt.get("action") or "")
            if alt_kind not in allowed:
                continue
            if action_signature(alt) in self._banned_sigs:
                continue
            alt = dict(alt)
            alt["reason"] = "recovery from teacher memory"
            return alt
        try:
            hints = self.memory.retrieve(
                self._current_skill_goal or self._goal,
                n_success=4,
                n_fail=4,
                banned_sigs=self._banned_sigs,
            )
            dets = self.vision.detect(frame) if frame is not None else []
            decided = self.policy.decide(
                frame,
                goal=self._current_skill_goal or self._goal,
                detections=dets,
                memory_hints=hints,
                last_reward=self._last_reward,
                last_note="recover within this skill only",
                banned_sigs=self._banned_sigs,
                recent_failures=self._recent_failures[-3:],
                allowed_actions=allowed,
            )
            if str(decided.get("action") or "") in allowed:
                decided["reason"] = "constrained VLM recovery"
                return decided
        except Exception:
            pass
        nxt = self._next_master_step()
        if nxt is not None and str(nxt.get("action") or "") != "stop":
            nxt = self._adapt_step(nxt, rect, frame)
            nxt["reason"] = "skip failed skill step; continue"
            return nxt
        # Mid-plan: never abort the whole run when one step fails.
        if self._plan is not None:
            return {"action": "wait", "seconds": 0.2, "reason": "skip failed step; keep plan going"}
        return {"action": "stop", "reason": "skill step failed; no recovery left"}

    def _recovery_action(self, failed: dict[str, Any]) -> dict[str, Any] | None:
        sig = action_signature(failed)
        if sig:
            self._banned_sigs.add(sig)
        alts = self.memory.teacher_alternatives(
            self._goal,
            banned_sigs=self._banned_sigs,
            limit=8,
        )
        for alt in alts:
            if action_signature(alt) not in self._banned_sigs:
                alt = dict(alt)
                alt["reason"] = "recovery from teacher memory (failed step)"
                return alt
        nxt = self._next_guide_action()
        if nxt is not None:
            nxt = dict(nxt)
            nxt["reason"] = "skip failed demo step; continue plan"
            return nxt
        return {
            "action": "stop",
            "reason": "demo step failed and no teacher alternative left",
        }

    def _run(self, max_steps: int) -> None:
        self.hands.reset_run()
        self.hands.arm_safety()
        blind = self._mode in {"replay", "recipe"}
        master = self._mode == "master"
        vlm = self._mode == "vlm"
        if blind:
            max_steps = min(max_steps, max(len(self._recipe) * 2, len(self._recipe)))
        elif master:
            total_steps = sum(len(s.get("steps") or []) for s in self._skill_queue) or 8
            from agent.planner import recommended_max_steps

            # Always finish the full plan — N-panel Max steps is a floor, not a mid-run cut.
            max_steps = recommended_max_steps(self._goal, int(total_steps))
            max_steps = max(int(max_steps), int(total_steps) + max(32, int(total_steps) // 3))
        elif vlm:
            max_steps = max(int(max_steps), 120)
        budget_base = int(max_steps)
        inserted_steps = 0
        skipped_banned = 0
        try:
            while not self._stop.is_set():
                # Retries/recoveries insert primitives mid-run; they must not eat the
                # budget meant for the original plan.
                if master:
                    inserted_steps = max(0, len(self._skill_steps) - self._plan_len_hint())
                    max_steps = budget_base + min(inserted_steps, budget_base)
                if self._step >= max_steps:
                    self._status = "Done"
                    self._detail = (
                        f"Step budget ({max_steps}) exhausted at op {self._sem_cur + 1}"
                        f"{'/' + str(len(self._plan.ops)) if self._plan is not None else ''}. "
                        "Raise Max steps or simplify the goal."
                    )
                    break
                rect = find_blender_window()
                if rect is None:
                    raise RuntimeError("Blender window not found")
                if self.hands._ensure_focus(rect):
                    # Focusing may have restored/moved the window: re-measure before capture.
                    rect = find_blender_window() or rect

                frame_preview = None
                skill_step_ref = None
                if master:
                    try:
                        frame_preview, rect = capture_window(rect)
                    except Exception:
                        frame_preview = None
                    raw_step = self._next_master_step()
                    if raw_step is None:
                        self._status = "Done"
                        self._detail = "Finished skill plan."
                        break
                    if str(raw_step.get("action") or "") == "stop":
                        self._status = "Done"
                        self._detail = str(raw_step.get("reason") or "Done.")
                        break
                    if self._plan is not None:
                        self._begin_semantic_step(raw_step, frame_preview)
                    action = self._adapt_step(raw_step, rect, frame_preview)
                    skill_step_ref = raw_step
                elif blind:
                    action = self._next_guide_action()
                    if action is None:
                        self._status = "Done"
                        self._detail = "Finished demo plan."
                        break
                elif vlm:
                    try:
                        frame_preview, rect = capture_window(rect)
                    except Exception as exc:
                        raise RuntimeError(f"Capture failed: {exc}") from exc
                    dets = []
                    try:
                        dets = self.vision.detect(frame_preview)
                    except Exception:
                        dets = []
                    hints = self.memory.retrieve(
                        self._goal,
                        n_success=4,
                        n_fail=4,
                        banned_sigs=self._banned_sigs,
                    )
                    action = self.policy.decide(
                        frame_preview,
                        goal=self._goal,
                        detections=dets,
                        memory_hints=hints,
                        last_reward=self._last_reward,
                        last_note=str((self._recent_failures[-1] or {}).get("note") or "")
                        if self._recent_failures
                        else "",
                        banned_sigs=self._banned_sigs,
                        recent_failures=self._recent_failures[-4:],
                    )
                    from agent.policy import validate_and_repair_action
                    action = validate_and_repair_action(
                        action,
                        width=int(rect.width),
                        height=int(rect.height),
                        banned_sigs=self._banned_sigs,
                    )
                else:
                    self._status = "Done"
                    self._detail = "No plan to follow."
                    break

                kind0 = str(action.get("action") or "")
                sig0 = action_signature(action)
                if (
                    (master or vlm)
                    and self._plan is None
                    and kind0 not in {"wait", "stop"}
                    and sig0
                    and sig0 in self._banned_sigs
                ):
                    self._detail = (
                        f"Skipping remembered mistake: {self._describe_upcoming(action)}"
                    )
                    self._plan_dirty = True
                    skipped_banned += 1
                    if skipped_banned >= 6 and vlm:
                        # Vision keeps proposing banned actions — each is a slow Ollama
                        # call and this branch never consumed budget before.
                        self._status = "Done"
                        self._detail = "Vision kept repeating banned actions; stopped."
                        break
                    if vlm:
                        self._step += 1
                    if not self._pause(0.35):
                        break
                    continue

                self._last_action = action
                self._step += 1
                upcoming = self._describe_upcoming(action)
                n_skills = max(1, len(self._skill_queue))
                prefix = (f"Skill {self._skill_i + 1}/{n_skills} " if master else ("Vision " if vlm else ""))
                self._detail = f"{prefix}Step {self._step}: thinking — next {upcoming}"

                if action.get("action") == "stop":
                    self._status = "Done"
                    self._detail = action.get("reason") or "Plan finished."
                    break

                kind = str(action.get("action") or "")
                if kind != "wait":
                    think = 0.0 if self._mode == "replay" else self._think_sec
                    if think > 0:
                        if not self._pause(think):
                            break
                    if self._stop.is_set():
                        break
                    self._detail = f"{prefix}Step {self._step}: doing {upcoming}"

                before = None
                if master or not blind:
                    before, rect = capture_window(rect)

                self.hands.aim_norm = self.context.viewport_center_norm()
                try:
                    self.hands.execute(action, rect)
                except Exception as exc:
                    # Mid-object builds: one bad keystroke must not abort the whole plan.
                    if self._stop.is_set():
                        break
                    msg = str(exc).lower()
                    if "failsafe" in msg or "corner" in msg:
                        raise
                    if master and self._plan is not None:
                        self._detail = f"{prefix}Step {self._step}: error — continuing ({str(exc)[:100]})"
                        if not self._pause(0.25):
                            break
                        continue
                    raise
                self._sem_exec_t = time.time()

                settle = self._settle_sec
                if kind == "wait":
                    settle = 0.0
                elif kind in {"type", "key", "hotkey"}:
                    settle = max(settle, 0.22)
                elif kind in {"click", "drag"}:
                    settle = max(settle, 0.10)
                if not self._pause(settle):
                    break

                if blind:
                    continue

                if self._plan is not None and skill_step_ref is not None:
                    # Semantic run: judge whole operations with Blender's own state,
                    # never individual keystrokes.
                    if not skill_step_ref.get("sem_last"):
                        continue
                    # Let Blender apply the operator before judging (the addon
                    # posts context every ~0.2 s).
                    if not self._pause(0.2):
                        break
                    after, _ = capture_window(rect)
                    self._finish_semantic_step(skill_step_ref, after, prefix)
                    if not self._pause(0.1):
                        break
                    continue

                after, _ = capture_window(rect)
                critique = score_transition(before, after)
                reward = float(critique["reward"])
                last_note = str(critique.get("note") or "")
                self._last_reward = reward
                self._detail = (
                    f"{prefix}Step {self._step}: {upcoming} → "
                    f"{'ok' if reward >= 0 else 'miss'} ({last_note})"
                )
                if not self._pause(0.12):
                    break

                shot_id = (
                    f"agent_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
                )
                (screenshots_dir(self.root) / shot_id).write_bytes(
                    encode_png_bytes(before)
                )
                prune_old_files(screenshots_dir(self.root), "agent_*.png", keep=80)
                self.memory.append(
                    goal=self._current_skill_goal or self._goal,
                    action=action,
                    reward=reward,
                    screenshot_id=shot_id,
                    reason=last_note,
                    source="agent",
                )

                if master and skill_step_ref is not None:
                    idx = self._skill_step_i - 1
                    if 0 <= idx < len(self._skill_steps_live):
                        self._skill_steps_live[idx] = bump_step_stats(
                            self._skill_steps_live[idx], reward
                        )

                soft = kind in {"wait", "type", "key", "hotkey"}
                if kind not in {"wait", "stop"}:
                    self._attempted_hard += 1
                if reward < 0 and not soft:
                    sig = action_signature(action)
                    if sig:
                        self._banned_sigs.add(sig)
                    self.memory.remember_mistake(
                        goal=self._current_skill_goal or self._goal,
                        action=action,
                        reason=last_note,
                    )
                    self._plan_dirty = True
                    self._recent_failures.append(
                        {
                            "action": action,
                            "reward": reward,
                            "note": last_note,
                            "sig": sig,
                        }
                    )
                    if vlm:
                        # Next VLM step sees banned_sigs + recent_failures.
                        self._guide_retries = 0
                        self._detail = (
                            "Mistake saved; vision will try a different action…"
                        )
                        if not self._pause(self._think_sec):
                            break
                    elif self._guide_retries < 1:
                        self._guide_retries += 1
                        if master:
                            recovery = self._constrained_recovery(action, rect, after)
                        else:
                            recovery = self._recovery_action(action)
                        if recovery is not None:
                            if master:
                                self._skill_steps.insert(self._skill_step_i, recovery)
                                self._skill_steps_live.insert(
                                    self._skill_step_i, dict(recovery)
                                )
                            else:
                                self._recipe.insert(self._recipe_i, recovery)
                            self._detail = (
                                "Remembered mistake; trying recovery for "
                                f"'{self._current_skill_goal or self._goal}'…"
                            )
                            if not self._pause(self._think_sec):
                                break
                    else:
                        self._guide_retries = 0
                        self._detail = (
                            "Mistake saved for "
                            f"'{self._current_skill_goal or self._goal}'; continuing."
                        )
                        if not self._pause(0.6):
                            break
                else:
                    self._guide_retries = 0
                    if kind not in {"wait", "stop"} and reward >= 0:
                        self._executed_plan.append(dict(action))
                        self._positive_hard += 1
                        self.memory.remember_success(
                            goal=self._current_skill_goal or self._goal,
                            action=action,
                            reward=reward,
                            reason=last_note,
                        )
        except Exception as exc:
            if self._stop.is_set():
                # User stop / Esc / mouse-corner failsafe raced with the loop: that is
                # a normal stop, not an error.
                self._status = "Stopped"
                self._detail = self._detail if "Stopped" in (self._detail or "") else "Agent stopped."
            else:
                self._error = str(exc)[:300]
                self._status = "Error"
                self._detail = self._error
        finally:
            if master and self._skill_steps_live and self._current_skill_goal and self._plan is None:
                try:
                    self.memory.update_skill_steps(
                        self._current_skill_goal, self._skill_steps_live
                    )
                except Exception:
                    pass
            success_ratio = (
                float(self._positive_hard) / float(self._attempted_hard)
                if self._attempted_hard
                else 0.0
            )
            if self._plan is not None:
                self._save_plan_outcome(success_ratio)
            elif (
                master
                and self._status == "Done"
                and self._executed_plan
                and not self._error
                and self._positive_hard >= 3
                and success_ratio >= 0.6
            ):
                if self._plan_dirty:
                    self.memory.save_refined_plan(
                        goal=self._goal,
                        actions=self._executed_plan,
                        reason="mastered after recovery",
                    )
                else:
                    self.memory.save_successful_run(
                        goal=self._goal,
                        actions=self._executed_plan,
                    )
            self._running = False
            self.hands.disarm()
            if self._status == "Running":
                self._status = "Idle"
                self._detail = "Finished."
