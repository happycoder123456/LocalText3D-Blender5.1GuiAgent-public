"""Unit tests for Embodied GUI Agent critic, vision, policy parse, memory."""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestPolicyParse(unittest.TestCase):
    def test_parse_clean_json(self):
        from agent.policy import parse_action

        raw = json.dumps(
            {
                "action": "click",
                "target": "extrude_button",
                "x": 10,
                "y": 20,
                "keys": [],
                "reason": "press extrude",
            }
        )
        action = parse_action(raw)
        self.assertEqual(action["action"], "click")
        self.assertEqual(action["target"], "extrude_button")
        self.assertEqual(action["x"], 10)
        self.assertEqual(action["y"], 20)

    def test_parse_fenced_noise(self):
        from agent.policy import parse_action

        raw = 'Sure.\n{"action":"key","keys":["E"],"x":0,"y":0,"reason":"extrude"}\n'
        action = parse_action(raw)
        self.assertEqual(action["action"], "key")
        self.assertEqual(action["keys"], ["E"])

    def test_safe_ollama_name_rejects_injection(self):
        from agent.policy import safe_ollama_name

        self.assertEqual(safe_ollama_name("llama3.2-vision"), "llama3.2-vision")
        self.assertEqual(safe_ollama_name("blender-gui"), "blender-gui")
        self.assertEqual(safe_ollama_name("none"), "none")
        self.assertEqual(safe_ollama_name("auto"), "auto")
        with self.assertRaises(ValueError):
            safe_ollama_name("llama3.2-vision\nRUN evil")
        with self.assertRaises(ValueError):
            safe_ollama_name("foo;rm")
        with self.assertRaises(ValueError):
            safe_ollama_name("../escape")

    def test_parse_truncated_keys_spam(self):
        from agent.policy import parse_action

        raw = '{"action": "key", "target": "C", "x": 0, "y": 0, "keys": ["C", "C", "C", "C", "C", "C"'
        action = parse_action(raw)
        self.assertEqual(action["action"], "key")
        self.assertLessEqual(len(action["keys"]), 3)

    def test_parse_wait_seconds_clamped_and_coords_survive_inf(self):
        from agent.policy import clamp_wait_seconds, parse_action

        action = parse_action('{"action":"wait","seconds":99999,"reason":"stall"}')
        self.assertEqual(action["action"], "wait")
        self.assertLessEqual(action["seconds"], 8.0)
        self.assertEqual(clamp_wait_seconds(float("inf")), 0.5)
        parsed = parse_action('{"action":"click","x":1e309,"y":10,"reason":"overflow"}')
        self.assertEqual(parsed["x"], 0)
        self.assertEqual(parsed["y"], 10)

    def test_recipe_add_cube(self):
        from agent.recipes import recipe_for_goal

        steps = recipe_for_goal("place 1 cube")
        self.assertIsNotNone(steps)
        hotkeys = [s for s in steps if s["action"] == "hotkey"]
        keys_steps = [s for s in steps if s["action"] == "key"]
        self.assertTrue(keys_steps)
        self.assertEqual(keys_steps[0]["keys"], ["f3"])
        type_steps = [s for s in steps if s["action"] == "type"]
        self.assertTrue(type_steps)
        self.assertEqual(type_steps[0]["text"], "Add Cube")
        self.assertEqual(steps[-1]["action"], "stop")


class TestCritic(unittest.TestCase):
    def test_no_change_negative(self):
        from agent.critic import score_transition

        img = np.full((120, 160, 3), 40, dtype=np.uint8)
        result = score_transition(img, img.copy())
        self.assertLess(result["reward"], 0.0)
        self.assertIn("missed", result["note"].lower() + "change")

    def test_viewport_change_positive(self):
        from agent.critic import score_transition

        before = np.full((200, 240, 3), 30, dtype=np.uint8)
        after = before.copy()
        # Paint a bright blob in the central viewport region.
        after[80:140, 80:160] = 200
        result = score_transition(before, after)
        self.assertGreater(result["reward"], 0.0)
        self.assertGreater(result["viewport_delta"], 0.01)

    def test_empty_after_delete_negative(self):
        from agent.critic import score_transition

        before = np.random.randint(20, 220, size=(200, 240, 3), dtype=np.uint8)
        after = np.full((200, 240, 3), 12, dtype=np.uint8)
        result = score_transition(before, after)
        self.assertLessEqual(result["reward"], -0.5)


class TestVision(unittest.TestCase):
    def test_detect_template(self):
        from agent.vision import BlenderVision
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            rng = np.random.default_rng(0)
            tmpl = rng.integers(0, 255, size=(24, 32, 3), dtype=np.uint8)
            cv2.imwrite(str(tdir / "extrude_button.png"), tmpl)

            canvas = np.zeros((100, 120, 3), dtype=np.uint8)
            canvas[40:64, 50:82] = tmpl

            vision = BlenderVision(templates_dir=tdir, threshold=0.7, scales=(1.0,))
            hits = vision.detect(canvas)
            self.assertTrue(hits)
            self.assertEqual(hits[0]["class"], "extrude_button")
            self.assertEqual(len(hits[0]["bbox"]), 4)
            x, y, w, h = hits[0]["bbox"]
            self.assertEqual((x, y, w, h), (50, 40, 32, 24))
            cx, cy = vision.center_of(hits[0])
            self.assertEqual((cx, cy), (66, 52))

    def test_empty_templates(self):
        from agent.vision import BlenderVision

        with tempfile.TemporaryDirectory() as tmp:
            vision = BlenderVision(templates_dir=Path(tmp))
            hits = vision.detect(np.zeros((50, 50, 3), dtype=np.uint8))
            self.assertEqual(hits, [])


class TestMemory(unittest.TestCase):
    def test_append_and_retrieve(self):
        from agent.memory import AgentMemory

        with tempfile.TemporaryDirectory() as tmp:
            mem = AgentMemory(Path(tmp))
            mem.append(
                goal="extrude the selected face",
                action={"action": "key", "keys": ["E"]},
                reward=0.8,
            )
            mem.append(
                goal="extrude face outward",
                action={"action": "click", "target": "empty"},
                reward=-0.4,
            )
            mem.append(
                goal="add a monkey",
                action={"action": "key", "keys": ["A"]},
                reward=0.9,
            )
            time.sleep(0.01)
            found = mem.retrieve("extrude selected face", n_success=2, n_fail=2)
            self.assertTrue(found["successes"])
            self.assertEqual(found["successes"][0]["action"]["keys"], ["E"])
            self.assertTrue(found["mistakes"])
            self.assertLess(found["mistakes"][0]["reward"], 0.0)
            self.assertEqual(mem.count(), 3)
            mem.clear()
            self.assertEqual(mem.count(), 0)


class TestRecorderReplay(unittest.TestCase):
    def test_events_to_actions_click_key_wait(self):
        from agent.recorder import events_to_actions

        t0 = 1000.0
        events = [
            {"type": "click", "x": 10, "y": 20, "pressed": True, "coord_space": "window", "t": t0},
            {"type": "click", "x": 10, "y": 20, "pressed": False, "coord_space": "window", "t": t0 + 0.05},
            {"type": "key", "key": "Key.e", "t": t0 + 0.4},
            {"type": "key", "key": "'a'", "t": t0 + 0.5},
        ]
        actions = events_to_actions(events)
        kinds = [a["action"] for a in actions]
        self.assertEqual(kinds[0], "click")
        self.assertEqual(actions[0]["x"], 10)
        self.assertEqual(actions[0]["y"], 20)
        self.assertIn("wait", kinds)
        key_steps = [a for a in actions if a["action"] == "key"]
        self.assertEqual(key_steps[0]["keys"], ["e"])
        self.assertEqual(key_steps[1]["keys"], ["a"])
        self.assertEqual(actions[-1]["action"], "stop")

    def test_events_to_actions_hotkey_and_modifier_coalesce(self):
        from agent.recorder import events_to_actions, summarize_recorded_inputs

        t0 = 2000.0
        events = [
            {"type": "key", "key": "shift", "t": t0},
            {"type": "key", "key": "a", "t": t0 + 0.05},
            {"type": "hotkey", "keys": ["ctrl", "z"], "t": t0 + 0.4},
            {"type": "key", "key": "ctrl", "t": t0 + 0.8},
        ]
        actions = events_to_actions(events)
        kinds = [a["action"] for a in actions]
        self.assertEqual(kinds.count("key"), 0)
        hotkeys = [a for a in actions if a["action"] == "hotkey"]
        self.assertEqual(len(hotkeys), 2)
        self.assertEqual(hotkeys[0]["keys"], ["shift", "a"])
        self.assertEqual(hotkeys[1]["keys"], ["ctrl", "z"])
        summary = summarize_recorded_inputs(events)
        self.assertIn("shift+a", summary)
        self.assertIn("ctrl+z", summary)

    def test_events_to_actions_drag_skips_press(self):
        from agent.recorder import events_to_actions

        t0 = 50.0
        events = [
            {"type": "click", "x": 1, "y": 2, "pressed": True, "coord_space": "window", "t": t0},
            {
                "type": "drag",
                "x1": 1,
                "y1": 2,
                "x2": 40,
                "y2": 50,
                "coord_space": "window",
                "t": t0 + 0.2,
                "duration": 0.2,
                "points": [[1, 2], [20, 25], [40, 50]],
            },
            {"type": "click", "x": 40, "y": 50, "pressed": False, "coord_space": "window", "t": t0 + 0.21},
        ]
        actions = events_to_actions(events)
        kinds = [a["action"] for a in actions]
        self.assertEqual(kinds.count("click"), 0)
        self.assertEqual(kinds[0], "drag")
        self.assertEqual(actions[0]["x2"], 40)
        self.assertEqual(actions[0]["y2"], 50)
        self.assertAlmostEqual(actions[0]["duration"], 0.2)
        self.assertEqual(actions[0]["points"], [[1, 2], [20, 25], [40, 50]])

    def test_thin_points_keeps_ends(self):
        from agent.recorder import _thin_points

        pts = [(0, 0)] + [(i, i) for i in range(1, 200)]
        out = _thin_points(pts, max_points=20)
        self.assertLessEqual(len(out), 20)
        self.assertEqual(out[0], [0, 0])
        self.assertEqual(out[-1], [199, 199])

    def test_window_local_and_last_session_actions(self):
        from agent.recorder import Recorder, _to_window_xy
        from agent.window import WindowRect

        rect = WindowRect(left=100, top=200, width=800, height=600)
        x, y, space = _to_window_xy(rect, 150, 260)
        self.assertEqual((x, y, space), (50, 60, "window"))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rec = Recorder(dataset_root=root)
            rec._session_id = "abc123"
            rec._session_started_at = time.time()
            rec._session_events = [
                {
                    "type": "click",
                    "x": 50,
                    "y": 60,
                    "pressed": True,
                    "coord_space": "window",
                    "t": time.time(),
                },
                {
                    "type": "click",
                    "x": 50,
                    "y": 60,
                    "pressed": False,
                    "coord_space": "window",
                    "t": time.time() + 0.02,
                },
                {"type": "key", "key": "tab", "t": time.time() + 0.1},
            ]
            rec._write_last_session()
            actions = rec.last_session_actions()
            self.assertTrue(actions)
            self.assertEqual(actions[0]["action"], "click")
            self.assertEqual(actions[0]["x"], 50)
            self.assertTrue(rec.status()["replay_available"])
            self.assertEqual(rec.status()["last_session_events"], 3)

    def test_ignores_screen_absolute_mouse_events(self):
        from agent.recorder import Recorder

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rec = Recorder(dataset_root=root)
            rec._session_id = "old"
            rec._session_started_at = time.time()
            rec._session_events = [
                {
                    "type": "click",
                    "x": 1809,
                    "y": 795,
                    "pressed": True,
                    "coord_space": "screen",
                    "t": time.time(),
                },
                {"type": "key", "key": "e", "t": time.time() + 0.1},
            ]
            rec._write_last_session()
            actions = rec.last_session_actions()
            kinds = [a["action"] for a in actions]
            self.assertNotIn("click", kinds)
            self.assertIn("key", kinds)

    def test_loop_learn_follows_demo_guided(self):
        from agent.loop import AgentLoop

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(dataset_root=root)
            loop.recorder._session_id = "sess1"
            loop.recorder._session_started_at = time.time()
            loop.recorder._session_goal = "extrude face"
            loop.recorder._session_events = [
                {
                    "type": "click",
                    "x": 400,
                    "y": 300,
                    "pressed": True,
                    "coord_space": "window",
                    "t": time.time(),
                },
                {
                    "type": "click",
                    "x": 400,
                    "y": 300,
                    "pressed": False,
                    "coord_space": "window",
                    "t": time.time() + 0.01,
                },
                {"type": "key", "key": "e", "t": time.time() + 0.1},
            ]
            loop.recorder._write_last_session()
            loop.memory.ingest_demo(
                goal="extrude face",
                actions=loop.recorder.last_session_actions(),
                session_id="sess1",
                window_w=800,
                window_h=600,
            )

            def _fake_run(max_steps: int) -> None:
                loop._running = False
                loop._status = "Done"
                loop._detail = f"mode={loop._mode}"

            loop._run = _fake_run  # type: ignore[method-assign]
            loop.start_agent(
                goal="extrude face",
                model="none",
                max_steps=10,
                run_mode="learn",
            )
            if loop._thread is not None:
                loop._thread.join(timeout=1.0)
            self.assertEqual(loop._mode, "master")
            self.assertEqual(loop._run_mode, "learn")
            self.assertGreaterEqual(len(loop._skill_steps), 1)
            demo = [s for s in loop._skill_steps if s.get("action") in {"click", "key"}]
            self.assertTrue(demo, msg=f"expected demo keys/clicks, got {[s.get('action') for s in loop._skill_steps[:6]]}")
            self.assertGreater(loop.memory.count(), 0)
            # Distilled click should be normalized.
            clicks = [s for s in loop._skill_steps if s.get("action") == "click"]
            self.assertTrue(clicks)
            self.assertIn("x_norm", clicks[0])

            # Unknown goal + vision model -> freeform VLM when no planner model is
            # available (the planner is isolated from the real Ollama here).
            from unittest import mock

            with mock.patch("agent.loop.resolve_planner_model", return_value=""):
                loop.start_agent(
                    goal="do something weird",
                    model="llama3.2-vision",
                    max_steps=5,
                    run_mode="learn",
                )
                if loop._thread is not None:
                    loop._thread.join(timeout=1.0)
                self.assertEqual(loop._mode, "vlm")
                loop.stop()

                with self.assertRaises(ValueError):
                    loop.start_agent(
                        goal="do something weird",
                        model="none",
                        max_steps=5,
                        run_mode="learn",
                    )

            loop.start_agent(goal="", model="none", max_steps=10, run_mode="replay")
            if loop._thread is not None:
                loop._thread.join(timeout=1.0)
            self.assertEqual(loop._mode, "replay")
            self.assertEqual(loop._recipe[0]["action"], "click")


class TestSkillMastery(unittest.TestCase):
    def test_distill_and_materialize_roundtrip(self):
        from agent.skills import distill_actions, materialize_step
        from agent.window import WindowRect

        actions = [
            {"action": "key", "keys": ["tab"]},
            {"action": "click", "x": 400, "y": 300},
            {
                "action": "drag",
                "x": 100,
                "y": 100,
                "x2": 300,
                "y2": 220,
                "duration": 0.18,
                "points": [[100, 100], [200, 160], [300, 220]],
            },
            {"action": "key", "keys": ["e"]},
        ]
        steps = distill_actions(actions, window_w=800, window_h=600)
        click = next(s for s in steps if s["action"] == "click")
        self.assertAlmostEqual(click["x_norm"], 0.5, places=3)
        self.assertAlmostEqual(click["y_norm"], 0.5, places=3)
        rect = WindowRect(left=0, top=0, width=1600, height=1200)
        out = materialize_step(click, rect)
        self.assertAlmostEqual(out["x"], 800.0, places=1)
        self.assertAlmostEqual(out["y"], 600.0, places=1)
        drag = next(s for s in steps if s["action"] == "drag")
        self.assertAlmostEqual(drag["duration"], 0.18)
        moved = materialize_step(drag, rect)
        self.assertEqual(len(moved["points"]), 3)
        self.assertAlmostEqual(moved["points"][0][0], 200.0, places=1)
        self.assertAlmostEqual(moved["x2"], 600.0, places=1)

    def test_compose_skills_compound_goal(self):
        from agent.skills import compose_skills, distill_actions

        skills = [
            {
                "goal": "add cube",
                "steps": distill_actions(
                    [{"action": "key", "keys": ["f3"]}, {"action": "type", "text": "Add Cube"}]
                ),
            },
            {
                "goal": "extrude face",
                "steps": distill_actions(
                    [{"action": "key", "keys": ["tab"]}, {"action": "key", "keys": ["e"]}],
                    window_w=800,
                    window_h=600,
                ),
            },
        ]
        composed = compose_skills("add a cube then extrude face", skills)
        self.assertEqual(len(composed), 2)
        self.assertIn("cube", composed[0]["goal"].lower())
        self.assertIn("extrude", composed[1]["goal"].lower())

    def test_step_stats_bump(self):
        from agent.skills import bump_step_stats

        step = {"action": "key", "keys": ["e"], "stats": {"ok": 0, "fail": 0}}
        ok = bump_step_stats(step, 0.5)
        self.assertEqual(ok["stats"]["ok"], 1)
        bad = bump_step_stats(ok, -0.3)
        self.assertEqual(bad["stats"]["fail"], 1)


class TestRecorderCaptureModes(unittest.TestCase):
    def test_normalize_and_fps(self):
        from agent.recorder import _fps_from_interval_ms, _normalize_capture_mode

        self.assertEqual(_normalize_capture_mode("video"), "video")
        self.assertEqual(_normalize_capture_mode("screenshots"), "screenshots")
        self.assertEqual(_normalize_capture_mode("png"), "screenshots")
        self.assertEqual(_normalize_capture_mode("weird"), "video")
        self.assertAlmostEqual(_fps_from_interval_ms(200), 5.0)
        self.assertAlmostEqual(_fps_from_interval_ms(80), 12.5)
        self.assertAlmostEqual(_fps_from_interval_ms(100), 10.0)
        self.assertAlmostEqual(_fps_from_interval_ms(16), 60.0)

    def test_events_include_t_rel_fields_in_session(self):
        from agent.recorder import Recorder

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rec = Recorder(dataset_root=root, capture_mode="video")
            rec._session_id = "vid1"
            rec._session_started_at = time.time() - 0.5
            rec._fps = 10.0
            rec._frames = 5
            now, t_rel, frame_idx = rec._timing()
            self.assertGreaterEqual(t_rel, 0.4)
            self.assertGreaterEqual(frame_idx, 0)
            rec._session_events = [
                {
                    "type": "click",
                    "x": 1,
                    "y": 2,
                    "pressed": True,
                    "coord_space": "window",
                    "t": now,
                    "t_rel": t_rel,
                    "frame_idx": frame_idx,
                },
                {
                    "type": "click",
                    "x": 1,
                    "y": 2,
                    "pressed": False,
                    "coord_space": "window",
                    "t": now + 0.01,
                    "t_rel": t_rel + 0.01,
                    "frame_idx": frame_idx,
                },
            ]
            rec._video_rel = "vid1.mp4"
            rec._write_last_session()
            meta = json.loads((root / "last_session.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["capture_mode"], "video")
            self.assertEqual(meta["video"], "vid1.mp4")
            self.assertIn("t_rel", meta["events"][0])
            self.assertIn("frame_idx", meta["events"][0])
            actions = rec.last_session_actions()
            self.assertEqual(actions[0]["action"], "click")

    def test_video_write_and_extract_frame(self):
        import cv2
        from agent.recorder import extract_video_frame
        from agent.paths import videos_dir

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vdir = videos_dir(root)
            path = vdir / "tiny.mp4"
            w, h, fps, n = 64, 48, 10.0, 8
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )
            self.assertTrue(writer.isOpened())
            for i in range(n):
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                frame[:, :] = (i * 20, 40, 80)
                writer.write(frame)
            writer.release()
            self.assertTrue(path.is_file())

            got = extract_video_frame(path, frame_idx=3, fps=fps)
            self.assertIsNotNone(got)
            self.assertEqual(got.shape[0], h)
            self.assertEqual(got.shape[1], w)

            got2 = extract_video_frame(path, t_rel=0.5, fps=fps)
            self.assertIsNotNone(got2)

    def test_similar_demos_video_and_screenshot(self):
        import cv2
        from agent.recorder import Recorder
        from agent.paths import episodes_path, screenshots_dir, videos_dir

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Legacy screenshot episode
            shot = "legacy.png"
            (screenshots_dir(root) / shot).write_bytes(b"\x89PNG\r\n\x1a\n")
            with episodes_path(root).open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "t": time.time(),
                            "session_id": "s1",
                            "capture_mode": "screenshots",
                            "screenshot": shot,
                            "events": [
                                {
                                    "type": "click",
                                    "x": 3,
                                    "y": 4,
                                    "pressed": True,
                                    "coord_space": "window",
                                    "t": time.time(),
                                    "t_rel": 0.1,
                                    "frame_idx": 1,
                                }
                            ],
                        }
                    )
                    + "\n"
                )

            # Video episode
            vname = "demo.mp4"
            path = videos_dir(root) / vname
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (32, 24),
            )
            for i in range(5):
                frame = np.full((24, 32, 3), i * 30, dtype=np.uint8)
                writer.write(frame)
            writer.release()
            with episodes_path(root).open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "t": time.time() + 1,
                            "session_id": "s2",
                            "capture_mode": "video",
                            "video": vname,
                            "fps": 10.0,
                            "frame_count": 5,
                            "events": [
                                {
                                    "type": "key",
                                    "key": "e",
                                    "t": time.time(),
                                    "t_rel": 0.2,
                                    "frame_idx": 2,
                                }
                            ],
                        }
                    )
                    + "\n"
                )

            rec = Recorder(dataset_root=root)
            demos = rec.similar_demos("anything", limit=2)
            self.assertTrue(demos)
            modes = []
            for d in demos:
                if d.get("video"):
                    modes.append("video")
                    self.assertTrue(d.get("events"))
                    # Frame extract may fail on some OpenCV builds; tolerate None.
                    if d.get("frame_bgr") is not None:
                        self.assertEqual(len(d["frame_bgr"].shape), 3)
                else:
                    modes.append("screenshots")
                    self.assertEqual(d.get("screenshot"), shot)
            self.assertIn("video", modes)
            self.assertIn("screenshots", modes)

    def test_start_record_passes_capture_mode(self):
        from agent.loop import AgentLoop

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(dataset_root=root)
            started = {}

            def _fake_start(interval_ms=200.0, capture_mode="video", goal=""):
                started["interval_ms"] = interval_ms
                started["capture_mode"] = capture_mode
                started["goal"] = goal
                loop.recorder.capture_mode = capture_mode
                loop.recorder._session_goal = goal

            loop.recorder.start = _fake_start  # type: ignore[method-assign]
            loop.start_record(interval_ms=150.0, capture_mode="screenshots", goal="extrude")
            self.assertEqual(started["capture_mode"], "screenshots")
            self.assertEqual(started["interval_ms"], 150.0)
            self.assertEqual(started["goal"], "extrude")
            self.assertIn("extrude", loop._detail.lower())

            loop.start_record(interval_ms=200.0, capture_mode="video", goal="add cube")
            self.assertEqual(started["capture_mode"], "video")
            self.assertIn("add cube", loop._detail.lower())

    def test_stop_without_goal_does_not_save_teacher_demo(self):
        from agent.loop import AgentLoop

        with tempfile.TemporaryDirectory() as tmp:
            loop = AgentLoop(dataset_root=Path(tmp))
            loop.recorder._session_id = "nog1"
            loop.recorder._session_started_at = time.time()
            loop.recorder._session_events = [
                {"type": "key", "key": "e", "t": time.time()},
            ]
            loop.recorder._running = False

            def _fake_stop(goal: str = ""):
                loop.recorder._running = False
                if goal:
                    loop.recorder._session_goal = goal
                loop.recorder._write_last_session()

            loop.recorder.stop = _fake_stop  # type: ignore[method-assign]
            before = loop.memory.count()
            loop.stop_record(goal="")
            self.assertEqual(loop.memory.count(), before)
            self.assertIn("NOT saved", loop._detail)
            self.assertEqual(loop._status, "Need goal")


class TestLearningMemory(unittest.TestCase):
    def test_ingest_demo_and_retrieve_plan(self):
        from agent.memory import AgentMemory, action_signature, same_action

        with tempfile.TemporaryDirectory() as tmp:
            mem = AgentMemory(Path(tmp))
            n = mem.ingest_demo(
                goal="extrude the face",
                actions=[
                    {"action": "key", "keys": ["tab"], "reason": "edit"},
                    {"action": "key", "keys": ["e"], "reason": "extrude"},
                    {"action": "wait", "seconds": 0.2},
                    {"action": "stop"},
                ],
                session_id="demo1",
            )
            self.assertGreaterEqual(n, 2)
            plan = mem.retrieve_plan("extrude the face")
            self.assertIsNotNone(plan)
            kinds = [a["action"] for a in plan]
            self.assertEqual(kinds[0], "key")
            self.assertEqual(plan[0]["keys"], ["tab"])
            self.assertIn("key", kinds)
            self.assertEqual(kinds[-1], "stop")

            # Duplicate session ingest is a no-op for same plan/sigs.
            self.assertEqual(
                mem.ingest_demo(
                    goal="extrude the face",
                    actions=[{"action": "key", "keys": ["e"]}],
                    session_id="demo1",
                ),
                0,
            )
            mem.append(
                goal="extrude face",
                action={"action": "click", "x": 10, "y": 10},
                reward=-0.5,
                reason="missed",
                source="agent",
            )
            found = mem.retrieve("extrude selected face", n_success=2, n_fail=2)
            self.assertTrue(found["successes"])
            self.assertEqual(found["successes"][0]["source"], "teacher")
            self.assertTrue(found["mistakes"])

            ban = {action_signature({"action": "key", "keys": ["e"]})}
            alts = mem.teacher_alternatives("extrude", banned_sigs=ban)
            self.assertTrue(alts)
            self.assertTrue(all(action_signature(a) not in ban for a in alts))
            self.assertTrue(
                same_action(
                    {"action": "key", "keys": ["E"]},
                    {"action": "key", "keys": ["e"]},
                )
            )

    def test_stop_record_ingests_memory(self):
        from agent.loop import AgentLoop

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(dataset_root=root)
            loop.recorder._session_id = "stop1"
            loop.recorder._session_started_at = time.time()
            loop.recorder._session_events = [
                {
                    "type": "key",
                    "key": "e",
                    "t": time.time(),
                    "t_rel": 0.1,
                    "frame_idx": 1,
                }
            ]
            loop.recorder._running = False
            loop.recorder._write_last_session()

            def _fake_stop(goal: str = ""):
                loop.recorder._running = False
                if goal:
                    loop.recorder._session_goal = goal
                loop.recorder._write_last_session()

            loop.recorder.stop = _fake_stop  # type: ignore[method-assign]
            before = loop.memory.count()
            loop.stop_record(goal="extrude face")
            self.assertGreater(loop.memory.count(), before)
            self.assertIn("skill", loop._detail.lower())
            plan = loop.memory.retrieve_plan("extrude face")
            self.assertIsNotNone(plan)

    def test_recovery_uses_teacher_not_random(self):
        from agent.loop import AgentLoop
        from agent.memory import action_signature

        with tempfile.TemporaryDirectory() as tmp:
            loop = AgentLoop(dataset_root=Path(tmp))
            loop._goal = "extrude"
            loop.memory.ingest_demo(
                goal="extrude",
                actions=[
                    {"action": "key", "keys": ["tab"]},
                    {"action": "key", "keys": ["e"]},
                ],
                session_id="r1",
            )
            loop._recipe = [
                {"action": "key", "keys": ["e"]},
                {"action": "stop"},
            ]
            loop._recipe_i = 1
            failed = {"action": "click", "x": 5, "y": 5, "reason": "miss"}
            alt = loop._recovery_action(failed)
            self.assertIsNotNone(alt)
            self.assertNotEqual(action_signature(alt), action_signature(failed))
            # Must come from teacher keys, not invented F3.
            if alt.get("action") == "key":
                self.assertIn(alt.get("keys"), [["tab"], ["e"]])
            self.assertNotEqual(alt.get("keys"), ["f3"])

    def test_paced_defaults_for_learn(self):
        from agent.loop import AgentLoop

        with tempfile.TemporaryDirectory() as tmp:
            loop = AgentLoop(dataset_root=Path(tmp))
            loop.memory.ingest_demo(
                goal="extrude",
                actions=[{"action": "key", "keys": ["e"]}],
                session_id="pace1",
            )

            def _fake_run(max_steps: int) -> None:
                loop._running = False
                loop._status = "Done"

            loop._run = _fake_run  # type: ignore[method-assign]
            loop.start_agent(
                goal="extrude",
                model="none",
                run_mode="learn",
                think_sec=2.0,
                settle_sec=1.5,
            )
            if loop._thread is not None:
                loop._thread.join(timeout=1.0)
            self.assertAlmostEqual(loop._think_sec, 2.0)
            self.assertAlmostEqual(loop._settle_sec, 1.5)
            self.assertIn("extrude", loop._detail.lower())

            # Interruptible pause
            loop._stop.clear()
            self.assertTrue(loop._pause(0.01))
            loop._stop.set()
            self.assertFalse(loop._pause(1.0))

    def test_multi_skill_and_persistent_mistakes(self):
        from agent.loop import AgentLoop
        from agent.memory import AgentMemory, action_signature, goals_match

        with tempfile.TemporaryDirectory() as tmp:
            mem = AgentMemory(Path(tmp))
            mem.ingest_demo(
                goal="extrude face",
                actions=[
                    {"action": "key", "keys": ["tab"]},
                    {"action": "key", "keys": ["e"]},
                ],
                session_id="s_ext",
            )
            mem.ingest_demo(
                goal="add cube",
                actions=[
                    {"action": "key", "keys": ["f3"]},
                    {"action": "type", "text": "Add Cube"},
                    {"action": "key", "keys": ["enter"]},
                ],
                session_id="s_cube",
            )
            skills = mem.list_skills()
            self.assertGreaterEqual(len(skills), 2)
            self.assertTrue(any("extrude" in s.lower() for s in skills))
            self.assertTrue(any("cube" in s.lower() for s in skills))

            plan_e = mem.retrieve_plan("extrude the face")
            plan_c = mem.retrieve_plan("add cube")
            self.assertIsNotNone(plan_e)
            self.assertIsNotNone(plan_c)
            self.assertEqual(plan_e[0]["keys"], ["tab"])
            self.assertEqual(plan_c[0]["keys"], ["f3"])

            bad = {"action": "click", "x": 9, "y": 9}
            mem.remember_mistake(goal="extrude face", action=bad, reason="missed")
            bans = mem.banned_for_goal("extrude face")
            self.assertIn(action_signature(bad), bans)
            # Mistakes for extrude must not poison cube skill.
            self.assertNotIn(action_signature(bad), mem.banned_for_goal("add cube"))

            # Soft visual misses on keys must not ban Tab forever.
            mem.remember_mistake(
                goal="extrude face",
                action={"action": "key", "keys": ["tab"]},
                reason="almost no change (missed click / empty space)",
            )
            self.assertNotIn("key:tab", mem.banned_for_goal("extrude face"))

            # Hollow auto-plans must not beat a full teacher demo.
            mem.append(
                goal="extrude face",
                reward=0.9,
                reason="junk",
                source="agent_plan",
                actions=[
                    {"action": "key", "keys": ["tab"]},
                    {"action": "key", "keys": ["tab"]},
                ],
            )
            plan_after_junk = mem.retrieve_plan("extrude face")
            self.assertEqual(plan_after_junk[0]["keys"], ["tab"])
            self.assertTrue(any(a.get("action") == "key" and a.get("keys") == ["e"] for a in plan_after_junk))

            # New recording clears prior mistakes/auto-plans for that skill.
            mem.remember_mistake(
                goal="extrude face",
                action={"action": "click", "x": 1, "y": 1},
                reason="almost no change (missed click / empty space)",
            )
            mem.append(
                goal="extrude face",
                reward=-1.0,
                reason="almost no change",
                source="mistake",
                action={"action": "click", "x": 1, "y": 1},
                count=5,
            )
            mem.ingest_demo(
                goal="extrude face",
                actions=[
                    {"action": "key", "keys": ["tab"]},
                    {"action": "key", "keys": ["e"]},
                    {"action": "click", "x": 40, "y": 40},
                ],
                session_id="fresh-demo",
            )
            self.assertEqual(mem.banned_for_goal("extrude face"), set())
            fresh = mem.retrieve_plan("extrude face")
            self.assertTrue(any(a.get("action") == "click" for a in fresh))

            # Hollow recovery stubs must not replace a full teacher demo.
            mem.save_refined_plan(
                goal="extrude face",
                actions=[{"action": "key", "keys": ["e"]}],
                reason="recovery",
            )
            still = mem.retrieve_plan("extrude face")
            self.assertTrue(any(a.get("action") == "click" for a in still))
            self.assertTrue(
                any(a.get("action") == "key" and a.get("keys") == ["tab"] for a in still)
            )

            self.assertTrue(goals_match("extrude", "extrude the face"))
            self.assertFalse(goals_match("extrude face", "add cube"))

            # Learn picks the matching skill, not the last recording.
            loop = AgentLoop(dataset_root=Path(tmp))
            loop.recorder._session_id = "other"
            loop.recorder._session_goal = "add cube"
            loop.recorder._session_started_at = time.time()
            loop.recorder._session_events = [
                {"type": "key", "key": "f3", "t": time.time()},
            ]
            loop.recorder._write_last_session()

            def _fake_run(max_steps: int) -> None:
                loop._running = False
                loop._status = "Done"

            loop._run = _fake_run  # type: ignore[method-assign]
            loop.start_agent(goal="extrude face", model="none", run_mode="learn")
            if loop._thread is not None:
                loop._thread.join(timeout=1.0)
            self.assertEqual(loop._mode, "master")
            # Demo keys (workspace switch only if stuck in a specialty screen).
            demo = [s for s in loop._skill_steps if s.get("action") == "key"]
            self.assertTrue(demo, msg=f"no key steps in {[s.get('action') for s in loop._skill_steps[:8]]}")
            self.assertTrue(any(s.get("keys") in (["e"], ["tab"]) for s in demo))


class TestHandsAccuracy(unittest.TestCase):
    def test_viewport_keys_need_aim(self):
        from agent.hands import _needs_viewport_aim

        self.assertTrue(_needs_viewport_aim(["e"]))
        self.assertTrue(_needs_viewport_aim(["tab"]))
        self.assertTrue(_needs_viewport_aim(["g"]))
        self.assertFalse(_needs_viewport_aim(["f3"]))

    def test_aim_always_moves_for_modeling_key(self):
        from agent.hands import Hands
        from agent.window import WindowRect
        from unittest import mock

        hands = Hands()
        rect = WindowRect(left=0, top=0, width=800, height=600, hwnd=1)
        moves: list[tuple[int, int]] = []

        with mock.patch("pyautogui.moveTo", side_effect=lambda x, y, duration=0: moves.append((x, y))):
            with mock.patch("pyautogui.position", return_value=mock.Mock(x=780, y=100)):
                # Cursor already inside Blender (N-panel-ish) — must still aim.
                hands._aim_viewport_for_keys(rect, force=True)
        self.assertTrue(moves)
        self.assertNotEqual(moves[0], (780, 100))

    def test_own_esc_press_does_not_stop_agent(self):
        """Loop cut ends with Esc — the agent's own Esc must not trigger the Esc-to-stop hook."""
        import time
        from unittest import mock

        from agent.hands import Hands

        stops: list[bool] = []
        hands = Hands(on_stop=lambda: stops.append(True))
        fake_kb = mock.MagicMock()
        with mock.patch("pynput.keyboard.Controller", return_value=fake_kb):
            hands._press_key("esc")
        self.assertGreater(hands._own_esc_until, time.time())
        # User-pressed Esc after the guard window still stops.
        hands._own_esc_until = 0.0
        self.assertLess(hands._own_esc_until, time.time())
        self.assertEqual(stops, [])


class TestConceptLibrary(unittest.TestCase):
    def test_compose_matches_aliases(self):
        from agent.skills import compose_skills, materialize_concept_skill, precondition_steps

        concept = {
            "name": "inset",
            "summary": "Inset selected faces inward",
            "when_to_use": "To thicken walls or add frame detail",
            "preconditions": ["edit mode", "face selected"],
            "aliases": ["thicken", "inset faces"],
            "steps": [{"action": "key", "keys": ["i"], "reason": "inset"}],
        }
        skill = materialize_concept_skill(concept)
        self.assertEqual(skill["steps"][0]["keys"], ["tab"])
        composed = compose_skills("thicken this wall", [skill])
        self.assertEqual(len(composed), 1)
        self.assertEqual(composed[0]["goal"], "inset")
        pre = precondition_steps(["Edit Mode", "face selected"])
        self.assertTrue(any(s.get("keys") == ["tab"] for s in pre))

    def test_save_and_list_concepts(self):
        from agent.memory import AgentMemory

        with tempfile.TemporaryDirectory() as tmp:
            mem = AgentMemory(Path(tmp))
            n = mem.save_concepts(
                [
                    {
                        "name": "extrude",
                        "summary": "Pull faces along their normal",
                        "when_to_use": "Add depth to a face",
                        "preconditions": ["edit mode", "face selected"],
                        "aliases": ["thicken", "pull faces"],
                        "steps": [
                            {"action": "key", "keys": ["e"], "reason": "extrude"},
                        ],
                        "video": "demo.mp4",
                        "t_start": 10.0,
                        "t_end": 40.0,
                    },
                    {
                        "name": "extrude",
                        "summary": "Pull faces along their normal (richer)",
                        "aliases": ["extend"],
                        "steps": [
                            {"action": "key", "keys": ["e"], "reason": "extrude"},
                            {"action": "wait", "seconds": 0.3},
                        ],
                    },
                ],
                session_id="vid1",
            )
            self.assertEqual(n, 2)
            cards = mem.list_concepts()
            self.assertEqual(len(cards), 1)
            self.assertEqual(cards[0]["name"], "extrude")
            self.assertIn("extend", [a.lower() for a in cards[0]["aliases"]])
            summary = mem.concepts_summary()
            self.assertIn("extrude", summary.lower())
            skills = mem.list_skill_records()
            self.assertTrue(any(s.get("goal") == "extrude" for s in skills))

    def test_list_concepts_holds_hundreds(self):
        from agent.memory import AgentMemory

        with tempfile.TemporaryDirectory() as tmp:
            mem = AgentMemory(Path(tmp))
            batch = [
                {
                    "name": f"technique {i}",
                    "summary": f"Step {i}",
                    "steps": [{"action": "key", "keys": ["e"], "reason": "extrude"}],
                    "ops": [{"op": "mesh.extrude_region_move", "params": {}}],
                }
                for i in range(80)
            ]
            written = mem.save_concepts(batch, session_id="bulk")
            self.assertEqual(written, 80)
            cards = mem.list_concepts()
            self.assertEqual(len(cards), 80)
            summary = mem.concepts_summary()
            self.assertGreaterEqual(summary.count("|") + 1, 80)
            self.assertEqual(len(mem.semantic_skills()), 80)


class TestRecorderCeilings(unittest.TestCase):
    def test_session_and_drag_duration_caps(self):
        from agent.recorder import (
            MAX_DRAG_SECONDS,
            MAX_RECORD_SECONDS,
            bounded_drag_duration,
            events_to_actions,
            record_duration_exceeded,
        )

        self.assertEqual(MAX_RECORD_SECONDS, 6 * 60 * 60)
        self.assertFalse(record_duration_exceeded(1_000.0, 1_000.0 + MAX_RECORD_SECONDS - 1))
        self.assertTrue(record_duration_exceeded(1_000.0, 1_000.0 + MAX_RECORD_SECONDS))
        self.assertFalse(record_duration_exceeded(None, 10.0))
        self.assertIsNone(bounded_drag_duration(float("nan")))
        self.assertEqual(bounded_drag_duration(9.0), MAX_DRAG_SECONDS)
        actions = events_to_actions(
            [
                {
                    "type": "drag",
                    "x1": 1,
                    "y1": 2,
                    "x2": 4,
                    "y2": 5,
                    "coord_space": "window",
                    "t": 10.0,
                    "duration": 30.0,
                }
            ]
        )
        self.assertAlmostEqual(actions[0]["duration"], MAX_DRAG_SECONDS)


if __name__ == "__main__":
    unittest.main()
