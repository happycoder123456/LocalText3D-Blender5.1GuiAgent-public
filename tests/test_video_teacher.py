"""Unit tests for video concept teacher (no live YouTube)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestVideoTeacherPure(unittest.TestCase):
    def test_merge_and_sanitize(self):
        from agent.video_teacher import merge_concepts, sanitize_concept_steps

        merged = merge_concepts(
            [
                {
                    "name": "Extrude",
                    "summary": "short",
                    "aliases": ["thicken"],
                    "t_start": 5,
                    "t_end": 20,
                    "steps": [],
                },
                {
                    "name": "extrude",
                    "summary": "Pull faces out to add depth",
                    "aliases": ["pull faces"],
                    "t_start": 1,
                    "t_end": 40,
                    "steps": [{"action": "key", "keys": ["e"]}],
                },
                {"name": "subscribe", "summary": "junk", "steps": []},
            ]
        )
        self.assertEqual(len(merged), 1)
        self.assertIn("pull faces", merged[0]["aliases"])
        self.assertGreater(len(merged[0]["summary"]), 5)
        self.assertAlmostEqual(merged[0]["t_start"], 1.0)
        self.assertAlmostEqual(merged[0]["t_end"], 40.0)

        steps = sanitize_concept_steps(
            [
                {"action": "click", "x": 100, "y": 200, "reason": "bad pixel"},
                {"action": "key", "keys": ["f3"], "reason": "search"},
                {"action": "type", "text": "Extrude Region", "reason": "op"},
                {"action": "key", "keys": ["enter"], "reason": "run"},
                {"action": "drag", "x": 1, "y": 2, "x2": 9, "y2": 9},
            ]
        )
        kinds = [s["action"] for s in steps]
        self.assertNotIn("click", kinds)
        self.assertNotIn("drag", kinds)
        self.assertIn("key", kinds)
        self.assertIn("type", kinds)
        self.assertIn("wait", kinds)

    def test_captions_window(self):
        from agent.video_teacher import captions_window

        cues = [(0.0, "hello"), (5.0, "extrude the face"), (90.0, "later")]
        text = captions_window(cues, 4.0, 6.0)
        self.assertIn("extrude", text)
        self.assertNotIn("later", text)

    def test_learn_pipeline_mocked(self):
        from agent.video_teacher import learn_concepts_from_video

        fake_frame = np.zeros((120, 160, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "lesson.mp4"
            video.write_bytes(b"not-a-real-video")

            with mock.patch(
                "agent.video_teacher.fetch_video_source",
                return_value=(video, [(1.0, "press E to extrude the face")], "lesson"),
            ), mock.patch(
                "agent.video_teacher.sample_keyframes",
                return_value=[(1.0, fake_frame), (3.0, fake_frame)],
            ), mock.patch(
                "agent.video_teacher.extract_concept_candidates",
                return_value=[
                    {
                        "name": "extrude",
                        "summary": "Pull faces",
                        "when_to_use": "Add depth",
                        "preconditions": ["edit mode", "face selected"],
                        "aliases": ["thicken"],
                        "t_start": 0.0,
                        "t_end": 10.0,
                        "steps": [],
                    }
                ],
            ), mock.patch(
                "agent.video_teacher.fill_concept_steps",
                side_effect=lambda c, **kw: {
                    **c,
                    "steps": [{"action": "key", "keys": ["e"], "reason": "extrude"}],
                },
            ):
                result = learn_concepts_from_video(
                    path=str(video),
                    goal="extrude",
                    model="none",
                    max_minutes=2.0,
                    dataset_root=root,
                )
            self.assertTrue(result["ok"])
            self.assertEqual(result["concepts"][0]["name"], "extrude")
            self.assertEqual(result["concepts"][0]["steps"][0]["keys"], ["e"])

    def test_caption_seeds_extra_techniques(self):
        from agent.video_teacher import concepts_from_captions, _concept_budget, _segment_length_sec

        cues = [
            (10.0, "now we add a boolean modifier"),
            (40.0, "use the knife tool to cut"),
            (90.0, "bridge those edge loops"),
            (120.0, "and solidify the shell"),
        ]
        found = concepts_from_captions(cues, t0=0.0, t1=200.0)
        names = {c["name"] for c in found}
        self.assertIn("boolean", names)
        self.assertIn("knife", names)
        self.assertIn("bridge edge loops", names)
        self.assertIn("solidify", names)
        self.assertGreaterEqual(_concept_budget(60), 16)
        self.assertGreaterEqual(_concept_budget(240), 40)
        self.assertGreaterEqual(_concept_budget(300), 200)
        self.assertLessEqual(_concept_budget(300), 240)
        self.assertGreaterEqual(_concept_budget(1200), 500)
        self.assertLessEqual(_concept_budget(1200), 600)
        self.assertAlmostEqual(_segment_length_sec(6), 360.0)
        self.assertEqual(_segment_length_sec(120), 12.0 * 60.0)
        self.assertEqual(_segment_length_sec(1200), 20.0 * 60.0)

    def test_reject_empty_source(self):
        from agent.video_teacher import fetch_video_source

        with self.assertRaises(ValueError):
            fetch_video_source(url="", path="")

    def test_rejects_non_youtube_url_and_non_video_file(self):
        from agent.video_teacher import fetch_video_source, _is_youtube_url

        self.assertTrue(_is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
        self.assertTrue(_is_youtube_url("https://youtu.be/dQw4w9WgXcQ"))
        self.assertFalse(_is_youtube_url("http://127.0.0.1:8765/health"))
        self.assertFalse(_is_youtube_url("file:///etc/passwd"))
        self.assertFalse(_is_youtube_url("https://example.com/video.mp4"))
        with self.assertRaises(ValueError):
            fetch_video_source(url="http://127.0.0.1:8765/health")
        with tempfile.TemporaryDirectory() as tmp:
            secret = Path(tmp) / "notes.txt"
            secret.write_text("not a video", encoding="utf-8")
            with self.assertRaises(ValueError):
                fetch_video_source(path=str(secret))
            outside = Path(tmp) / "private.mp4"
            outside.write_bytes(b"not-really")
            dataset = Path(tmp) / "dataset"
            with self.assertRaises(ValueError):
                fetch_video_source(path=str(outside), dataset_root=dataset)
            inside = dataset / "videos" / "lesson.mp4"
            inside.parent.mkdir(parents=True, exist_ok=True)
            inside.write_bytes(b"lesson")
            path, cues, title = fetch_video_source(path=str(inside), dataset_root=dataset)
            self.assertEqual(path, inside.resolve())
            self.assertEqual(cues, [])
            self.assertEqual(title, "lesson")
            link = dataset / "videos" / "escape.mp4"
            link.symlink_to(outside)
            with self.assertRaises(ValueError):
                fetch_video_source(path=str(link), dataset_root=dataset)

    def test_youtube_unavailable_falls_back_to_local_demo(self):
        from agent.video_teacher import fetch_video_source

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = root / "videos"
            videos.mkdir(parents=True)
            demo = videos / "abcd1234ef.mp4"
            demo.write_bytes(b"fake-mp4")
            (root / "last_session.json").write_text(
                json.dumps({"session_id": "abcd1234ef", "video": "abcd1234ef.mp4", "goal": "extrude"}),
                encoding="utf-8",
            )

            def _boom(*_a, **_k):
                raise subprocess.CalledProcessError(
                    1, ["yt-dlp"], stderr="ERROR: [youtube] abc: This video is unavailable"
                )

            with mock.patch("agent.video_teacher.subprocess.run", side_effect=_boom), mock.patch(
                "agent.video_teacher._ffmpeg_available", return_value=False
            ):
                path, cues, title = fetch_video_source(
                    url="https://www.youtube.com/watch?v=JqmYYAbIqhp",
                    dataset_root=root,
                    max_minutes=1.0,
                )
            self.assertEqual(path, demo.resolve())
            self.assertEqual(cues, [])
            self.assertIn("local demo", title.lower())

    def test_subtitle_failure_retries_without_subs(self):
        from agent.video_teacher import fetch_video_source

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher = root / "videos" / "teacher"
            teacher.mkdir(parents=True)
            calls: list[list[str]] = []

            def _run(cmd, **_kwargs):
                calls.append(list(cmd))
                joined = " ".join(cmd)
                if "--write-auto-sub" in joined:
                    raise subprocess.CalledProcessError(
                        1,
                        cmd,
                        stderr="ERROR: Unable to download video subtitles for 'en': HTTP Error 429",
                    )
                # Video-only attempt: drop a file matching the session template.
                out = next(a for a in cmd if "%(ext)s" in a)
                session = Path(out).name.split(".", 1)[0]
                (teacher / f"{session}.mp4").write_bytes(b"ok")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch("agent.video_teacher.subprocess.run", side_effect=_run), mock.patch(
                "agent.video_teacher._ffmpeg_available", return_value=False
            ), mock.patch("agent.video_teacher._js_runtime_args", return_value=[]):
                path, _cues, _title = fetch_video_source(
                    url="https://www.youtube.com/watch?v=YE7VzlLtp-4",
                    dataset_root=root,
                    max_minutes=1.0,
                )
            self.assertTrue(path.is_file())
            self.assertTrue(any("--no-write-auto-subs" in " ".join(c) for c in calls))
            self.assertTrue(all("--ignore-config" in c for c in calls))
            self.assertTrue(all("--" in c for c in calls))
            self.assertTrue(all("--max-filesize" in c for c in calls))


class TestLoopVideoLearn(unittest.TestCase):
    def test_learn_from_video_ingests_concepts(self):
        from agent.loop import AgentLoop

        with tempfile.TemporaryDirectory() as tmp:
            loop = AgentLoop(dataset_root=Path(tmp))

            def _fake_learn(**kwargs):
                return {
                    "ok": True,
                    "concepts": [
                        {
                            "name": "loop cut",
                            "summary": "Add an edge loop",
                            "when_to_use": "To add resolution across a mesh",
                            "preconditions": ["edit mode"],
                            "aliases": ["ctrl r", "edge loop"],
                            "steps": [
                                {"action": "hotkey", "keys": ["ctrl", "r"], "reason": "loop cut"}
                            ],
                            "video": "x.mp4",
                            "t_start": 0,
                            "t_end": 20,
                        }
                    ],
                    "title": "x",
                    "video": "x.mp4",
                    "elapsed_sec": 0.1,
                }

            with mock.patch(
                "agent.video_teacher.learn_concepts_from_video",
                side_effect=_fake_learn,
            ):
                loop.learn_from_video(url="https://youtube.com/watch?v=abc", goal="loop cut", model="none")
                if loop._video_thread is not None:
                    loop._video_thread.join(timeout=2.0)
            self.assertFalse(loop._learning_video)
            cards = loop.memory.list_concepts()
            self.assertEqual(cards[0]["name"], "loop cut")
            self.assertIn("loop cut", (loop._concepts_summary or "").lower())


if __name__ == "__main__":
    unittest.main()
