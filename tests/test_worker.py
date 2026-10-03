from __future__ import annotations

import http.client
import json
import os
import threading
import tempfile
import time
import unittest
import unittest.mock
import urllib.error
import urllib.request
from pathlib import Path

from worker.cli import default_output_dir
from worker.glb_util import write_box_glb
from worker.jobs import JobBusy, JobStore
from worker.manager import EngineManager, EngineNotInstalled
from worker.models import GenerateRequest, request_from_payload
from worker.server import make_server, request_is_local

TINY_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8A"
    "AwMCAO+ip1sAAAAASUVORK5CYII="
)


def _http_json(method: str, url: str, body: dict | None = None, timeout: float = 5.0):
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        payload = json.loads(raw.decode("utf-8")) if raw else {}
        return resp.status, payload


class GlbUtilTests(unittest.TestCase):
    def test_writes_gltf_magic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_box_glb(Path(tmp) / "cube.glb")
            data = path.read_bytes()
        self.assertGreater(len(data), 64)
        self.assertTrue(data.startswith(b"glTF"))
        self.assertEqual(int.from_bytes(data[4:8], "little"), 2)


class RequestTests(unittest.TestCase):
    def test_requires_prompt(self):
        with self.assertRaises(ValueError):
            request_from_payload({"engine": "trellis", "prompt": "  "})

    def test_rejects_unknown_engine(self):
        with self.assertRaises(ValueError):
            request_from_payload({"engine": "meshy", "prompt": "a mug"})

    def test_normalizes_shap_e_alias(self):
        req = request_from_payload({"engine": "shape-e", "prompt": "a mug"})
        self.assertEqual(req.engine, "shap_e")

    def test_defaults(self):
        req = request_from_payload({"engine": "trellis", "prompt": "a stool"})
        self.assertEqual(req.trellis_variant, "text-large")
        self.assertEqual(req.structure_steps, 25)
        self.assertEqual(req.slat_steps, 25)
        self.assertEqual(req.texture_size, 2048)
        self.assertEqual(req.source, "text")
        self.assertEqual(req.cache_key(), ("trellis", "text-large"))

    def test_image_source_default_slat_cfg(self):
        req = request_from_payload(
            {"engine": "trellis", "source": "image", "images": [TINY_PNG]}
        )
        self.assertEqual(req.slat_cfg_strength, 3.0)

    def test_texture_bake_env_error_detection(self):
        from worker.engines.trellis import _is_texture_bake_env_error

        self.assertTrue(
            _is_texture_bake_env_error(
                OSError("CUDA_HOME environment variable is not set. Please set it to your CUDA install root.")
            )
        )
        self.assertTrue(_is_texture_bake_env_error(RuntimeError("Ninja is required to load C++ extensions")))
        self.assertFalse(_is_texture_bake_env_error(RuntimeError("prompt is required")))

    def test_image_source_does_not_need_prompt(self):
        req = request_from_payload(
            {"engine": "trellis", "source": "image", "images": [TINY_PNG]}
        )
        self.assertEqual(req.source, "image")
        self.assertEqual(req.prompt, "image")
        self.assertEqual(req.cache_key(), ("trellis", "image-large"))

    def test_two_images_become_views(self):
        req = request_from_payload(
            {
                "engine": "trellis",
                "source": "orbit",
                "images": [TINY_PNG, TINY_PNG],
            }
        )
        self.assertEqual(req.source, "views")
        self.assertEqual(len(req.images), 2)

    def test_rejects_image_source_without_bytes(self):
        with self.assertRaises(ValueError):
            request_from_payload({"engine": "trellis", "source": "image"})

    def test_variant_from_scan_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            glb = write_box_glb(Path(tmp) / "scan.glb").read_bytes()
        payload = {
            "engine": "trellis",
            "source": "variant",
            "prompt": "mossy limestone stairs",
            "mesh_glb": __import__("base64").b64encode(glb).decode("ascii"),
        }
        req = request_from_payload(payload)
        self.assertEqual(req.source, "variant")
        self.assertEqual(req.cache_key(), ("trellis", "text-large"))
        self.assertTrue(req.mesh_glb)

    def test_shap_e_rejects_mesh_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            glb = write_box_glb(Path(tmp) / "scan.glb").read_bytes()
        with self.assertRaises(ValueError):
            request_from_payload(
                {
                    "engine": "shap_e",
                    "source": "variant",
                    "prompt": "mossy",
                    "mesh_glb": __import__("base64").b64encode(glb).decode("ascii"),
                }
            )

    def test_rejects_oversized_sampling_and_prompt(self):
        with self.assertRaises(ValueError):
            request_from_payload(
                {"engine": "trellis", "prompt": "a mug", "structure_steps": 10_000}
            )
        with self.assertRaises(ValueError):
            request_from_payload({"engine": "trellis", "prompt": "x" * 9000})
        req = request_from_payload(
            {"engine": "trellis", "prompt": "a mug", "structure_steps": 25, "seed": 3}
        )
        self.assertEqual(req.structure_steps, 25)
        self.assertEqual(req.seed, 3)

    def test_pil_images_rejects_huge_dimensions(self):
        import base64
        import struct
        import zlib

        try:
            from worker.images import pil_images
        except ModuleNotFoundError:
            self.skipTest("Pillow is not installed")

        def png_with_size(width: int, height: int) -> str:
            def chunk(tag: bytes, data: bytes) -> bytes:
                crc = zlib.crc32(tag + data) & 0xFFFFFFFF
                return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

            ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
            raw = (
                b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", ihdr)
                + chunk(b"IDAT", zlib.compress(b""))
                + chunk(b"IEND", b"")
            )
            return base64.b64encode(raw).decode("ascii")

        try:
            with self.assertRaises(ValueError):
                pil_images([png_with_size(20_000, 20_000)])
        except ModuleNotFoundError:
            self.skipTest("Pillow is not installed")

    def test_request_is_local_headers(self):
        from http.client import HTTPMessage

        local = HTTPMessage()
        local["Host"] = "127.0.0.1:8765"
        self.assertTrue(request_is_local(local))
        rebind = HTTPMessage()
        rebind["Host"] = "evil.example"
        self.assertFalse(request_is_local(rebind))
        csrf = HTTPMessage()
        csrf["Host"] = "127.0.0.1:8765"
        csrf["Origin"] = "https://evil.example"
        self.assertFalse(request_is_local(csrf))
        ok_origin = HTTPMessage()
        ok_origin["Host"] = "localhost:8765"
        ok_origin["Origin"] = "http://127.0.0.1:1234"
        self.assertTrue(request_is_local(ok_origin))


class EngineSwitchTests(unittest.TestCase):
    def test_switch_unloads_previous_engine(self):
        manager = EngineManager(mock=True)
        dest_dir = Path(tempfile.mkdtemp())
        trellis = manager._engines["trellis"]
        shap_e = manager._engines["shap_e"]

        manager.generate(
            GenerateRequest(engine="trellis", prompt="a chair"),
            dest_dir / "a.glb",
        )
        self.assertEqual(trellis.loads, 1)
        self.assertEqual(trellis.unloads, 0)
        self.assertTrue(trellis.loaded)
        self.assertFalse(shap_e.loaded)

        manager.generate(
            GenerateRequest(engine="shap_e", prompt="a cup"),
            dest_dir / "b.glb",
        )
        self.assertEqual(trellis.unloads, 1)
        self.assertFalse(trellis.loaded)
        self.assertEqual(shap_e.loads, 1)
        self.assertTrue(shap_e.loaded)
        self.assertEqual(manager.loaded_engine(), "shap_e")

    def test_same_engine_does_not_reload(self):
        manager = EngineManager(mock=True)
        dest_dir = Path(tempfile.mkdtemp())
        trellis = manager._engines["trellis"]
        req = GenerateRequest(engine="trellis", prompt="one")
        manager.generate(req, dest_dir / "a.glb")
        manager.generate(GenerateRequest(engine="trellis", prompt="two"), dest_dir / "b.glb")
        self.assertEqual(trellis.loads, 1)
        self.assertEqual(trellis.generates, 2)

    def test_variant_change_reloads_trellis(self):
        manager = EngineManager(mock=True)
        dest_dir = Path(tempfile.mkdtemp())
        trellis = manager._engines["trellis"]
        manager.generate(
            GenerateRequest(engine="trellis", prompt="a", trellis_variant="text-large"),
            dest_dir / "a.glb",
        )
        manager.generate(
            GenerateRequest(engine="trellis", prompt="b", trellis_variant="text-base"),
            dest_dir / "b.glb",
        )
        self.assertEqual(trellis.loads, 2)
        self.assertEqual(trellis.unloads, 1)
        self.assertEqual(trellis.last_variant, "text-base")

    def test_image_source_reloads_trellis(self):
        manager = EngineManager(mock=True)
        dest_dir = Path(tempfile.mkdtemp())
        trellis = manager._engines["trellis"]
        manager.generate(
            GenerateRequest(engine="trellis", prompt="a", trellis_variant="text-large"),
            dest_dir / "a.glb",
        )
        manager.generate(
            GenerateRequest(
                engine="trellis",
                prompt="viewport",
                source="image",
                images=[TINY_PNG],
            ),
            dest_dir / "b.glb",
        )
        self.assertEqual(trellis.loads, 2)
        self.assertEqual(trellis.unloads, 1)
        self.assertEqual(trellis.last_variant, "image-large")

    def test_variant_keeps_text_pipeline(self):
        manager = EngineManager(mock=True)
        dest_dir = Path(tempfile.mkdtemp())
        trellis = manager._engines["trellis"]
        manager.generate(
            GenerateRequest(engine="trellis", prompt="a", trellis_variant="text-large"),
            dest_dir / "a.glb",
        )
        manager.generate(
            GenerateRequest(
                engine="trellis",
                prompt="mossy stairs",
                source="variant",
                mesh_glb="unused-in-mock",
            ),
            dest_dir / "b.glb",
        )
        self.assertEqual(trellis.loads, 1)
        self.assertEqual(trellis.last_variant, "text-large")

    def test_real_manager_reports_engines_as_list(self):
        manager = EngineManager(mock=False)
        engines = manager.available_engines()
        self.assertIsInstance(engines, list)
        for name in engines:
            self.assertIn(name, {"trellis", "shap_e"})
        if "trellis" not in engines:
            with self.assertRaises(EngineNotInstalled):
                manager.generate(
                    GenerateRequest(engine="trellis", prompt="a mug"),
                    Path(tempfile.mkdtemp()) / "out.glb",
                )


class JobAndHttpTests(unittest.TestCase):
    def test_rejects_second_job_while_busy(self):
        manager = EngineManager(mock=True)
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(manager, Path(tmp))
            with store._lock:
                store._busy = True
            with self.assertRaises(JobBusy):
                store.submit(GenerateRequest(engine="shap_e", prompt="two"))
            with store._lock:
                store._busy = False
            first = store.submit(GenerateRequest(engine="trellis", prompt="one"))
            deadline = time.time() + 5
            while time.time() < deadline:
                job = store.get(first.job_id)
                if job and job.status == "done":
                    break
                time.sleep(0.05)
            else:
                self.fail("first job did not finish")
            self.assertEqual(store.get(first.job_id).status, "done")

    def test_http_generate_poll_and_download(self):
        manager = EngineManager(mock=True)
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(manager, Path(tmp))
            httpd = make_server("127.0.0.1", 0, store, manager)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = httpd.server_address
                base = f"http://{host}:{port}"

                status, health = _http_json("GET", f"{base}/health")
                self.assertEqual(status, 200)
                self.assertTrue(health["ok"])
                self.assertTrue(health["mock"])
                self.assertIn("trellis", health["available_engines"])
                self.assertIn("shap_e", health["available_engines"])

                status, created = _http_json(
                    "POST",
                    f"{base}/generate",
                    {"engine": "shap_e", "prompt": "a rubber duck", "seed": 3},
                )
                self.assertEqual(status, 202)
                job_id = created["job_id"]

                job = None
                deadline = time.time() + 5
                while time.time() < deadline:
                    _status, job = _http_json("GET", f"{base}/jobs/{job_id}")
                    if job.get("status") in {"done", "failed"}:
                        break
                    time.sleep(0.05)
                self.assertEqual(job["status"], "done")
                self.assertTrue(job["output_path"])

                dest = Path(tmp) / "downloaded.glb"
                req = urllib.request.Request(f"{base}/jobs/{job_id}/file")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    dest.write_bytes(resp.read())
                self.assertTrue(dest.read_bytes().startswith(b"glTF"))

                status, listed = _http_json("GET", f"{base}/jobs")
                self.assertEqual(status, 200)
                jobs = listed.get("jobs") or []
                self.assertTrue(any(row.get("job_id") == job_id for row in jobs))
                match = next(row for row in jobs if row.get("job_id") == job_id)
                self.assertEqual(match.get("prompt"), "a rubber duck")
                self.assertEqual(match.get("seed"), 3)
                sidecar = Path(tmp) / f"{job_id}.json"
                self.assertTrue(sidecar.is_file())
            finally:
                httpd.shutdown()
                httpd.server_close()

    def test_http_rejects_non_loopback_host_and_origin(self):
        manager = EngineManager(mock=True)
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(manager, Path(tmp))
            httpd = make_server("127.0.0.1", 0, store, manager)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                _host, port = httpd.server_address
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
                conn.putrequest("GET", "/health", skip_host=True)
                conn.putheader("Host", "evil.example")
                conn.endheaders()
                rebind = conn.getresponse()
                self.assertEqual(rebind.status, 403)
                rebind.read()
                conn.close()

                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
                conn.putrequest("GET", "/health", skip_host=True)
                conn.putheader("Host", f"127.0.0.1:{port}")
                conn.putheader("Origin", "https://evil.example")
                conn.endheaders()
                csrf = conn.getresponse()
                self.assertEqual(csrf.status, 403)
                csrf.read()
                conn.close()

                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
                conn.putrequest("POST", "/generate")
                conn.putheader("Host", f"127.0.0.1:{port}")
                conn.putheader("Content-Type", "application/json")
                conn.putheader("Content-Length", "-1")
                conn.endheaders()
                bad_len = conn.getresponse()
                self.assertEqual(bad_len.status, 400)
                bad_len.read()
                conn.close()
            finally:
                httpd.shutdown()
                httpd.server_close()

    def test_http_rejects_oversized_glb_download(self):
        from worker import server as worker_server

        manager = EngineManager(mock=True)
        with tempfile.TemporaryDirectory() as tmp, unittest.mock.patch.object(
            worker_server, "MAX_GLB_DOWNLOAD_BYTES", 32
        ):
            store = JobStore(manager, Path(tmp))
            httpd = make_server("127.0.0.1", 0, store, manager)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = httpd.server_address
                base = f"http://{host}:{port}"
                _status, created = _http_json(
                    "POST", f"{base}/generate", {"engine": "shap_e", "prompt": "a cube"}
                )
                job_id = created["job_id"]
                deadline = time.time() + 5
                while time.time() < deadline:
                    _status, job = _http_json("GET", f"{base}/jobs/{job_id}")
                    if job.get("status") in {"done", "failed"}:
                        break
                    time.sleep(0.05)
                self.assertEqual(job["status"], "done")
                req = urllib.request.Request(f"{base}/jobs/{job_id}/file")
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(req, timeout=5)
                self.assertEqual(ctx.exception.code, 413)
            finally:
                httpd.shutdown()
                httpd.server_close()


class SidecarHistoryTests(unittest.TestCase):
    def test_list_recent_reads_sidecar_after_new_store(self):
        manager = EngineManager(mock=True)
        with tempfile.TemporaryDirectory() as tmp:
            first = JobStore(manager, Path(tmp))
            job = first.submit(GenerateRequest(engine="trellis", prompt="a mug", seed=11))
            deadline = time.time() + 5
            while time.time() < deadline:
                current = first.get(job.job_id)
                if current and current.status == "done":
                    break
                time.sleep(0.05)
            else:
                self.fail("job did not finish")
            second = JobStore(manager, Path(tmp))
            recent = second.list_recent()
            ids = [row.get("job_id") for row in recent]
            self.assertIn(job.job_id, ids)
            row = next(item for item in recent if item.get("job_id") == job.job_id)
            self.assertEqual(row.get("prompt"), "a mug")
            self.assertEqual(row.get("seed"), 11)

    def test_list_recent_skips_huge_sidecar(self):
        from worker.jobs import MAX_SIDECAR_BYTES, JobStore

        manager = EngineManager(mock=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            glb = root / "deadbeef.glb"
            write_box_glb(glb)
            payload = {
                "job_id": "deadbeef",
                "output_path": str(glb),
                "status": "done",
                "created": 1.0,
                "prompt": "pad",
                "padding": "x" * (MAX_SIDECAR_BYTES + 8),
            }
            (root / "deadbeef.json").write_text(json.dumps(payload), encoding="utf-8")
            store = JobStore(manager, root)
            recent = store.list_recent()
            self.assertFalse(any(row.get("job_id") == "deadbeef" for row in recent))

    def test_list_recent_ignores_glb_outside_output_dir(self):
        from worker.jobs import JobStore

        manager = EngineManager(mock=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            outside = Path(tmp) / "secret.glb"
            write_box_glb(outside)
            payload = {
                "job_id": "deadbeef",
                "output_path": str(outside),
                "status": "done",
                "created": 1.0,
                "prompt": "outside",
            }
            (root / "deadbeef.json").write_text(json.dumps(payload), encoding="utf-8")
            store = JobStore(manager, root)
            recent = store.list_recent()
            self.assertFalse(any(row.get("job_id") == "deadbeef" for row in recent))


class StudioMathTests(unittest.TestCase):
    def _load_studio(self):
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "addon" / "studio.py"
        spec = importlib.util.spec_from_file_location("localtext3d_studio", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_fit_scale_uses_longest_axis(self):
        studio = self._load_studio()
        self.assertAlmostEqual(studio.fit_scale((2.0, 1.0, 0.5), 1.0), 0.5)
        self.assertEqual(studio.fit_scale((0.0, 0.0, 0.0), 1.0), 1.0)

    def test_line_up_and_grounding(self):
        studio = self._load_studio()
        self.assertAlmostEqual(studio.line_up_x(2, 1.0, spacing=1.4), 2.8)
        self.assertEqual(studio.line_up_x(0, 1.0), 0.0)
        xy = studio.destination_xy(
            (10.0, 3.0),
            slot=1,
            target_size=1.0,
            place_at_cursor=True,
            line_up=True,
        )
        self.assertAlmostEqual(xy[0], 11.4)
        self.assertAlmostEqual(xy[1], 3.0)
        delta = studio.move_delta((-1.0, -2.0, -3.0), (1.0, 2.0, 5.0), (0.0, 0.0), 0.0, True)
        self.assertAlmostEqual(delta[0], 0.0)
        self.assertAlmostEqual(delta[1], 0.0)
        self.assertAlmostEqual(delta[2], 3.0)


class ScanLibraryTests(unittest.TestCase):
    def _load_library(self):
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "addon" / "library.py"
        spec = importlib.util.spec_from_file_location("localtext3d_library", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_reads_uassets_catalog(self):
        import json

        library = self._load_library()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "UAssets" / "scjuQ"
            folder.mkdir(parents=True)
            (folder / "preview.png").write_bytes(
                __import__("base64").b64decode(TINY_PNG)
            )
            catalog = root / "UAssets" / "uassetsData.json"
            catalog.write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "id": "scjuQ",
                                "name": "Castle Stairs",
                                "assetType": "3D asset",
                                "searchStr": "castle stairs medieval",
                                "previewUrl": "http://localhost:5017/scjuQ/scjuQ.jpg",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            items = library.list_scan_assets(root, query="stairs")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["name"], "Castle Stairs")
        self.assertTrue(items[0]["preview_file"].endswith("preview.png"))

    def test_embedded_preview_accepts_string_entries(self):
        import json

        library = self._load_library()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "asset.json"
            path.write_text(
                json.dumps({"previews": {"images": ["not-a-dict", {"uri": "x" * 50}]}}),
                encoding="utf-8",
            )
            self.assertEqual(library._embedded_preview(path), "")

    def test_picks_best_match_without_asking(self):
        import json

        library = self._load_library()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for asset_id, name, kind, search in (
                ("scjuQ", "Castle Stairs", "3D asset", "castle stairs medieval steps"),
                ("rjenotp2", "Large Asphalt Patch", "decal", "asphalt patch road"),
                ("tlhjacuva", "Asphalt Debris Pack", "3D asset", "asphalt debris scatter"),
            ):
                folder = root / "UAssets" / asset_id
                folder.mkdir(parents=True)
                (folder / "preview.png").write_bytes(__import__("base64").b64decode(TINY_PNG))
            catalog = root / "UAssets" / "uassetsData.json"
            catalog.write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "id": "scjuQ",
                                "name": "Castle Stairs",
                                "assetType": "3D asset",
                                "searchStr": "castle stairs medieval steps",
                            },
                            {
                                "id": "rjenotp2",
                                "name": "Large Asphalt Patch",
                                "assetType": "decal",
                                "searchStr": "asphalt patch road",
                            },
                            {
                                "id": "tlhjacuva",
                                "name": "Asphalt Debris Pack",
                                "assetType": "3D asset",
                                "searchStr": "asphalt debris scatter",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            assets = library.list_scan_assets(root, limit=40)
            stairs = library.best_match(assets, "weathered castle stairs")
            debris = library.best_match(assets, "asphalt debris")
            none = library.best_match(assets, "a wooden stool with three legs")
            look = library.pick_auto_scan(assets, "road with asphalt")
            shape = library.pick_auto_scan(assets, "weathered castle stairs")
            missing = library.pick_auto_scan(assets, "a wooden stool with three legs")
        self.assertIsNotNone(stairs)
        self.assertEqual(stairs["name"], "Castle Stairs")
        self.assertEqual(debris["name"], "Asphalt Debris Pack")
        self.assertIsNone(none)
        self.assertIsNotNone(look)
        self.assertEqual(look[0], "look")
        self.assertIn("Asphalt", look[1]["name"])
        self.assertEqual(shape[0], "shape")
        self.assertEqual(shape[1]["name"], "Castle Stairs")
        self.assertIsNone(missing)

    def test_finds_documents_megascans_if_present(self):
        library = self._load_library()
        root = library.resolve_library_root("")
        docs = Path.home() / "Documents" / "Megascans Library"
        if not docs.is_dir():
            self.skipTest("No local Megascans Library")
        self.assertEqual(root, docs)


class OutputDirTests(unittest.TestCase):
    def test_linux_default_uses_cache(self):
        if os.name == "nt":
            self.skipTest("Windows uses LOCALAPPDATA")
        path = default_output_dir()
        self.assertIn("localtext3d", str(path))


class AddonLayoutTests(unittest.TestCase):
    def test_manifest_is_complete(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "addon" / "blender_manifest.toml").read_text(encoding="utf-8")
        for needle in (
            'id = "localtext3d"',
            'blender_version_min = "5.1.0"',
            'type = "add-on"',
            "SPDX:GPL-3.0-or-later",
        ):
            self.assertIn(needle, text)
        # Version is owned by the manifest; only require a well-formed semver.
        self.assertRegex(text, r'(?m)^version = "\d+\.\d+\.\d+"$')
        for name in (
            "__init__.py",
            "ui.py",
            "operators.py",
            "prefs.py",
            "client.py",
            "props.py",
            "studio.py",
            "capture.py",
            "library.py",
            "scan.py",
            "llm_client.py",
            "gn_spec.py",
            "gn_prompts.py",
            "gn_builder.py",
            "gn_operators.py",
            "gn_instructions.py",
            "agent_ops.py",
            "agent_props.py",
            "agent_ui.py",
            "agent_client.py",
            "agent_context.py",
        ):
            self.assertTrue((root / "addon" / name).is_file(), name)

    def test_build_addon_zip_matches_manifest_version(self):
        import importlib.util

        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("build_addon", root / "scripts" / "build_addon.py")
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        version = mod.addon_version()
        self.assertRegex(version, r"^\d+\.\d+\.\d+$")
        self.assertEqual(mod.zip_path().name, f"localtext3d-{version}.zip")

    def test_cancel_marks_job_and_frees_worker(self):
        import time

        from worker.jobs import JobStore
        from worker.manager import EngineManager
        from worker.models import GenerateRequest

        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(EngineManager(mock=True), Path(tmp))
            job = store.submit(GenerateRequest(engine="mock", prompt="cube"))
            store.cancel(job.job_id)
            deadline = time.time() + 5.0
            while time.time() < deadline:
                current = store.get(job.job_id)
                if current is not None and current.status in {"done", "cancelled", "failed"}:
                    break
                time.sleep(0.05)
            current = store.get(job.job_id)
            self.assertIsNotNone(current)
            self.assertIn(current.status, {"cancelled", "done"})
            # Worker slot is free again either way.
            self.assertFalse(store._busy)
            self.assertFalse(store.cancel(job.job_id))


class AddonClientUrlTests(unittest.TestCase):
    def _load_client(self):
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "addon" / "client.py"
        spec = importlib.util.spec_from_file_location("localtext3d_client", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_require_loopback_http(self):
        client = self._load_client()
        client.require_loopback_http("http://127.0.0.1:8765")
        client.require_loopback_http("http://localhost:8765")
        client.require_loopback_http("http://[::1]:8765")
        with self.assertRaises(client.WorkerError):
            client.require_loopback_http("http://example.com:8765")
        with self.assertRaises(client.WorkerError):
            client.require_loopback_http("https://127.0.0.1:8765")
        with self.assertRaises(client.WorkerError):
            client.require_loopback_http("http://127.0.0.1:8765@evil.example/")
        with self.assertRaises(client.WorkerError):
            client.require_loopback_http("http://127.0.0.1:8766")
        with self.assertRaises(client.WorkerError):
            client.require_loopback_http("http://127.0.0.1")
        with self.assertRaises(client.WorkerError):
            client.require_loopback_http("http://127.0.0.1:11434")

    def test_glb_path_stays_in_worker_output(self):
        import os
        from unittest import mock

        client = self._load_client()
        from worker.cli import default_output_dir

        self.assertEqual(client.default_worker_output_dir(), default_output_dir())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            inside = root / "job.glb"
            inside.write_bytes(b"glb")
            outside = Path(tmp) / "secret.glb"
            outside.write_bytes(b"glb")
            link = root / "escape.glb"
            link.symlink_to(outside)
            with mock.patch.dict(os.environ, {"LOCALTEXT3D_OUTPUT_DIR": str(root)}):
                self.assertEqual(client.glb_under_worker_output(str(inside)), str(inside.resolve()))
                self.assertEqual(
                    client.glb_under_worker_output(str(root / "sub" / ".." / "job.glb")),
                    str(inside.resolve()),
                )
                self.assertIsNone(client.glb_under_worker_output(str(outside)))
                self.assertIsNone(client.glb_under_worker_output(str(link)))
                self.assertIsNone(client.glb_under_worker_output(str(root / "missing.glb")))
                self.assertIsNone(client.glb_under_worker_output(str(root / "notes.txt")))

    def _load_agent_client(self):
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "addon" / "agent_client.py"
        spec = importlib.util.spec_from_file_location("localtext3d_agent_client", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_sidecar_url_pinned_to_8766(self):
        client = self._load_agent_client()
        client.require_loopback_http("http://127.0.0.1:8766")
        client.require_loopback_http("http://localhost:8766")
        client.require_loopback_http("http://[::1]:8766")
        with self.assertRaises(client.AgentClientError):
            client.require_loopback_http("http://127.0.0.1:8765")
        with self.assertRaises(client.AgentClientError):
            client.require_loopback_http("http://127.0.0.1:11434")
        with self.assertRaises(client.AgentClientError):
            client.require_loopback_http("http://example.com:8766")

    def test_stage_video_copies_only_into_dataset(self):
        client = self._load_agent_client()
        from agent.paths import default_dataset_dir

        self.assertEqual(client.default_agent_dataset_dir(), default_dataset_dir())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            outside = Path(tmp) / "lesson.mp4"
            outside.write_bytes(b"video-bytes")
            staged = Path(client.stage_video_into_dataset(str(outside), root))
            self.assertTrue(staged.is_file())
            staged.relative_to(root.resolve())
            self.assertEqual(staged.read_bytes(), b"video-bytes")
            again = Path(client.stage_video_into_dataset(str(outside), root))
            self.assertEqual(again, staged)
            inside = root / "videos" / "already.mp4"
            inside.parent.mkdir(parents=True, exist_ok=True)
            inside.write_bytes(b"in")
            self.assertEqual(client.stage_video_into_dataset(str(inside), root), str(inside.resolve()))
            notes = Path(tmp) / "notes.txt"
            notes.write_text("nope", encoding="utf-8")
            with self.assertRaises(client.AgentClientError):
                client.stage_video_into_dataset(str(notes), root)


class LoopbackBindTests(unittest.TestCase):
    def test_rejects_lan_and_all_interfaces(self):
        from worker.loopback import require_loopback_bind, require_loopback_port

        self.assertEqual(require_loopback_bind("127.0.0.1"), "127.0.0.1")
        self.assertEqual(require_loopback_bind("localhost"), "localhost")
        with self.assertRaises(ValueError):
            require_loopback_bind("0.0.0.0")
        with self.assertRaises(ValueError):
            require_loopback_bind("192.168.1.10")
        self.assertEqual(require_loopback_port(8765), 8765)
        with self.assertRaises(ValueError):
            require_loopback_port(0)
        with self.assertRaises(ValueError):
            require_loopback_port(70000)


if __name__ == "__main__":
    unittest.main()
