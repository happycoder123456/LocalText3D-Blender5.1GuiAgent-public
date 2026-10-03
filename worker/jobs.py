from __future__ import annotations

import json
import queue
import threading
import uuid
from pathlib import Path

from worker.device import device_info
from worker.manager import EngineManager, EngineNotInstalled
from worker.models import GenerateRequest, Job

MAX_SIDECAR_BYTES = 256_000
MAX_SIDECAR_SCAN = 80


class JobBusy(RuntimeError):
    pass


class JobCancelled(RuntimeError):
    """Raised inside the worker thread when the user cancelled the running job."""


class JobStore:
    def __init__(self, manager: EngineManager, output_dir: Path) -> None:
        self.manager = manager
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._busy = False
        self._cancelled: set[str] = set()
        self._thread = threading.Thread(target=self._loop, name="localtext3d-worker", daemon=True)
        self._thread.start()

    def cancel(self, job_id: str) -> bool:
        """Cooperative cancel: the job aborts at its next progress checkpoint.

        Engines report progress between stages, so a cancelled TRELLIS/Shap-E job
        stops within one stage instead of blocking the worker until it finishes.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status in {"done", "failed", "cancelled"}:
                return False
            self._cancelled.add(job_id)
            job.status = "cancelling"
            job.message = "Cancelling after the current stage"
        return True

    def submit(self, request: GenerateRequest) -> Job:
        with self._lock:
            if self._busy or not self._queue.empty():
                vram_mb = device_info().get("vram_total_mb")
                vram_note = f"{vram_mb} MB" if isinstance(vram_mb, int) else "your GPU"
                raise JobBusy(
                    "A generation is already running. Wait for it to finish. "
                    f"Only one job at a time keeps {vram_note} from running out of memory."
                )
            job = Job(job_id=uuid.uuid4().hex, request=request)
            self._jobs[job.job_id] = job
            self._busy = True
        self._queue.put(job.job_id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _release_request_blobs(self, job_id: str) -> None:
        """Drop base64 images/meshes from finished jobs so they are not retained in RAM."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.request.images = []
            job.request.mesh_glb = ""

    def list_recent(self, limit: int = 24) -> list[dict]:
        cap = max(1, min(int(limit), 50))
        items: dict[str, dict] = {}

        with self._lock:
            for job in self._jobs.values():
                if job.status == "done" and job.output_path:
                    payload = job.to_dict()
                    items[job.job_id] = payload

        try:
            sidecars = list(self.output_dir.glob("*.json"))
        except OSError:
            sidecars = []

        def _mtime(path: Path) -> float:
            try:
                return path.stat().st_mtime
            except OSError:
                return 0.0

        sidecars.sort(key=_mtime, reverse=True)
        for sidecar in sidecars[: max(MAX_SIDECAR_SCAN, cap * 4)]:
            try:
                if sidecar.stat().st_size > MAX_SIDECAR_BYTES:
                    continue
                data = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            job_id = str(data.get("job_id") or sidecar.stem)
            glb = self._sidecar_glb(data, sidecar)
            if glb is None:
                continue
            data["job_id"] = job_id
            data["output_path"] = str(glb)
            data["status"] = "done"
            items.setdefault(job_id, data)

        ranked = list(items.values())
        ranked.sort(key=lambda row: float(row.get("created") or 0.0), reverse=True)
        return ranked[:cap]

    def _sidecar_glb(self, data: dict, sidecar: Path) -> Path | None:
        raw = str(data.get("output_path") or "")
        candidates = []
        if raw:
            candidates.append(Path(raw))
        candidates.append(sidecar.with_suffix(".glb"))
        root = self.output_dir.resolve()
        for path in candidates:
            try:
                resolved = path.resolve()
                resolved.relative_to(root)
            except (OSError, ValueError):
                continue
            if resolved.is_file():
                return resolved
        return None

    def _write_sidecar(self, job: Job) -> None:
        if not job.output_path:
            return
        dest = self.output_dir / f"{job.job_id}.json"
        dest.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")

    def _update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in fields.items():
                setattr(job, key, value)

    def _progress(self, job_id: str, status: str, message: str) -> None:
        with self._lock:
            if job_id in self._cancelled:
                raise JobCancelled("Cancelled by user")
        self._update(job_id, status=status, message=message)

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                # Unknown id must not wedge the store with _busy stuck on.
                with self._lock:
                    self._busy = False
                continue
            try:
                with self._lock:
                    if job_id in self._cancelled:
                        raise JobCancelled("Cancelled before start")
                dest = self.output_dir / f"{job.job_id}.glb"
                path = self.manager.generate(
                    job.request,
                    dest,
                    progress=lambda status, message, _id=job_id: self._progress(_id, status, message),
                )
                self._update(job_id, output_path=str(path), error=None)
                finished = self.get(job_id)
                if finished is not None:
                    try:
                        self._write_sidecar(finished)
                    except OSError:
                        pass
                self._update(
                    job_id,
                    status="done",
                    message="Mesh is ready",
                    output_path=str(path),
                    error=None,
                )
            except JobCancelled:
                self._update(job_id, status="cancelled", message="Cancelled", error=None)
            except EngineNotInstalled as exc:
                self._update(job_id, status="failed", message=str(exc), error=str(exc))
            except Exception as exc:
                self._update(job_id, status="failed", message="Generation failed", error=str(exc))
            finally:
                self._release_request_blobs(job_id)
                with self._lock:
                    self._cancelled.discard(job_id)
                    self._busy = False
