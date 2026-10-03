from __future__ import annotations

from pathlib import Path

from worker.glb_util import write_box_glb
from worker.models import GenerateRequest, ProgressFn


class MockEngine:
    """Writes a cube GLB. Used for tests and machines without a GPU."""

    def __init__(self, name: str = "mock"):
        self.name = name
        self.loaded = False
        self.loads = 0
        self.unloads = 0
        self.generates = 0
        self.last_variant: str | None = None

    def is_available(self) -> bool:
        return True

    def load(self, request: GenerateRequest, progress: ProgressFn) -> None:
        progress("switching_engine", f"Loading mock stand-in for {self.name}")
        self.loaded = True
        self.loads += 1
        self.last_variant = request.cache_key()[1] or None

    def unload(self) -> None:
        self.loaded = False
        self.unloads += 1
        self.last_variant = None

    def generate(self, request: GenerateRequest, dest: Path, progress: ProgressFn) -> Path:
        if not self.loaded:
            self.load(request, progress)
        progress("generating", f"Mock {self.name}: building a placeholder cube")
        self.generates += 1
        progress("extracting_mesh", f"Writing {dest.name}")
        return write_box_glb(dest)
