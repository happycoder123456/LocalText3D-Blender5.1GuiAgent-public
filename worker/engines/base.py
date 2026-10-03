from __future__ import annotations

from pathlib import Path
from typing import Protocol

from worker.models import GenerateRequest, ProgressFn


class Engine(Protocol):
    name: str

    def is_available(self) -> bool: ...

    def load(self, request: GenerateRequest, progress: ProgressFn) -> None: ...

    def unload(self) -> None: ...

    def generate(self, request: GenerateRequest, dest: Path, progress: ProgressFn) -> Path: ...
