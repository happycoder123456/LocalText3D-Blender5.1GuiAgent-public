from __future__ import annotations

from pathlib import Path

from worker.engines.mock import MockEngine
from worker.engines.shap_e import ShapEEngine
from worker.engines.trellis import TrellisEngine
from worker.models import GenerateRequest, ProgressFn, noop_progress


class EngineNotInstalled(RuntimeError):
    pass


class EngineManager:
    """Keeps at most one real model in VRAM. Switching unloads the previous engine."""

    def __init__(self, mock: bool = False) -> None:
        self.mock = mock
        if mock:
            self._engines = {
                "trellis": MockEngine("trellis"),
                "shap_e": MockEngine("shap_e"),
                "mock": MockEngine("mock"),
            }
        else:
            self._engines = {
                "trellis": TrellisEngine(),
                "shap_e": ShapEEngine(),
                "mock": MockEngine("mock"),
            }
        self._loaded_key: tuple[str, str] | None = None

    def available_engines(self) -> list[str]:
        names = []
        for name in ("trellis", "shap_e"):
            if self._engines[name].is_available():
                names.append(name)
        if self.mock and "mock" not in names:
            names.append("mock")
        return names

    def loaded_engine(self) -> str | None:
        if not self._loaded_key:
            return None
        return self._loaded_key[0]

    def loaded_variant(self) -> str | None:
        if not self._loaded_key:
            return None
        return self._loaded_key[1] or None

    def _engine_for(self, request: GenerateRequest):
        if request.engine not in self._engines:
            raise EngineNotInstalled(f"Unknown engine '{request.engine}'")
        engine = self._engines[request.engine]
        if not engine.is_available():
            if request.engine == "trellis":
                raise EngineNotInstalled(
                    "TRELLIS is not installed in this worker. "
                    "On Windows run: py -3.13 scripts\\setup_trellis.py "
                    "then restart START.bat (uses .venv-trellis)."
                )
            if request.engine == "shap_e":
                raise EngineNotInstalled(
                    "Shap-E is not installed in this worker. "
                    "Run: pip install -r worker/requirements-shap-e.txt"
                )
            raise EngineNotInstalled(f"Engine '{request.engine}' is not available")
        return engine

    def ensure_loaded(self, request: GenerateRequest, progress: ProgressFn = noop_progress) -> None:
        key = request.cache_key()
        if self._loaded_key == key:
            return
        # Resolve the target first: asking for an uninstalled engine must not
        # evict the one that is already working.
        engine = self._engine_for(request)
        if self._loaded_key is not None:
            previous = self._engines[self._loaded_key[0]]
            progress("switching_engine", f"Unloading {self._loaded_key[0]} to free VRAM")
            previous.unload()
            self._loaded_key = None
        engine.load(request, progress)
        self._loaded_key = key

    def generate(
        self,
        request: GenerateRequest,
        dest: Path,
        progress: ProgressFn = noop_progress,
    ) -> Path:
        self.ensure_loaded(request, progress)
        engine = self._engine_for(request)
        return engine.generate(request, dest, progress)

    def unload_all(self) -> None:
        if self._loaded_key is None:
            return
        self._engines[self._loaded_key[0]].unload()
        self._loaded_key = None
