"""Unit tests for Geometry Nodes instruction files."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_INSTRUCTIONS_PATH = Path(__file__).resolve().parents[1] / "addon" / "gn_instructions.py"
_loader = importlib.util.spec_from_file_location("gn_instructions", _INSTRUCTIONS_PATH)
assert _loader and _loader.loader
gn_instructions = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(gn_instructions)

SPEC = {
    "name": "Highway",
    "description": "Procedural road strip",
    "inputs": [{"name": "Road Width", "type": "FLOAT", "default": 6.0}],
}


class TestGnInstructions(unittest.TestCase):
    def test_write_instructions_file(self):
        # patch.dict restores LOCALAPPDATA so later tests don't point at a deleted temp dir.
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"LOCALAPPDATA": tmp}):
            path = gn_instructions.write_instructions_file(
                object_name="Highway",
                prompt="procedural highway",
                model="llama3.2",
                seed=7,
                spec=SPEC,
                instructions_text="1. Select the object.\n2. Adjust Road Width.",
            )
            self.assertTrue(os.path.isfile(path))
            text = Path(path).read_text(encoding="utf-8")
            self.assertIn("Road Width", text)
            self.assertIn("procedural highway", text)
            self.assertNotIn("...", text)

    def test_instruction_index_is_capped(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"LOCALAPPDATA": tmp}):
            with mock.patch.object(Path, "exists", return_value=True):
                with self.assertRaises(RuntimeError):
                    gn_instructions.write_instructions_file(
                        object_name="Highway",
                        prompt="procedural highway",
                        model="llama3.2",
                        seed=7,
                        spec=SPEC,
                        instructions_text="1. Select the object.",
                    )


if __name__ == "__main__":
    unittest.main()
