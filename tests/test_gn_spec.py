"""Unit tests for Geometry Nodes JSON spec validation."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

_SPEC_PATH = Path(__file__).resolve().parents[1] / "addon" / "gn_spec.py"
_loader = importlib.util.spec_from_file_location("gn_spec", _SPEC_PATH)
assert _loader and _loader.loader
gn_spec = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(gn_spec)

GnSpecError = gn_spec.GnSpecError
parse_and_validate = gn_spec.parse_and_validate
validate_spec = gn_spec.validate_spec
format_instructions = gn_spec.format_instructions

VALID_WALL = {
    "name": "StoneWall",
    "description": "Grid wall with random missing bricks",
    "instructions": [
        "Select the wall object and open the Geometry Nodes modifier.",
        "Adjust Wall Width to change the wall length.",
    ],
    "inputs": [
        {"name": "Wall Width", "type": "FLOAT", "default": 4.0, "min": 0.5, "max": 20.0},
    ],
    "nodes": [
        {"id": "grid", "type": "MeshGrid", "inputs": {"Size X": 4.0, "Vertices X": 16}},
        {"id": "extrude", "type": "ExtrudeMesh", "inputs": {"Offset Scale": 0.08}},
    ],
    "links": [
        {"from": "grid", "from_socket": "Mesh", "to": "extrude", "to_socket": "Mesh"},
        {"from": "extrude", "from_socket": "Mesh", "to": "output", "to_socket": "Geometry"},
    ],
}


class TestGnSpec(unittest.TestCase):
    def test_valid_wall_spec(self):
        spec = validate_spec(VALID_WALL)
        self.assertEqual(spec["name"], "StoneWall")
        self.assertEqual(len(spec["nodes"]), 2)

    def test_rejects_unknown_node_type(self):
        bad = {
            **VALID_WALL,
            "nodes": [{"id": "bad", "type": "FakeNode", "inputs": {}}],
            "links": [],
        }
        with self.assertRaises(GnSpecError):
            validate_spec(bad)

    def test_rejects_broken_link_reference(self):
        bad = {
            **VALID_WALL,
            "links": [{"from": "missing", "from_socket": "Mesh", "to": "output", "to_socket": "Geometry"}],
        }
        with self.assertRaises(GnSpecError):
            validate_spec(bad)

    def test_parse_json_with_fences(self):
        text = "```json\n" + json.dumps(VALID_WALL) + "\n```"
        spec = parse_and_validate(text)
        self.assertEqual(spec["name"], "StoneWall")

    def test_rejects_empty_nodes(self):
        bad = {**VALID_WALL, "nodes": []}
        with self.assertRaises(GnSpecError):
            validate_spec(bad)

    def test_rejects_missing_instructions(self):
        bad = {k: v for k, v in VALID_WALL.items() if k != "instructions"}
        with self.assertRaises(GnSpecError):
            validate_spec(bad)

    def test_format_instructions(self):
        text = format_instructions(VALID_WALL)
        self.assertIn("1.", text)
        self.assertIn("Wall Width", text)

    def test_rejects_oversized_graph_and_clamps_vertices(self):
        huge = {
            **VALID_WALL,
            "nodes": [
                {"id": f"n{i}", "type": "MeshGrid", "inputs": {}} for i in range(gn_spec.MAX_NODES + 1)
            ],
            "links": [],
        }
        with self.assertRaises(GnSpecError):
            validate_spec(huge)
        spec = validate_spec(
            {
                **VALID_WALL,
                "nodes": [
                    {
                        "id": "grid",
                        "type": "MeshGrid",
                        "inputs": {"Size X": 4.0, "Vertices X": 10_000_000},
                    }
                ],
                "links": [{"from": "grid", "from_socket": "Mesh", "to": "output", "to_socket": "Geometry"}],
            }
        )
        self.assertEqual(spec["nodes"][0]["inputs"]["Vertices X"], gn_spec._MAX_VERTEX)
        self.assertEqual(spec["nodes"][0]["inputs"]["Size X"], 4.0)


if __name__ == "__main__":
    unittest.main()
