"""Write a tiny valid GLB without third-party mesh libraries."""

from __future__ import annotations

import json
import struct
from pathlib import Path


def write_box_glb(path: str | Path, size: float = 1.0) -> Path:
    """Write an axis-aligned cube GLB. Used by the mock engine and tests."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    h = size / 2.0
    positions = [
        -h, -h, -h,
         h, -h, -h,
         h,  h, -h,
        -h,  h, -h,
        -h, -h,  h,
         h, -h,  h,
         h,  h,  h,
        -h,  h,  h,
    ]
    # glTF front faces are counter-clockwise; every triangle below winds CCW
    # seen from outside so the cube imports with outward normals.
    indices = [
        0, 2, 1, 0, 3, 2,  # -Z
        4, 5, 6, 4, 6, 7,  # +Z
        0, 5, 4, 0, 1, 5,  # -Y
        2, 7, 6, 2, 3, 7,  # +Y
        0, 7, 3, 0, 4, 7,  # -X
        1, 6, 5, 1, 2, 6,  # +X
    ]

    pos_bytes = b"".join(struct.pack("<f", value) for value in positions)
    idx_bytes = b"".join(struct.pack("<H", value) for value in indices)
    # BIN chunk must be 4-byte aligned.
    bin_blob = pos_bytes + idx_bytes
    bin_pad = (4 - (len(bin_blob) % 4)) % 4
    bin_blob += b"\x00" * bin_pad

    pos_len = len(pos_bytes)
    idx_len = len(idx_bytes)
    gltf = {
        "asset": {"version": "2.0", "generator": "localtext3d-mock"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "MockMesh"}],
        "meshes": [
            {
                "name": "MockMesh",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        "mode": 4,
                    }
                ],
            }
        ],
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": pos_len, "target": 34962},
            {"buffer": 0, "byteOffset": pos_len, "byteLength": idx_len, "target": 34963},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 8,
                "type": "VEC3",
                "min": [-h, -h, -h],
                "max": [h, h, h],
            },
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": 36,
                "type": "SCALAR",
            },
        ],
    }

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_pad = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * json_pad

    total = 12 + 8 + len(json_bytes) + 8 + len(bin_blob)
    header = struct.pack("<4sII", b"glTF", 2, total)
    json_chunk = struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
    bin_chunk = struct.pack("<I4s", len(bin_blob), b"BIN\x00") + bin_blob
    dest.write_bytes(header + json_chunk + bin_chunk)
    return dest
