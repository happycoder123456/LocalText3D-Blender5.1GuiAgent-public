"""Scene-ready placement math. No Blender imports so tests can load this file."""

from __future__ import annotations


def fit_scale(size_xyz: tuple[float, float, float], target_size: float) -> float:
    longest = max(abs(float(axis)) for axis in size_xyz)
    if longest < 1e-8 or target_size <= 0:
        return 1.0
    return float(target_size) / longest


def line_up_x(slot: int, target_size: float, spacing: float = 1.4) -> float:
    if slot <= 0 or target_size <= 0:
        return 0.0
    return float(slot) * float(target_size) * float(spacing)


def destination_xy(
    cursor_xy: tuple[float, float],
    *,
    slot: int,
    target_size: float,
    place_at_cursor: bool,
    line_up: bool,
) -> tuple[float, float]:
    x, y = cursor_xy if place_at_cursor else (0.0, 0.0)
    if line_up:
        x += line_up_x(slot, target_size)
    return (x, y)


def move_delta(
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
    dest_xy: tuple[float, float],
    dest_z: float,
    sit_on_ground: bool,
) -> tuple[float, float, float]:
    """World-space translation so the bbox lands at dest.

    Origin of the object does not need to be at the bbox center.
    XY is centered on dest. Z sits the lowest point on dest_z, or centers on dest_z.
    """
    minx, miny, minz = (float(v) for v in bbox_min)
    maxx, maxy, maxz = (float(v) for v in bbox_max)
    dx = dest_xy[0] - (minx + maxx) * 0.5
    dy = dest_xy[1] - (miny + maxy) * 0.5
    if sit_on_ground:
        dz = dest_z - minz
    else:
        dz = dest_z - (minz + maxz) * 0.5
    return (dx, dy, dz)
