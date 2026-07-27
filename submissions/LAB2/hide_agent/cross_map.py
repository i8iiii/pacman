"""Spawn-relative vertical band helpers."""


def vertical_band(position, rows):
    """Return the proportional vertical band containing position."""

    row = int(position[0])
    rows = max(1, int(rows))
    scaled = 3 * row
    if scaled < rows:
        return "top"
    if scaled < 2 * rows:
        return "middle"
    return "bottom"


def opposite_outer_band(ghost_spawn, rows):
    """Return the outer third opposite the original spawn half."""

    return (
        "bottom"
        if int(ghost_spawn[0]) < int(rows) / 2.0
        else "top"
    )
