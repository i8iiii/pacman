"""Optional structured diagnostics for the seek agent."""

import json
import os
from pathlib import Path

import numpy as np


PACMAN_SEEK_DIAGNOSTICS = True
_ENABLED_VALUES = {"1", "true", "yes", "on"}
_DISABLED_VALUES = {"0", "false", "no", "off"}


def _diagnostics_enabled_from_environment():
    configured = os.getenv("PACMAN_SEEK_DIAGNOSTICS")
    if configured is None:
        return PACMAN_SEEK_DIAGNOSTICS

    configured = configured.strip().lower()
    if configured in _ENABLED_VALUES:
        return True
    if configured in _DISABLED_VALUES:
        return False
    return PACMAN_SEEK_DIAGNOSTICS


class SeekDiagnostics:
    """Write enough state to reconstruct each seeker decision.

    Diagnostics are enabled by default and must never affect agent behavior.
    Set ``PACMAN_SEEK_DIAGNOSTICS=0`` to disable diagnostic output.
    """

    def __init__(self, enabled=None, log_path=None, area_path=None):
        if enabled is None:
            enabled = _diagnostics_enabled_from_environment()

        self.enabled = bool(enabled)
        self.log_path = Path(log_path) if log_path else (
            Path(__file__).resolve().parent.parent / "debug" / "seek-agent.jsonl"
        )
        self.area_path = Path(area_path) if area_path else (
            Path(__file__).resolve().parent.parent / "debug" / "seek-agent-areas.txt"
        )

        if self.enabled:
            self._safely_prepare_log()

    def reset_for_match(self):
        """Start a fresh diagnostic log for a newly detected match."""
        if self.enabled:
            self._safely_prepare_log()

    def _safely_prepare_log(self):
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.area_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("", encoding="utf-8")
            self.area_path.write_text("", encoding="utf-8")
        except Exception:
            self.enabled = False

    def write_area_analysis(self, analysis, map_state, cache_hit):
        """Write a static, human-readable representation of the map areas."""
        if not self.enabled or analysis is None:
            return

        try:
            observation = np.asarray(map_state)
            symbols = {
                area.area_id: _area_symbol(area.area_id)
                for area in analysis.areas
            }
            lines = [
                "SEEK AGENT AREA MAP",
                f"fingerprint: {analysis.fingerprint}",
                f"shape: {analysis.shape[0]} x {analysis.shape[1]}",
                f"areas: {len(analysis.areas)}",
                f"analysis_seconds: {analysis.analysis_seconds:.6f}",
                f"cache_hit: {bool(cache_hit)}",
                f"error: {analysis.error or 'none'}",
                "",
                "MAP (### = wall, centered value = area ID)",
            ]
            lines.extend(
                _render_area_grid(
                    observation,
                    analysis.cell_to_area,
                    symbols,
                )
            )

            lines.extend(("", "AREAS"))
            for area in analysis.areas:
                lines.extend(
                    (
                        (
                            f"[{symbols[area.area_id]}] AREA {area.area_id} "
                            f"{area.position_label}"
                        ),
                        f"  cells: {len(area.cells)}",
                        (
                            "  centroid: "
                            f"({area.centroid[0]:.2f}, {area.centroid[1]:.2f})"
                        ),
                        f"  viewpoints ({len(area.viewpoints)}): {list(area.viewpoints)}",
                        f"  neighbors: {list(area.neighbors)}",
                    )
                )

            lines.extend(("", "GATEWAYS"))
            if not analysis.gateways:
                lines.append("none")
            for gateway in analysis.gateways:
                lines.append(
                    f"AREA {gateway.area_a} <-> AREA {gateway.area_b}: "
                    f"{list(gateway.connections)}"
                )
            self.area_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            return

    def record_decision(
        self,
        *,
        step_number,
        mode,
        previous_mode,
        transition_reasons,
        map_state,
        topology,
        my_position,
        enemy_position,
        last_seen_position,
        last_seen_step,
        target,
        path,
        move,
        move_steps,
        duration_seconds,
        area_analysis=None,
        area_cache_hit=False,
        current_area_id=None,
        target_area_id=None,
        error=None,
    ):
        if not self.enabled:
            return

        try:
            observation = np.asarray(map_state)
            topology_array = np.asarray(topology, dtype=np.int8)
            record = {
                "step": int(step_number),
                "mode": mode.value,
                "previous_mode": previous_mode.value,
                "transition_reasons": list(transition_reasons),
                "my_position": _position(my_position),
                "enemy_position": _position(enemy_position),
                "last_seen_position": _position(last_seen_position),
                "last_seen_step": last_seen_step,
                "target": _position(target),
                "path": [_position(position) for position in (path or [])],
                "action": {"move": move.name, "steps": int(move_steps)},
                "duration_seconds": float(duration_seconds),
                "area_fingerprint": (
                    None if area_analysis is None else area_analysis.fingerprint
                ),
                "area_count": (
                    0 if area_analysis is None else len(area_analysis.areas)
                ),
                "area_analysis_seconds": (
                    None if area_analysis is None
                    else float(area_analysis.analysis_seconds)
                ),
                "area_cache_hit": bool(area_cache_hit),
                "area_error": (
                    None if area_analysis is None else area_analysis.error
                ),
                "current_area_id": current_area_id,
                "target_area_id": target_area_id,
                "observation": observation.tolist(),
                "visible_mask": (observation == 0).astype(np.int8).tolist(),
                "topology": topology_array.tolist(),
                "error": error,
            }
            with self.log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception:
            # Diagnostics must never alter or terminate the agent's decisions.
            return


def _position(value):
    if value is None:
        return None
    return [int(value[0]), int(value[1])]


def _area_symbol(area_id):
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if area_id is None or area_id < 0 or area_id >= len(alphabet):
        return "?"
    return alphabet[area_id]


def _render_area_grid(observation, cell_to_area, symbols):
    rows, columns = observation.shape
    column_header = "    " + "".join(
        f"{column:02d}".center(3) + " "
        for column in range(columns)
    )
    border = "   +" + "+".join("---" for _ in range(columns)) + "+"
    rendered = [column_header.rstrip(), border]

    for row in range(rows):
        cells = []
        for column in range(columns):
            position = (row, column)
            if observation[position] == 1:
                cells.append("###")
                continue
            area_id = cell_to_area.get(position)
            cells.append(symbols.get(area_id, "?").center(3))
        rendered.append(f"{row:02d} |" + "|".join(cells) + "|")
        rendered.append(border)

    return rendered
