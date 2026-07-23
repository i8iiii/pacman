"""Best-effort JSON-lines diagnostics for Hide phases."""

import json
from pathlib import Path

import numpy as np


DIAGNOSTICS_ENABLED = True


class JsonlDiagnostics:
    """Append structured diagnostics without affecting the agent on failure."""

    def __init__(self, log_path, enabled=None):
        self.log_path = Path(log_path)
        self.enabled = DIAGNOSTICS_ENABLED if enabled is None else bool(enabled)
        self.disabled_reason = None

    def reset(self):
        if not self.enabled:
            return False

        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("", encoding="utf-8")
            return True
        except Exception as error:
            self.enabled = False
            self.disabled_reason = f"{type(error).__name__}: {error}"
            return False

    def write(self, event, **fields):
        if not self.enabled:
            return False

        record = {"event": event, **fields}
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, sort_keys=True) + "\n\n")
            return True
        except Exception as error:
            self.enabled = False
            self.disabled_reason = f"{type(error).__name__}: {error}"
            return False


class MapDiagnostics:
    """Write synchronized map snapshots without affecting Hide on failure."""

    CELL_SYMBOLS = {
        -1: "?",
        0: ".",
        1: "#",
    }

    def __init__(self, text_path, jsonl_path, enabled=None):
        self.text_path = Path(text_path)
        self.jsonl_path = Path(jsonl_path)
        self.enabled = DIAGNOSTICS_ENABLED if enabled is None else bool(enabled)
        self.disabled_reason = None

    def reset(self):
        if not self.enabled:
            return False

        try:
            for path in (self.text_path, self.jsonl_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")
            return True
        except Exception as error:
            self._disable(error)
            return False

    def write_snapshot(
        self,
        step_number,
        current_map,
        hideout_candidates=(),
        selected_hideout=None,
        compromised_hideouts=(),
        pacman_belief=(),
        road_visibility=(),
        road_excluded_hideouts=(),
    ):
        if not self.enabled:
            return False

        try:
            rows, cols = current_map.shape
            observed_open_cells = int(np.count_nonzero(current_map == 0))
            known_wall_cells = int(np.count_nonzero(current_map == 1))
            unseen_cells = int(np.count_nonzero(current_map == -1))
            total_cells = int(current_map.size)
            candidate_records = [
                (
                    candidate.to_log_record()
                    if hasattr(candidate, "to_log_record")
                    else dict(candidate)
                )
                for candidate in hideout_candidates
            ]
            selected_position = (
                None
                if selected_hideout is None
                else list(selected_hideout)
            )
            compromised_positions = [
                list(position)
                for position in sorted(
                    {tuple(position) for position in compromised_hideouts}
                )
            ]
            belief_positions = [
                list(position)
                for position in sorted(
                    {tuple(position) for position in pacman_belief}
                )
            ]
            road_records = [
                record.to_log_record()
                for record in road_visibility
            ]
            road_excluded_positions = [
                list(position)
                for position in sorted(
                    {
                        tuple(
                            candidate.position
                            if hasattr(candidate, "position")
                            else candidate
                        )
                        for candidate in road_excluded_hideouts
                    }
                )
            ]

            human_snapshot = self._human_snapshot(
                step_number,
                current_map,
                observed_open_cells,
                known_wall_cells,
                unseen_cells,
                total_cells,
                candidate_records,
                selected_position,
                compromised_positions,
                belief_positions,
                road_records,
                road_excluded_positions,
            )
            machine_snapshot = {
                "event": "map_snapshot",
                "step_number": step_number,
                "shape": [rows, cols],
                "observed_open_cells": observed_open_cells,
                "known_wall_cells": known_wall_cells,
                "unseen_cells": unseen_cells,
                "total_cells": total_cells,
                "map": current_map.astype(int).tolist(),
                "hideout_candidates": candidate_records,
                "selected_hideout": selected_position,
                "compromised_hideouts": compromised_positions,
                "pacman_belief": belief_positions,
                "road_visibility": road_records,
                "road_excluded_hideouts": road_excluded_positions,
            }

            with self.text_path.open("a", encoding="utf-8") as text_file:
                text_file.write(human_snapshot)
            with self.jsonl_path.open("a", encoding="utf-8") as jsonl_file:
                jsonl_file.write(
                    json.dumps(machine_snapshot, separators=(",", ":")) + "\n"
                )
            return True
        except Exception as error:
            self._disable(error)
            return False

    def _human_snapshot(
        self,
        step_number,
        current_map,
        observed_open_cells,
        known_wall_cells,
        unseen_cells,
        total_cells,
        hideout_candidates,
        selected_hideout,
        compromised_hideouts,
        pacman_belief,
        road_visibility,
        road_excluded_hideouts,
    ):
        rows, cols = current_map.shape
        lines = [
            (
                f"=== step {step_number} | shape {rows}x{cols} "
                f"| observed-open {observed_open_cells} "
                f"| walls {known_wall_cells} "
                f"| unseen {unseen_cells} "
                f"| total {total_cells} ==="
            ),
            "    " + " ".join(f"{column:02d}" for column in range(cols)),
        ]

        for row_index, row in enumerate(current_map):
            symbols = "  ".join(
                self.CELL_SYMBOLS.get(int(cell), str(int(cell))) for cell in row
            )
            lines.append(f"{row_index:02d}  {symbols}")

        lines.extend(
            [
                "",
                self._hideout_lines(hideout_candidates),
                self._selected_hideout_line(selected_hideout),
                self._position_list("Compromised hideouts", compromised_hideouts),
                self._position_list("Pacman belief", pacman_belief),
                self._road_visibility_lines(road_visibility),
                self._position_list(
                    "Road-excluded hideouts",
                    road_excluded_hideouts,
                ),
            ]
        )
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def _position_list(label, positions):
        rendered = ", ".join(
            f"({position[0]}, {position[1]})" for position in positions
        )
        return f"{label} ({len(positions)}): {rendered or 'none'}"

    @staticmethod
    def _selected_hideout_line(selected_hideout):
        if selected_hideout is None:
            return "Selected hideout: none"

        row, column = selected_hideout
        return f"Selected hideout: ({row}, {column})"

    @staticmethod
    def _hideout_lines(candidates):
        lines = [f"Hideout candidates ({len(candidates)}):"]
        for candidate in candidates:
            position = candidate["position"]
            entrance = candidate["entrance"]
            lines.append(
                "  "
                f"({position[0]}, {position[1]}) "
                f"class={candidate['kind']} "
                f"entrance={None if entrance is None else tuple(entrance)} "
                f"hidden={candidate['entrance_hidden']} "
                f"gates={candidate['gate_depth']} "
                f"inspect={candidate['inspection_depth']} "
                f"footprint={candidate['visibility_footprint']} "
                f"backtrack={candidate['must_backtrack']} "
                f"spawn={candidate['spawn_discovery_distance']}"
            )
        if len(lines) == 1:
            lines.append("  none")
        return "\n".join(lines)

    @classmethod
    def _road_visibility_lines(cls, records):
        approach_records = [
            record
            for record in records
            if record["is_approach"]
        ]
        lines = [
            f"Approach road visibility ({len(approach_records)}):"
        ]
        for record in approach_records:
            lines.append(
                "  "
                f"road {record['road_id']} "
                f"visible ({len(record['visible_cells'])}): "
                + ", ".join(
                    f"({position[0]}, {position[1]})"
                    for position in record["visible_cells"]
                )
            )
        if len(lines) == 1:
            lines.append("  none")
        return "\n".join(lines)

    def _disable(self, error):
        self.enabled = False
        self.disabled_reason = f"{type(error).__name__}: {error}"
