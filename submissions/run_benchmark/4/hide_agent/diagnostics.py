"""Best-effort JSON-lines diagnostics for Hide phases."""

import json
from pathlib import Path

import numpy as np


DIAGNOSTICS_ENABLED = False


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
        road_cycle=None,
        active_road_stage=None,
        active_road_ids=(),
        active_road_excluded_cells=(),
        migration_state=None,
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
            cycle_record = (
                None
                if road_cycle is None
                else road_cycle.to_log_record()
            )
            stage_record = (
                None
                if active_road_stage is None
                else active_road_stage.to_log_record()
            )
            active_ids = [
                int(road_id) for road_id in active_road_ids
            ]
            active_excluded_positions = [
                list(position)
                for position in sorted(
                    {
                        tuple(position)
                        for position in active_road_excluded_cells
                    }
                )
            ]
            migration_record = (
                None
                if migration_state is None
                else dict(migration_state)
            )

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
                cycle_record,
                stage_record,
                active_ids,
                active_excluded_positions,
                migration_record,
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
                "road_cycle": cycle_record,
                "active_road_stage": stage_record,
                "active_road_ids": active_ids,
                "active_road_excluded_cells": (
                    active_excluded_positions
                ),
                "migration": migration_record,
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
        road_cycle,
        active_road_stage,
        active_road_ids,
        active_road_excluded_cells,
        migration_state,
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
                self._road_cycle_stage_line(active_road_stage),
                self._road_ids_line(active_road_ids),
                self._position_list(
                    "Active road excluded cells",
                    active_road_excluded_cells,
                ),
                self._migration_line(migration_state),
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
    def _road_cycle_stage_line(stage):
        if stage is None:
            return "Road cycle stage: none"
        return (
            "Road cycle stage: "
            f"{stage['index']} {stage['label']}"
        )

    @staticmethod
    def _road_ids_line(road_ids):
        rendered = ", ".join(str(road_id) for road_id in road_ids)
        return (
            f"Active road IDs ({len(road_ids)}): "
            f"{rendered or 'none'}"
        )

    @staticmethod
    def _migration_line(state):
        if state is None:
            return "Hideout migration: unavailable"
        waypoint = (
            "none"
            if state["waypoint"] is None
            else (
                f"({state['waypoint'][0]}, "
                f"{state['waypoint'][1]})"
            )
        )
        return (
            "Hideout migration: "
            f"phase={state['phase']} "
            f"middle_holds={state['middle_hold_turns']} "
            f"opposite_holds={state['opposite_hold_turns']} "
            f"waypoint={waypoint} "
            f"spawn={state['spawn_band']} "
            f"destination={state['destination_band']} "
            f"junction_distance={state['junction_distance']} "
            f"blocked={state['blocked_reason']}"
        )

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
