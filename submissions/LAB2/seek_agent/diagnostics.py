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

    def write_area_analysis(
        self,
        analysis,
        map_state,
        cache_hit,
        reachable_cells=None,
    ):
        """Write a static, human-readable representation of the map areas."""
        if not self.enabled or analysis is None:
            return

        try:
            observation = np.asarray(map_state)
            reachability = _reachability_summary(
                analysis,
                reachable_cells,
            )
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
                f"reachable_cell_count: {reachability['reachable_cell_count']}",
                f"reachable_area_ids: {reachability['reachable_area_ids']}",
                f"excluded_area_ids: {reachability['excluded_area_ids']}",
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
                            "  reachable_from_spawn: "
                            f"{'yes' if area.area_id in reachability['reachable_area_ids'] else 'no'}"
                        ),
                        (
                            "  excluded_from_search: "
                            f"{'yes' if area.area_id in reachability['excluded_area_ids'] else 'no'}"
                        ),
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
        visible_cells=(),
        reachable_cells=(),
        ghost_belief=None,
        search_decision=None,
        investigation_decision=None,
        error=None,
    ):
        if not self.enabled:
            return

        try:
            observation = np.asarray(map_state)
            topology_array = np.asarray(topology, dtype=np.int8)
            visible_cells = tuple(sorted(_position(cell) for cell in visible_cells))
            visible_mask = np.zeros(observation.shape, dtype=np.int8)
            for row, column in visible_cells:
                visible_mask[row, column] = 1
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
                "visible_mask": visible_mask.tolist(),
                "visible_cells": [list(cell) for cell in visible_cells],
                "topology": topology_array.tolist(),
                "reachability": _reachability_summary(
                    area_analysis,
                    reachable_cells,
                    ghost_belief,
                ),
                "ghost_belief": _belief_summary(
                    ghost_belief,
                    area_analysis,
                    step_number,
                    reachable_cells,
                ),
                "search": _search_summary(search_decision),
                "investigation": _investigation_summary(
                    investigation_decision,
                ),
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


def _reachability_summary(area_analysis, reachable_cells, belief=None):
    """Return stable match-reachability diagnostics without changing policy."""
    reachable = tuple(sorted(
        (int(cell[0]), int(cell[1]))
        for cell in (reachable_cells or ())
    ))
    reachable_set = frozenset(reachable)
    if area_analysis is None:
        reachable_area_ids = ()
        excluded_area_ids = ()
    else:
        reachable_area_ids = tuple(sorted(
            area.area_id
            for area in area_analysis.areas
            if any(cell in reachable_set for cell in area.cells)
        ))
        reachable_area_set = frozenset(reachable_area_ids)
        excluded_area_ids = tuple(sorted(
            area.area_id
            for area in area_analysis.areas
            if area.area_id not in reachable_area_set
        ))
    excluded_candidates = (
        ()
        if belief is None
        else tuple(belief.sorted_excluded_unreachable_positions())
    )
    return {
        "reachable_cell_count": len(reachable),
        "reachable_cells": [_position(cell) for cell in reachable],
        "reachable_area_ids": list(reachable_area_ids),
        "excluded_area_ids": list(excluded_area_ids),
        "excluded_initial_ghost_candidate_count": len(excluded_candidates),
        "excluded_initial_ghost_candidate_cells": [
            _position(cell) for cell in excluded_candidates
        ],
    }


def _belief_summary(belief, area_analysis, step_number, reachable_cells=()):
    """Return a JSON-ready snapshot without making diagnostics authoritative."""
    if belief is None:
        return None

    possible = tuple(belief.sorted_possible_positions())
    summary = {
        "possible_count": len(possible),
        "possible_cells": [_position(cell) for cell in possible],
        "areas": [],
    }
    if area_analysis is None:
        return summary

    reachable_set = frozenset(
        (int(cell[0]), int(cell[1]))
        for cell in (reachable_cells or ())
    )
    excluded_candidates = frozenset(
        belief.sorted_excluded_unreachable_positions()
    )
    for area in area_analysis.areas:
        cells = tuple(sorted(area.cells))
        possible_cells = tuple(belief.possible_positions_in_area(area))
        excluded_cells = tuple(sorted(excluded_candidates & area.cells))
        never_observed = tuple(belief.never_observed_in_area(area))
        ages = [
            belief.last_observed_age(cell, step_number)
            for cell in cells
        ]
        known_ages = [age for age in ages if age is not None]
        summary["areas"].append({
            "area_id": area.area_id,
            "reachable_from_spawn": any(cell in reachable_set for cell in cells),
            "risk": float(belief.risk_fraction_for_area(area)),
            "possible_count": len(possible_cells),
            "possible_cells": [_position(cell) for cell in possible_cells],
            "excluded_unreachable_initial_candidate_count": len(excluded_cells),
            "excluded_unreachable_initial_candidate_cells": [
                _position(cell) for cell in excluded_cells
            ],
            "never_observed_count": len(never_observed),
            "never_observed_cells": [_position(cell) for cell in never_observed],
            "fresh_count": sum(age == 0 for age in known_ages),
            "stale_count": sum(age > 0 for age in known_ages),
            "oldest_observation_age": (
                None if not known_ages else max(known_ages)
            ),
        })
    return summary


def _search_summary(decision):
    """Make the absence of a Search action explicit for chase/investigate."""
    if decision is None:
        return {
            "decision_made": False,
            "phase": None,
            "current_area_id": None,
            "target_area_id": None,
            "planned_area_order": [],
            "reachable_area_ids": [],
            "excluded_area_ids": [],
            "entry": None,
            "exit": None,
            "route": [],
            "actions": [],
            "chosen_action": None,
            "required_cells": [],
            "required_count": 0,
            "covered_cells": [],
            "covered_count": 0,
            "completed_this_step": None,
            "completed_area_ids": [],
            "replan_reason": None,
            "planning_seconds": None,
            "exact": None,
            "fallback": False,
            "fallback_scope": None,
            "fallback_meaning": None,
        }

    fallback = _fallback_fields(decision)
    return {
        "decision_made": True,
        "phase": decision.phase.value,
        "current_area_id": decision.current_area_id,
        "target_area_id": decision.target_area_id,
        "planned_area_order": list(decision.planned_area_order),
        "reachable_area_ids": list(decision.reachable_area_ids),
        "excluded_area_ids": list(decision.excluded_area_ids),
        "entry": _position(decision.entry),
        "exit": _position(decision.exit),
        "route": [_position(cell) for cell in decision.route],
        "actions": [
            {"move": move.name, "steps": int(steps)}
            for move, steps in decision.actions
        ],
        "chosen_action": {
            "move": decision.chosen_action[0].name,
            "steps": int(decision.chosen_action[1]),
        },
        "required_cells": [_position(cell) for cell in sorted(decision.required_cells)],
        "required_count": len(decision.required_cells),
        "covered_cells": [_position(cell) for cell in sorted(decision.covered_cells)],
        "covered_count": len(decision.covered_cells),
        "completed_this_step": decision.completed_this_step,
        "completed_area_ids": sorted(decision.completed_area_ids),
        "replan_reason": decision.replan_reason,
        "planning_seconds": float(decision.planning_seconds),
        "exact": bool(decision.exact),
        "local_route_fallback": bool(decision.local_route_fallback),
        **fallback,
    }


def _fallback_fields(decision):
    """Explain the narrow scope of the planner's fallback flag."""
    safety_reasons = {
        "position_outside_areas",
        "unreachable_target_skipped",
        "no_reachable_area",
    }
    if decision.replan_reason in safety_reasons:
        return {
            "fallback": True,
            "fallback_scope": "search_safety",
            "fallback_meaning": (
                "safe_stay_no_target"
                if decision.replan_reason == "no_reachable_area"
                else decision.replan_reason
            ),
        }
    if decision.fallback:
        return {
            "fallback": True,
            "fallback_scope": "global_priority",
            "fallback_meaning": "deadline_or_incomplete_global_profile",
        }
    return {
        "fallback": False,
        "fallback_scope": None,
        "fallback_meaning": None,
    }


def _investigation_summary(decision):
    """Return the evidence and scoring behind INVESTIGATING actions."""
    if decision is None:
        return {
            "decision_made": False,
            "phase": None,
            "last_seen_position": None,
            "arrival_step": None,
            "investigation_turn": None,
            "turn_limit": None,
            "turns_remaining_after_action": None,
            "possible_count": 0,
            "possible_cells": [],
            "considered_actions": [],
            "chosen_action": None,
            "chosen_endpoint": None,
            "target": None,
            "route": [],
            "selection_reason": None,
            "finished_reason": None,
        }

    considered = [
        {
            "move": score.move.name,
            "steps": int(score.steps),
            "endpoint": _position(score.endpoint),
            "confirmable_count": len(score.confirmable_positions),
            "confirmable_cells": [
                _position(cell)
                for cell in sorted(score.confirmable_positions)
            ],
            "nearest_candidate_distance": score.nearest_candidate_distance,
        }
        for score in decision.considered_actions
    ]
    if decision.finished_reason is not None:
        selection_reason = decision.finished_reason
    elif decision.phase.value == "transit":
        selection_reason = "shortest_route_to_last_seen"
    elif len(decision.possible_positions) == 1:
        selection_reason = "single_candidate_pursuit"
    elif any(item["confirmable_count"] for item in considered):
        selection_reason = "maximum_current_candidate_coverage"
    else:
        selection_reason = "nearest_current_candidate"

    turn = decision.investigation_turn
    remaining = (
        None
        if turn is None
        else max(0, int(decision.turn_limit) - int(turn))
    )
    return {
        "decision_made": True,
        "phase": decision.phase.value,
        "last_seen_position": _position(decision.last_seen_position),
        "arrival_step": decision.arrival_step,
        "investigation_turn": turn,
        "turn_limit": int(decision.turn_limit),
        "turns_remaining_after_action": remaining,
        "possible_count": len(decision.possible_positions),
        "possible_cells": [
            _position(cell) for cell in sorted(decision.possible_positions)
        ],
        "considered_actions": considered,
        "chosen_action": {
            "move": decision.chosen_action[0].name,
            "steps": int(decision.chosen_action[1]),
        },
        "chosen_endpoint": _position(decision.chosen_endpoint),
        "target": _position(decision.target),
        "route": [_position(cell) for cell in decision.route],
        "selection_reason": selection_reason,
        "finished_reason": decision.finished_reason,
    }


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
