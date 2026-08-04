"""Optional structured diagnostics for the seek agent."""

import json
import os
from pathlib import Path

import numpy as np


_ENABLED_VALUES = {"1", "true", "yes", "on"}


class SeekDiagnostics:
    """Write enough state to reconstruct each seeker decision.

    Diagnostics are disabled by default and must never affect agent behavior.
    Set ``PACMAN_SEEK_DIAGNOSTICS=1`` to enable the default log.
    """

    def __init__(self, enabled=None, log_path=None):
        if enabled is None:
            enabled = (
                os.getenv("PACMAN_SEEK_DIAGNOSTICS", "").strip().lower()
                in _ENABLED_VALUES
            )

        self.enabled = bool(enabled)
        self.log_path = Path(log_path) if log_path else (
            Path(__file__).resolve().parent.parent / "debug" / "seek-agent.jsonl"
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
            self.log_path.write_text("", encoding="utf-8")
        except Exception:
            self.enabled = False

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
