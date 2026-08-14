"""LAB2 submission entry points for the new seeker and existing hider."""

import sys
from pathlib import Path


src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from hide_agent.controller import HideController
from seek_agent.controller import SeekController


class PacmanAgent(BasePacmanAgent):
    """Thin framework adapter for the new chasing-first seek controller."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Seek-Agent"
        self._controller = SeekController(
            pacman_speed=kwargs.get("pacman_speed", 2),
            diagnostics_enabled=kwargs.get("diagnostics_enabled"),
            diagnostic_log_path=kwargs.get("seek_log_path"),
            diagnostic_area_path=kwargs.get("seek_area_path"),
        )

    def step(self, map_state, my_position, enemy_position, step_number):
        return self._controller.step(
            map_state,
            my_position,
            enemy_position,
            step_number,
        )


class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Hide-Agent"
        self._controller = HideController(
            log_path=kwargs.get("log_path"),
            map_text_path=kwargs.get("map_text_path"),
            map_jsonl_path=kwargs.get("map_jsonl_path"),
            diagnostics_enabled=kwargs.get("diagnostics_enabled"),
            pacman_speed=kwargs.get("pacman_speed", 2),
            capture_distance=kwargs.get("capture_distance", 2),
            observation_radius=kwargs.get("observation_radius", 5),
        )

    def step(self, map_state, my_position, enemy_position, step_number):
        return self._controller.step(
            map_state,
            my_position,
            enemy_position,
            step_number,
        )
