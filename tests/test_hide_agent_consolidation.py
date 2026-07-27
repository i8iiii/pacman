import sys
import unittest
from pathlib import Path


PACMAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACMAN_ROOT / "src"))
sys.path.insert(0, str(PACMAN_ROOT / "submissions" / "LAB2"))

from environment import Environment, Move


class HideAgentConsolidationTests(unittest.TestCase):
    def test_merged_domain_apis_are_importable(self):
        from hide_agent.spatial import move_between, vertical_band
        from hide_agent.concealment import scan_campsites, scan_hideouts
        from hide_agent.relocation import build_road_cycle, migration_direction
        from hide_agent.evasion import (
            choose_visible_junction_escape,
            choose_visible_mobile_escape,
        )
        from hide_agent.belief import PacmanBeliefTracker, PursuitTracker

        self.assertEqual(move_between((1, 1), (1, 2)), Move.RIGHT)
        self.assertEqual(vertical_band((1, 1), 21), "top")
        self.assertTrue(callable(scan_campsites))
        self.assertTrue(callable(scan_hideouts))
        self.assertTrue(callable(build_road_cycle))
        self.assertTrue(callable(migration_direction))
        self.assertTrue(callable(choose_visible_junction_escape))
        self.assertTrue(callable(choose_visible_mobile_escape))
        self.assertIsInstance(PacmanBeliefTracker(), PacmanBeliefTracker)
        self.assertIsInstance(PursuitTracker(), PursuitTracker)

    def test_ghost_agent_returns_a_valid_move_without_debug_output(self):
        from agent import GhostAgent

        debug_dir = PACMAN_ROOT / "submissions" / "LAB2" / "debug"
        agent = GhostAgent()
        environment = Environment(
            pacman_speed=2,
            capture_distance_threshold=2,
        )
        observation, position, enemy = environment.get_observation("ghost", 5, 5)

        action = agent.step(observation, position, enemy, 1)

        self.assertIsInstance(action, Move)
        self.assertFalse(debug_dir.exists())


if __name__ == "__main__":
    unittest.main()
