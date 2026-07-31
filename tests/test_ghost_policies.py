"""Tests for Ghost Policy classes -- random, greedy, and factory."""

import sys
from pathlib import Path

PACMAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACMAN_ROOT / "src"))

import unittest
import numpy as np
from environment import Move

from rl.ghost_policies import (
    RandomGhostPolicy,
    GreedyGhostPolicy,
    get_ghost_policy,
)


class TestGhostPolicies(unittest.TestCase):
    """Verify ghost policies return valid Move enums and behave correctly."""

    def setUp(self):
        # Simple 5x5 map with walls on the border and a central corridor
        # Layout:
        #   #####
        #   #.G.#
        #   #.#.#
        #   #.P.#
        #   #####
        # Walls=1, Empty=0
        self.map_state = np.ones((5, 5), dtype=int)  # all walls
        for r in range(1, 4):
            for c in [1, 3]:
                self.map_state[r, c] = 0  # open corridors
        self.map_state[1, 2] = 0  # ghost start area
        self.map_state[2, 1] = 0
        self.map_state[2, 3] = 0
        self.map_state[3, 2] = 0  # pacman start area

    # ------------------------------------------------------------------
    # RandomGhostPolicy
    # ------------------------------------------------------------------

    def test_random_policy_returns_move_enum(self):
        policy = RandomGhostPolicy()
        move = policy.step(
            map_state=self.map_state,
            my_position=(1, 2),
            enemy_position=(3, 2),
            step_number=0,
        )
        self.assertIsInstance(move, Move)

    def test_random_policy_only_returns_valid_moves(self):
        policy = RandomGhostPolicy()
        # Corner position where some moves go into walls
        pos = (1, 1)
        valid_deltas = set()
        for m in Move:
            dr, dc = m.value
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < 5 and 0 <= nc < 5 and self.map_state[nr, nc] != 1:
                valid_deltas.add(m)
        # STAY is always valid
        self.assertIn(Move.STAY, valid_deltas)

        # Run many times and ensure we only see valid moves
        for _ in range(200):
            move = policy.step(self.map_state, pos, (3, 2), 0)
            self.assertIn(move, valid_deltas,
                          f"Got invalid move {move} from position {pos}")

    def test_random_policy_surrounded_by_walls_returns_stay(self):
        policy = RandomGhostPolicy()
        # Create a map where position is completely surrounded by walls
        trapped_map = np.ones((3, 3), dtype=int)
        trapped_map[1, 1] = 0  # only this cell is open
        move = policy.step(trapped_map, (1, 1), None, 0)
        self.assertEqual(move, Move.STAY)

    # ------------------------------------------------------------------
    # GreedyGhostPolicy
    # ------------------------------------------------------------------

    def test_greedy_moves_away_from_enemy(self):
        policy = GreedyGhostPolicy()
        # Ghost at (1,2), Pacman at (3,2) -- ghost should move UP toward (0,2)
        # but UP is a wall, so valid moves from (1,2) are:
        #   DOWN -> (2,2) wall? map[2,2] = 1 (wall), so invalid
        #   LEFT -> (1,1) valid, distance to (3,2) = |1-3| + |1-2| = 3
        #   RIGHT -> (1,3) valid, distance to (3,2) = |1-3| + |3-2| = 3
        #   STAY -> (1,2) valid, distance to (3,2) = |1-3| + |2-2| = 2
        # So LEFT and RIGHT are equally farthest at distance 3.
        # LEFT and RIGHT are both valid and equally good.

        # Let us use a simpler setup: ghost at (3,1), pacman at (3,3)
        # Valid moves from (3,1): UP (2,1)-valid, DOWN (4,1)-wall, LEFT (3,0)-wall, RIGHT (3,2)-valid, STAY
        # Distance from (2,1) to (3,3) = 1+2 = 3
        # Distance from (3,1) to (3,3) = 0+2 = 2 (STAY)
        # Distance from (3,2) to (3,3) = 0+1 = 1 (RIGHT)
        # So UP maximizes distance -> should choose UP

        ghost_pos = (3, 1)
        enemy_pos = (3, 3)
        # Run multiple times; should always pick UP
        for _ in range(50):
            move = policy.step(self.map_state, ghost_pos, enemy_pos, 0)
            self.assertEqual(move, Move.UP,
                             f"Expected UP (away from enemy at {enemy_pos}), got {move}")

    def test_greedy_handles_enemy_none(self):
        policy = GreedyGhostPolicy()
        # When enemy is None, falls through to random behavior
        # Should still return a valid Move
        for _ in range(50):
            move = policy.step(self.map_state, (1, 2), None, 0)
            self.assertIsInstance(move, Move)
            # Should be a valid move
            self.assertIn(move, [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY])

    def test_greedy_no_valid_moves_returns_stay(self):
        policy = GreedyGhostPolicy()
        trapped_map = np.ones((3, 3), dtype=int)
        trapped_map[1, 1] = 0
        move = policy.step(trapped_map, (1, 1), (0, 0), 0)
        self.assertEqual(move, Move.STAY)

    def test_greedy_enemy_none_falls_through_random_valid(self):
        policy = GreedyGhostPolicy()
        # Position with only one non-stay valid move
        pos = (1, 1)
        # Valid from (1,1): DOWN -> (2,1)-valid, RIGHT -> (1,2)-valid, STAY
        # UP and LEFT are walls
        valid_set = {Move.DOWN, Move.RIGHT, Move.STAY}
        for _ in range(100):
            move = policy.step(self.map_state, pos, None, 0)
            self.assertIn(move, valid_set)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def test_factory_returns_random(self):
        policy = get_ghost_policy("random")
        self.assertIsInstance(policy, RandomGhostPolicy)

    def test_factory_returns_greedy(self):
        policy = get_ghost_policy("greedy")
        self.assertIsInstance(policy, GreedyGhostPolicy)

    def test_factory_minimax_falls_back_to_greedy(self):
        policy = get_ghost_policy("minimax")
        self.assertIsInstance(policy, GreedyGhostPolicy)

    def test_factory_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            get_ghost_policy("nonexistent")

    def test_factory_case_sensitive(self):
        with self.assertRaises(ValueError):
            get_ghost_policy("Random")
        with self.assertRaises(ValueError):
            get_ghost_policy("GREEDY")


if __name__ == "__main__":
    unittest.main()
