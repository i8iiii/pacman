"""
PacmanAgent: 6-ply Minimax with Alpha-Beta pruning, topology-aware ghost model.

The ghost model mirrors the reference/5 hide agent's scoring:
    distance * 10  +  exit_count * 3

Line-of-sight escape incentive and dead-end preferences are also modelled.
"""

import sys
from collections import deque
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move
import numpy as np

DIRS = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
INF = 10 ** 9


class PacmanAgent(BasePacmanAgent):
    """
    6-ply Minimax seeker with Alpha-Beta pruning.

    The Ghost is modelled using the same topology-inspired scoring that
    the reference/5 hide agent actually uses, so the minimax tree
    anticipates the real opponent's decisions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "TopoMinimax6"
        self._valid = None
        self._bfs_cache = {}
        self._pair_dist = {}
        self._exit_cache = {}

    # ---------- precomputation ------------------------------------------

    def _precompute(self, map_state):
        h, w = map_state.shape
        self._valid = {
            (r, c) for r in range(h) for c in range(w)
            if map_state[r, c] == 0
        }
        for pos in self._valid:
            self._exit_cache[pos] = sum(
                1 for mv in DIRS
                if (pos[0] + mv.value[0], pos[1] + mv.value[1]) in self._valid
            )

    # ---------- step entry point ----------------------------------------

    def step(self, map_state, my_position, enemy_position, step_number):
        if enemy_position is None:
            return (Move.STAY, 1)

        my_pos = (int(my_position[0]), int(my_position[1]))
        enemy_pos = (int(enemy_position[0]), int(enemy_position[1]))

        if self._valid is None:
            self._precompute(map_state)

        if self._manhattan(my_pos, enemy_pos) < 2:
            return (Move.STAY, 1)

        self._bfs_cache.clear()
        self._pair_dist.clear()

        return self._minimax_root(my_pos, enemy_pos)

    # ---------- minimax search ------------------------------------------

    def _minimax_root(self, pac_pos, ghost_pos):
        """Pacman chooses the root action that maximises the minimax score."""
        actions = self._pacman_actions(pac_pos)
        actions.sort(key=lambda a: self._bfs_dist(
            self._apply_action(pac_pos, a), ghost_pos))

        best_score, best_action = -INF, (Move.STAY, 1)
        alpha, beta = -INF, INF

        for action in actions:
            new_pac = self._apply_action(pac_pos, action)
            score = self._min_node(new_pac, ghost_pos, 6, alpha, beta)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, score)

        return best_action

    def _min_node(self, pac_pos, ghost_pos, depth, alpha, beta):
        """Ghost turn -- minimise Pacman's score."""
        if self._manhattan(pac_pos, ghost_pos) < 2:
            return 100000 + depth

        if depth == 0:
            return self._evaluate(pac_pos, ghost_pos)

        ghost_moves = self._scored_ghost_moves(pac_pos, ghost_pos)
        best = INF

        for new_ghost, _ in ghost_moves:
            val = self._max_node(pac_pos, new_ghost, depth - 1, alpha, beta)
            if val < best:
                best = val
            if best <= alpha:
                return best
            beta = min(beta, best)

        if best == INF:
            return self._evaluate(pac_pos, ghost_pos)
        return best

    def _max_node(self, pac_pos, ghost_pos, depth, alpha, beta):
        """Pacman turn -- maximise score."""
        if self._manhattan(pac_pos, ghost_pos) < 2:
            return 100000 + depth

        if depth == 0:
            return self._evaluate(pac_pos, ghost_pos)

        actions = self._pacman_actions(pac_pos)
        if not actions:
            return self._evaluate(pac_pos, ghost_pos)

        actions.sort(key=lambda a: self._bfs_dist(
            self._apply_action(pac_pos, a), ghost_pos))

        best = -INF
        for action in actions:
            new_pac = self._apply_action(pac_pos, action)
            val = self._min_node(new_pac, ghost_pos, depth - 1, alpha, beta)
            if val > best:
                best = val
            if best >= beta:
                return best
            alpha = max(alpha, best)

        return best

    # ---------- ghost move scoring (mirrors reference/5 GhostAgent) -----

    def _scored_ghost_moves(self, pac_pos, ghost_pos):
        """
        Score each ghost move identically to reference/5's GhostAgent:
            distance * 10  +  exit_count * 3

        An extra bonus is added for perpendicular turns when ghost and
        pacman are aligned on a clear row or column (matching the
        _aligned_axis / _fastest_turn_move escape logic).
        """
        moves = []
        aligned = self._aligned_with_pacman(ghost_pos, pac_pos)
        perpendicular = self._perpendicular_to(ghost_pos, pac_pos) if aligned else set()

        for mv in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY):
            if mv == Move.STAY:
                new_pos = ghost_pos
            else:
                new_pos = (ghost_pos[0] + mv.value[0],
                           ghost_pos[1] + mv.value[1])
                if new_pos not in self._valid:
                    continue

            dist = self._bfs_dist(pac_pos, new_pos)
            exits = self._exit_cache.get(new_pos, 0)
            score = dist * 10 + exits * 3

            if mv in perpendicular:
                score += 30

            moves.append((new_pos, score))

        moves.sort(key=lambda x: x[1], reverse=True)
        return moves

    def _aligned_with_pacman(self, ghost_pos, pac_pos):
        if ghost_pos[0] == pac_pos[0]:
            left, right = sorted((ghost_pos[1], pac_pos[1]))
            return all((ghost_pos[0], c) in self._valid
                       for c in range(left + 1, right))
        if ghost_pos[1] == pac_pos[1]:
            top, bottom = sorted((ghost_pos[0], pac_pos[0]))
            return all((r, ghost_pos[1]) in self._valid
                       for r in range(top + 1, bottom))
        return False

    def _perpendicular_to(self, ghost_pos, pac_pos):
        if ghost_pos[0] == pac_pos[0]:
            return {Move.UP, Move.DOWN}
        return {Move.LEFT, Move.RIGHT}

    # ---------- evaluation ----------------------------------------------

    def _evaluate(self, pac_pos, ghost_pos):
        """
        Pacman-centric evaluation (higher = better for Pacman).

        Mirrors the reference/5 ghost's own scoring, which the ghost
        tries to maximise.  Pacman therefore tries to minimise it.
        """
        dist = self._bfs_dist(pac_pos, ghost_pos)
        exits = self._exit_cache.get(ghost_pos, 0)
        return -(dist * 10 + exits * 3)

    # ---------- BFS distance helpers ------------------------------------

    def _bfs_dist(self, a, b):
        if a == b:
            return 0
        if a not in self._valid or b not in self._valid:
            return INF
        key = (a, b)
        if key not in self._pair_dist:
            self._pair_dist[key] = self._bfs_compute(a).get(b, INF)
        return self._pair_dist[key]

    def _bfs_compute(self, start):
        if start not in self._bfs_cache:
            dist = {start: 0}
            q = deque([start])
            while q:
                cur = q.popleft()
                for mv in DIRS:
                    nxt = (cur[0] + mv.value[0], cur[1] + mv.value[1])
                    if nxt in self._valid and nxt not in dist:
                        dist[nxt] = dist[cur] + 1
                        q.append(nxt)
            self._bfs_cache[start] = dist
        return self._bfs_cache[start]

    # ---------- Pacman action generation --------------------------------

    def _pacman_actions(self, pos):
        actions = []
        for mv in DIRS:
            r, c = pos
            valid_steps = 0
            for _ in range(self.pacman_speed):
                r += mv.value[0]
                c += mv.value[1]
                if (r, c) not in self._valid:
                    break
                valid_steps += 1
            for s in range(1, valid_steps + 1):
                actions.append((mv, s))
        return actions if actions else [(Move.STAY, 1)]

    def _apply_action(self, pos, action):
        move, steps = action
        if move == Move.STAY:
            return pos
        r, c = pos
        for _ in range(steps):
            nr, nc = r + move.value[0], c + move.value[1]
            if (nr, nc) not in self._valid:
                break
            r, c = nr, nc
        return (r, c)

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
from hide_agent import panic
from debug import debug
from hide_agent import topology
from hide_agent import control

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = "Skidadal"
        self.last_known_enemy_pos = None

        self.pacman_speed = 2
        self.topology_map = None
        self.previous_position = None

        debug.clear_log()

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        control.reset_timer()
        map_state = np.asarray(map_state)

        if self.topology_map is None:
            self.topology_map = topology.build_topology_score_map(map_state)
            if debug.DEBUG_ENABLED:
                topology.write_topology_score_map(self.topology_map, map_state)

        my_position = (int(my_position[0]), int(my_position[1]))
        enemy_position = (int(enemy_position[0]), int(enemy_position[1]))

        mode = (
            "PANIC"
            if panic.should_panic(enemy_position, my_position, map_state, self.pacman_speed)
            else "CONTROL"
        )
        debug.start_turn(
            step_number,
            mode,
            my_position,
            enemy_position,
            self.previous_position,
        )

        try:
            move = self._choose_move(
                map_state,
                my_position,
                enemy_position,
                self.pacman_speed,
                mode,
            )
        except Exception as error:
            debug.log_exception(error)
            debug.finish_turn(control.get_run_time())
            raise

        self.previous_position = my_position

        if not isinstance(move, Move):
            debug.event("invalid-move", returned=repr(move), fallback="STAY")
            debug.decision(Move.STAY, my_position, "agent returned an invalid move")
            debug.finish_turn(control.get_run_time())
            return Move.STAY

        debug.finish_turn(control.get_run_time())
        return move

    def _choose_move(self, map_state: np.ndarray, my_position: tuple[int, int], enemy_position: tuple[int, int], pacman_speed=2, mode=None):
        if enemy_position is None:
            return Move.STAY

        if mode is None:
            mode = (
                "PANIC"
                if panic.should_panic(enemy_position, my_position, map_state, pacman_speed)
                else "CONTROL"
            )
        
        if mode == "PANIC":
            return panic.choose_move(map_state, my_position, enemy_position)
        
        return control.choose_move(
            my_position,
            enemy_position,
            map_state,
            self.topology_map,
            pacman_speed,
            self.previous_position
        )
