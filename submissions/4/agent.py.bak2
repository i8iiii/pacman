import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move
import numpy as np
from collections import deque
import heapq

DIRS = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
INF = 10 ** 9


class PacmanAgent(BasePacmanAgent):
    """Pacman seeker: A* for long range, Minimax Alpha-Beta (depth 6) when close.

    Uses BFS distance threshold to switch: A* above threshold, minimax below.
    Minimax models the ghost as maximising BFS distance from Pacman.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "HomeLander"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self._valid = None
        self._bfs_cache = {}
        self._pair_dist = {}

    def step(self, map_state, my_position, enemy_position, step_number):
        if enemy_position is None:
            return (Move.STAY, 1)
        my_pos = (int(my_position[0]), int(my_position[1]))
        enemy_pos = (int(enemy_position[0]), int(enemy_position[1]))
        if self._valid is None:
            self._precompute(map_state)
        if self._manhattan(my_pos, enemy_pos) < 2:
            return (Move.STAY, 1)
        dist = self._bfs_dist(my_pos, enemy_pos)
        if dist > 10:
            return self._astar_chase(my_pos, enemy_pos)
        else:
            return self._minimax_root(my_pos, enemy_pos)

    def _precompute(self, map_state):
        h, w = map_state.shape
        self._valid = {(r, c) for r in range(h) for c in range(w) if map_state[r, c] == 0}
        self._bfs_cache.clear(); self._pair_dist.clear()

    def _bfs_from(self, start):
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

    def _bfs_dist(self, a, b):
        if a == b: return 0
        k = (a, b)
        if k not in self._pair_dist:
            self._pair_dist[k] = self._bfs_from(a).get(b, INF)
        return self._pair_dist[k]

    def _astar_chase(self, my_pos, enemy_pos):
        path = self._astar(my_pos, enemy_pos)
        if path and len(path) >= 2:
            return self._follow_path(my_pos, path)
        return self._greedy(my_pos, enemy_pos)

    def _astar(self, start, goal):
        if start == goal or start not in self._valid or goal not in self._valid:
            return None
        frontier = [(0, start)]
        came_from = {start: None}; cost = {start: 0}
        while frontier:
            _, cur = heapq.heappop(frontier)
            if cur == goal: break
            for mv in DIRS:
                nxt = (cur[0]+mv.value[0], cur[1]+mv.value[1])
                if nxt not in self._valid: continue
                nc = cost[cur] + 1
                if nxt not in cost or nc < cost[nxt]:
                    cost[nxt] = nc
                    heapq.heappush(frontier, (nc + self._manhattan(nxt, goal), nxt))
                    came_from[nxt] = cur
        if goal not in came_from: return None
        path = []; cur = goal
        while cur is not None: path.append(cur); cur = came_from[cur]
        path.reverse(); return path

    def _follow_path(self, my_pos, path):
        nxt = path[1]
        dr, dc = nxt[0]-my_pos[0], nxt[1]-my_pos[1]
        direction = Move.DOWN if dr>0 else Move.UP if dr<0 else Move.RIGHT if dc>0 else Move.LEFT
        consecutive = 0; cur = my_pos
        for node in path[1:]:
            ndr, ndc = node[0]-cur[0], node[1]-cur[1]
            nd = Move.DOWN if ndr>0 else Move.UP if ndr<0 else Move.RIGHT if ndc>0 else Move.LEFT
            if nd != direction: break
            cur = node; consecutive += 1
        return (direction, max(1, min(consecutive, self.pacman_speed)))

    def _greedy(self, my_pos, target):
        best_move, best_steps, best_dist = Move.STAY, 1, self._manhattan(my_pos, target)
        for mv in DIRS:
            r, c = my_pos; valid_steps = 0
            for _ in range(self.pacman_speed):
                r += mv.value[0]; c += mv.value[1]
                if (r,c) not in self._valid: break
                valid_steps += 1
            for s in range(1, valid_steps+1):
                dist = self._manhattan((my_pos[0]+mv.value[0]*s, my_pos[1]+mv.value[1]*s), target)
                if dist < best_dist: best_dist, best_move, best_steps = dist, mv, s
        return (best_move, best_steps)

    def _minimax_root(self, pac_pos, ghost_pos):
        actions = self._pacman_actions(pac_pos)
        actions.sort(key=lambda a: self._bfs_dist(self._apply_action(pac_pos, a), ghost_pos))
        best_score, best_action = -INF, (Move.STAY, 1)
        alpha, beta = -INF, INF
        for action in actions:
            new_pac = self._apply_action(pac_pos, action)
            score = self._min_node(new_pac, ghost_pos, 6, alpha, beta)
            if score > best_score: best_score = score; best_action = action
            alpha = max(alpha, score)
        return best_action

    def _min_node(self, new_pac, ghost_pos, depth, alpha, beta):
        ghost_opts = self._ghost_positions(ghost_pos)
        ghost_opts.sort(key=lambda g: -self._bfs_dist(new_pac, g[0]))
        best = INF
        for new_ghost, _ in ghost_opts:
            if self._manhattan(new_pac, new_ghost) < 2:
                val = 10000 + depth
            elif depth <= 1:
                val = self._heuristic(new_pac, new_ghost)
            else:
                val = self._max_node(new_pac, new_ghost, depth - 1, alpha, beta)
            best = min(best, val)
            if best <= alpha: return best
            beta = min(beta, best)
        return best

    def _max_node(self, pac_pos, ghost_pos, depth, alpha, beta):
        actions = self._pacman_actions(pac_pos)
        if not actions: return self._heuristic(pac_pos, ghost_pos)
        actions.sort(key=lambda a: self._bfs_dist(self._apply_action(pac_pos, a), ghost_pos))
        best = -INF
        for action in actions:
            new_pac = self._apply_action(pac_pos, action)
            val = self._min_node(new_pac, ghost_pos, depth, alpha, beta)
            best = max(best, val)
            if best >= beta: return best
            alpha = max(alpha, best)
        return best

    def _heuristic(self, pac_pos, ghost_pos):
        dist = self._bfs_dist(pac_pos, ghost_pos)
        mobility = self._count_exits(ghost_pos)
        return -dist * 10 - mobility * 5

    def _ghost_positions(self, ghost_pos):
        positions = [(ghost_pos, Move.STAY)]
        for mv in DIRS:
            nxt = (ghost_pos[0]+mv.value[0], ghost_pos[1]+mv.value[1])
            if nxt in self._valid: positions.append((nxt, mv))
        return positions

    def _count_exits(self, pos):
        return sum(1 for mv in DIRS if (pos[0]+mv.value[0], pos[1]+mv.value[1]) in self._valid)

    def _pacman_actions(self, pos):
        actions = []
        for mv in DIRS:
            r, c = pos
            for s in range(1, self.pacman_speed + 1):
                r += mv.value[0]; c += mv.value[1]
                if (r, c) not in self._valid: break
                actions.append((mv, s))
        return actions if actions else [(Move.STAY, 1)]

    def _apply_action(self, pos, action):
        move, steps = action
        if move == Move.STAY: return pos
        r, c = pos
        for _ in range(steps):
            nr, nc = r + move.value[0], c + move.value[1]
            if (nr, nc) not in self._valid: break
            r, c = nr, nc
        return (r, c)

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])


# ===================================================================
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
            if debug.DEBUG_ENABLED: topology.write_topology_score_map(self.topology_map, map_state)
        my_position = (int(my_position[0]), int(my_position[1]))
        enemy_position = (int(enemy_position[0]), int(enemy_position[1]))
        mode = "PANIC" if panic.should_panic(enemy_position, my_position, map_state, self.pacman_speed) else "CONTROL"
        debug.start_turn(step_number, mode, my_position, enemy_position, self.previous_position)
        try: move = self._choose_move(map_state, my_position, enemy_position, self.pacman_speed, mode)
        except Exception as error:
            debug.log_exception(error); debug.finish_turn(control.get_run_time()); raise
        self.previous_position = my_position
        if not isinstance(move, Move):
            debug.event("invalid-move", returned=repr(move), fallback="STAY")
            debug.decision(Move.STAY, my_position, "agent returned an invalid move")
            debug.finish_turn(control.get_run_time()); return Move.STAY
        debug.finish_turn(control.get_run_time()); return move
    def _choose_move(self, map_state: np.ndarray, my_position: tuple[int, int], enemy_position: tuple[int, int], pacman_speed=2, mode=None):
        if enemy_position is None: return Move.STAY
        if mode is None: mode = "PANIC" if panic.should_panic(enemy_position, my_position, map_state, pacman_speed) else "CONTROL"
        if mode == "PANIC": return panic.choose_move(map_state, my_position, enemy_position)
        return control.choose_move(my_position, enemy_position, map_state, self.topology_map, pacman_speed, self.previous_position)
