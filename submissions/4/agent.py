import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move
import numpy as np
import random
from collections import deque

from seek_agent.topology import build_trap_map

class PacmanAgent(BasePacmanAgent):
    """Pacman seeker: 1-turn lookahead with turn-distance + topology evaluation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "HomeLander"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.last_known_enemy_pos = None
        self._bfs_cache = {}
        self._turn_cache = {}

    # ── public ────────────────────────────────────────────────────
    def step(self, map_state, my_position, enemy_position, step_number):
        if enemy_position is not None:
            enemy_position = (int(enemy_position[0]), int(enemy_position[1]))
            self.last_known_enemy_pos = enemy_position
        my_position = (int(my_position[0]), int(my_position[1]))
        target = enemy_position or self.last_known_enemy_pos
        if target is None:
            return self._explore(my_position, map_state)
        if self._md(my_position, target) < 2:
            return Move.STAY, 1
        return self._decide(my_position, target, map_state)

    # ── 1-turn lookahead ──────────────────────────────────────────
    def _decide(self, my_pos, g_pos, m):
        pacts = self._p_acts(my_pos, m)
        gmoves = self._g_moves(g_pos, m)
        best_act, best_val = pacts[0], 10 ** 9

        for pa in pacts:
            np_ = self._step(my_pos, pa, m)
            worst = 0
            for gm in gmoves:
                ng_ = self._g_step(g_pos, gm, m)
                goals = self._goals(ng_, m)
                td = self._td(np_, goals, m)
                md = self._bfs(np_, m).get(ng_, 10 ** 9)
                exits = sum(1 for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
                            if self._ok((ng_[0]+mv.value[0], ng_[1]+mv.value[1]), m))
                v = td * 100 + md * 2 + exits * 10
                if v > worst:
                    worst = v
            if worst < best_val:
                best_val = worst
                best_act = pa
        return best_act

    # ── evaluation helpers ────────────────────────────────────────
    def _goals(self, pos, m):
        goals = [pos]
        for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            n = (pos[0] + mv.value[0], pos[1] + mv.value[1])
            if self._ok(n, m):
                goals.append(n)
        return goals

    def _bfs(self, start, m):
        if start in self._bfs_cache:
            return self._bfs_cache[start]
        if not self._ok(start, m):
            return {}
        d = {start: 0}
        q = deque([start])
        while q:
            c = q.popleft()
            for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                n = (c[0]+mv.value[0], c[1]+mv.value[1])
                if self._ok(n, m) and n not in d:
                    d[n] = d[c] + 1
                    q.append(n)
        self._bfs_cache[start] = d
        return d

    def _td(self, start, goals, m):
        td = self._turns(start, m)
        return min(td.get(g, 10 ** 9) for g in goals)

    def _turns(self, start, m):
        if start in self._turn_cache:
            return self._turn_cache[start]
        if not self._ok(start, m):
            self._turn_cache[start] = {}
            return {}
        d = {start: 0}
        q = deque([start])
        while q:
            c = q.popleft()
            for a in self._p_acts(c, m):
                n = self._step(c, a, m)
                if n not in d:
                    d[n] = d[c] + 1
                    q.append(n)
        self._turn_cache[start] = d
        return d

    # ── actions ───────────────────────────────────────────────────
    def _p_acts(self, pos, m):
        acts = []
        for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            ms = self._speed(pos, mv, m)
            for s in range(1, ms + 1):
                acts.append((mv, s))
        return acts or [(Move.STAY, 1)]

    def _g_moves(self, pos, m):
        return [mv for mv in Move
                if self._ok((pos[0]+mv.value[0], pos[1]+mv.value[1]), m)]

    def _step(self, pos, a, m):
        mv, s = a
        if mv == Move.STAY:
            return pos
        c = pos
        for _ in range(s):
            n = (c[0]+mv.value[0], c[1]+mv.value[1])
            if not self._ok(n, m):
                break
            c = n
        return c

    def _g_step(self, pos, mv, m):
        n = (pos[0]+mv.value[0], pos[1]+mv.value[1])
        return n if self._ok(n, m) else pos

    # ── helpers ───────────────────────────────────────────────────
    @staticmethod
    def _md(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    @staticmethod
    def _ok(pos, m):
        r, c = pos
        h, w = m.shape
        return 0 <= r < h and 0 <= c < w and m[r, c] == 0

    def _speed(self, pos, mv, m):
        s = 0
        c = pos
        for _ in range(self.pacman_speed):
            n = (c[0]+mv.value[0], c[1]+mv.value[1])
            if not self._ok(n, m):
                break
            s += 1
            c = n
        return s

    def _explore(self, p, m):
        ms = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(ms)
        for mv in ms:
            s = self._speed(p, mv, m)
            if s > 0:
                return (mv, s)
        return (Move.STAY, 1)


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
