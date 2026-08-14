import sys
from collections import deque
from pathlib import Path
import numpy as np

_SRC = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(_SRC))

from agent_interface import PacmanAgent as _BasePacmanAgent
from agent_interface import GhostAgent as _BaseGhostAgent
from environment import Move

MAP_SIZE: int = 21
CELL_UNKNOWN: int = -1
CELL_EMPTY:   int =  0
CELL_WALL:    int =  1
DIRS = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
DELTA_TO_MOVE = {(-1, 0): Move.UP, ( 1, 0): Move.DOWN, ( 0,-1): Move.LEFT, ( 0, 1): Move.RIGHT,}

#Next position
def next_pos(pos: tuple, mv: Move) -> tuple:
    dr, dc = mv.value
    return (pos[0] + dr, pos[1] + dc)

def manhattan(a: tuple, b: tuple) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

class Region:
    __slots__ = ("id", "cells", "frontier_cells", "unknown_border", "last_visit_step")

    def __init__(self, region_id: int):
        self.id = region_id
        self.cells: list = []             # every known EMPTY cell in the region
        self.frontier_cells: list = []    # cells in `cells` that touch an UNKNOWN cell
        self.unknown_border: set = set()  # distinct UNKNOWN cells touching the region
        self.last_visit_step: int = -1    # filled in by _update_region_stats

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def unexplored_estimate(self) -> int:
        return len(self.unknown_border)

    @property
    def exploration_ratio(self) -> float:
        total = self.size + self.unexplored_estimate
        if total == 0:
            return 1.0
        return self.size / total

    @property
    def is_fully_explored(self) -> bool:
        return self.unexplored_estimate == 0


class PacmanAgent(_BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "YoruNikakeru Pacman"
        self.pacman_speed: int = max(1, int(kwargs.get("pacman_speed", 2)))
        self.global_map: np.ndarray = np.full((MAP_SIZE, MAP_SIZE), CELL_UNKNOWN, dtype=np.int8)
        self.last_seen_ghost: tuple | None = None
        self.ghost_velocity: tuple = (0, 0)
        self.steps_since_seen_ghost: int = 0
        self.last_frontier_target: tuple | None = None
        self.regions: dict[int, Region] = {}
        self.region_id_grid: np.ndarray = np.full((MAP_SIZE, MAP_SIZE), -1, dtype=np.int32)
        self.visit_step_grid: np.ndarray = np.full((MAP_SIZE, MAP_SIZE), -1, dtype=np.int32)
        self.committed_anchor: tuple | None = None

        self.REGION_NEAR_DONE_THRESHOLD = 0.85  
        self.REGION_SWITCH_MARGIN = 6.0        

    #Update the map after observation
    def update_map(self, map_state: np.ndarray) -> None:
        visible_mask = map_state >= 0
        self.global_map[visible_mask] = map_state[visible_mask].astype(np.int8)

    #If agent can move to this cell
    def is_valid_position(self, pos: tuple) -> bool:
        r, c = pos
        h, w = self.global_map.shape
        return 0 <= r < h and 0 <= c < w and self.global_map[r, c] != CELL_WALL

    #Get all cell that pacman can move to
    def neighbors(self, pos: tuple) -> list:
        result = []
        for mv in DIRS:
            nb = next_pos(pos, mv)
            if self.is_valid_position(nb):
                result.append(nb)
        return result

    #count how many unknown cells nearby
    def unknown_neighbor_count(self, pos: tuple) -> int:
        r, c = pos
        count = 0
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ]
        for dr, dc in directions:
            for i in range(1, 6):
                nr = r + dr * i
                nc = c + dc * i
                if not (0 <= nr < MAP_SIZE and 0 <= nc < MAP_SIZE):
                    break
                if self.global_map[nr, nc] == CELL_WALL:
                    break
                if self.global_map[nr, nc] == CELL_UNKNOWN:
                    count += 1
        return count

    def is_frontier(self, pos: tuple) -> bool:
        return self.unknown_neighbor_count(pos) > 0

    def frontier_score(self, pos: tuple) -> int:
        unknowns = set()
        for nb in self.neighbors(pos):
            if self.global_map[nb] == CELL_UNKNOWN:
                unknowns.add(nb)
            for nb2 in self.neighbors(nb):
                if self.global_map[nb2] == CELL_UNKNOWN:
                    unknowns.add(nb2)
        return len(unknowns)

    def heuristic(self, pos: tuple, dist: int) -> float:
        reveal = self.unknown_neighbor_count(pos)
        frontier = self.frontier_score(pos)
        degree = len(self.neighbors(pos))
        top_bias = 0
        if reveal > 0:
            top_bias = (MAP_SIZE - 1 - pos[0]) * 0.5
        efficiency = frontier / (dist + 1)
        dead_end_penalty = -3 if degree == 1 else 0
        score = (
            reveal * 8.0
            + frontier * 2.5
            + efficiency * 10.0
            + top_bias
            - dist * 0.8
            + dead_end_penalty
        )
        if pos == self.last_frontier_target:
            score += 2
        return score

    def _cell_neighbors(self, pos: tuple) -> list:
        result = []
        for mv in DIRS:
            nr, nc = next_pos(pos, mv)
            if 0 <= nr < MAP_SIZE and 0 <= nc < MAP_SIZE:
                result.append((nr, nc))
        return result

    def _generate_regions(self) -> None:
        self.region_id_grid = np.full((MAP_SIZE, MAP_SIZE), -1, dtype=np.int32)
        self.regions = {}
        next_id = 0
        for r in range(MAP_SIZE):
            for c in range(MAP_SIZE):
                if self.global_map[r, c] != CELL_EMPTY or self.region_id_grid[r, c] != -1:
                    continue
                region = Region(next_id)
                stack = [(r, c)]
                self.region_id_grid[r, c] = next_id
                while stack:
                    cur = stack.pop()
                    region.cells.append(cur)
                    touches_unknown = False
                    for nb in self._cell_neighbors(cur):
                        nr, nc = nb
                        val = self.global_map[nr, nc]
                        if val == CELL_UNKNOWN:
                            region.unknown_border.add(nb)
                            touches_unknown = True
                        elif val == CELL_EMPTY and self.region_id_grid[nr, nc] == -1:
                            self.region_id_grid[nr, nc] = next_id
                            stack.append(nb)
                    if touches_unknown:
                        region.frontier_cells.append(cur)
                self.regions[next_id] = region
                next_id += 1

    def _update_region_stats(self, my_pos: tuple, step_number: int) -> None:
        self.visit_step_grid[my_pos] = step_number
        for region in self.regions.values():
            best = -1
            for cell in region.cells:
                v = self.visit_step_grid[cell]
                if v > best:
                    best = v
            region.last_visit_step = best

    def _score_region(self, region: Region, my_pos: tuple, step_number: int) -> float:
        if region.is_fully_explored:
            return float("-inf")
        unexplored_ratio = 1.0 - region.exploration_ratio
        frontier_count = len(region.frontier_cells)
        nearest_frontier_dist = min(manhattan(my_pos, cell) for cell in region.frontier_cells)
        if region.last_visit_step < 0:
            steps_idle = step_number + 1 
        else:
            steps_idle = step_number - region.last_visit_step
        return (
            unexplored_ratio * 12.0
            + frontier_count * 1.5
            - nearest_frontier_dist * 1.0
            + min(steps_idle, 50) * 0.3
        )

    def _select_target_region(self, my_pos: tuple, step_number: int) -> "Region | None":
        candidates = [r for r in self.regions.values() if not r.is_fully_explored]
        if not candidates:
            return None
        best_region = max(candidates, key=lambda r: self._score_region(r, my_pos, step_number))
        best_score = self._score_region(best_region, my_pos, step_number)
        committed_region = None
        if self.committed_anchor is not None:
            rid = int(self.region_id_grid[self.committed_anchor])
            if rid != -1:
                committed_region = self.regions.get(rid)
        if committed_region is None or committed_region.is_fully_explored:
            chosen = best_region
        else:
            committed_score = self._score_region(committed_region, my_pos, step_number)
            almost_done = committed_region.exploration_ratio >= self.REGION_NEAR_DONE_THRESHOLD
            rival_much_better = (
                best_region.id != committed_region.id
                and best_score > committed_score + self.REGION_SWITCH_MARGIN
            )
            chosen = best_region if (almost_done or rival_much_better) else committed_region
        self.committed_anchor = chosen.cells[0]
        return chosen

    def _select_frontier_in_region(self, region: Region, my_pos: tuple) -> "tuple | None":
        best_cell = None
        best_score = float("-inf")
        for cell in region.frontier_cells:
            dist = manhattan(my_pos, cell)
            gain = self.unknown_neighbor_count(cell)
            score = gain * 3.0 - dist * 1.0
            if score > best_score:
                best_score = score
                best_cell = cell
        return best_cell

    def _explore_with_regions(self, my_pos: tuple, step_number: int) -> "list | None":
        target_region = self._select_target_region(my_pos, step_number)
        if target_region is None:
            return self.bfs(my_pos, target=None)

        frontier_cell = self._select_frontier_in_region(target_region, my_pos)
        if frontier_cell is None:
            return self.bfs(my_pos, target=None)

        path = self.bfs(my_pos, target=frontier_cell)
        if path is None:
            self.committed_anchor = None
            return self.bfs(my_pos, target=None)
        return path

    #Shortest path to target
    def bfs(self, start: tuple, target: tuple | None) -> list | None:
        parent: dict = {start: None}
        depth: dict = {start: 0}
        queue: deque = deque([start])
        frontier_candidates: list = []
        while queue:
            current = queue.popleft()
            if target is not None:
                if current == target:
                    return self.reconstruct_path(parent, current)
            else:
                if (self.global_map[current[0], current[1]] == CELL_EMPTY
                        and self.is_frontier(current)):
                    frontier_candidates.append(current)
            for nb in self.neighbors(current):
                if nb not in parent:
                    parent[nb] = current
                    depth[nb] = depth[current] + 1
                    queue.append(nb)
        if target is not None or not frontier_candidates:
            return None
        best = max(frontier_candidates, key=lambda p: self.heuristic(p, depth[p]))
        if depth[best] == 0:
            r, c = best
            for mv in DIRS:
                nr, nc = next_pos(best, mv)
                if (0 <= nr < MAP_SIZE and 0 <= nc < MAP_SIZE
                        and self.global_map[nr, nc] == CELL_UNKNOWN):
                    self.last_frontier_target = (nr, nc)
                    return [start, (nr, nc)]
            return None
        self.last_frontier_target = best
        return self.reconstruct_path(parent, best)

    def reconstruct_path(self, parent: dict, goal: tuple) -> list:
        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path

    #Predict ghost location after miss
    def project_ghost_pos(self, last_seen: tuple, velocity: tuple, steps_hidden: int) -> tuple:
        if velocity == (0, 0):
            return last_seen
        pos = last_seen
        for _ in range(min(6, steps_hidden)):
            nxt = (pos[0] + velocity[0], pos[1] + velocity[1])
            if not self.is_valid_position(nxt):
                break
            pos = nxt
        return pos

    #Path -> action
    def path_to_action(self, path: list) -> tuple:
        assert len(path) >= 2, "Require start and next"
        dr = path[1][0] - path[0][0]
        dc = path[1][1] - path[0][1]
        move = DELTA_TO_MOVE.get((dr, dc))
        if move is None:
            return (Move.STAY, 1)
        steps = 1
        for i in range(2, len(path)):
            delta = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            if delta != (dr, dc):
                break
            steps += 1
            if steps >= self.pacman_speed:
                break
        return (move, steps)

    def step(self, map_state: np.ndarray, my_position: tuple,
             enemy_position: tuple | None, step_number: int) -> tuple:
        my_pos: tuple = (int(my_position[0]), int(my_position[1]))
        self.update_map(map_state)
        self._generate_regions()
        self._update_region_stats(my_pos, step_number)

        path = None
        if enemy_position is not None:
            self.committed_anchor = None
            ghost_pos = (int(enemy_position[0]), int(enemy_position[1]))

            if self.last_seen_ghost is not None and self.steps_since_seen_ghost == 0:
                vr = ghost_pos[0] - self.last_seen_ghost[0]
                vc = ghost_pos[1] - self.last_seen_ghost[1]
                self.ghost_velocity = (
                    (vr // abs(vr)) if vr != 0 else 0,
                    (vc // abs(vc)) if vc != 0 else 0,
                )
            else:
                self.ghost_velocity = (0, 0)
            self.last_seen_ghost = ghost_pos
            self.steps_since_seen_ghost = 0
            path = self.bfs(my_pos, target=ghost_pos)
        else:
            self.steps_since_seen_ghost += 1

        if path is None and self.last_seen_ghost is not None:
            projected = self.project_ghost_pos(
                self.last_seen_ghost, self.ghost_velocity, self.steps_since_seen_ghost
            )
            if projected != self.last_seen_ghost:
                path = self.bfs(my_pos, target=projected)
            if path is None:
                path = self.bfs(my_pos, target=self.last_seen_ghost)

        if path is None:
            path = self._explore_with_regions(my_pos, step_number)

        if path is None or len(path) < 2:
            return (Move.STAY, 1)
        return self.path_to_action(path)


"""
Ghost hides from pacman in blind mode arena.
Strategy: when pacman is visible, flee toward the reachable cell that
maximizes BFS distance from pacman (weighted by escape routes and by
avoiding cells that share pacman's row/column, since that is what lets
pacman "see" the ghost along a vision ray). When pacman is not visible,
patrol toward unexplored / rarely-visited cells, biased away from the
last known pacman position.
"""
class GhostAgent(_BaseGhostAgent):
    OPPOSITE = {
        Move.UP: Move.DOWN, Move.DOWN: Move.UP,
        Move.LEFT: Move.RIGHT, Move.RIGHT: Move.LEFT,
        Move.STAY: Move.STAY,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "YoruNikakeru Ghost"
        self.global_map: np.ndarray = np.full((MAP_SIZE, MAP_SIZE), CELL_UNKNOWN, dtype=np.int8)
        self.visit_count: np.ndarray = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int32)
        self.last_seen_enemy: tuple | None = None
        self.steps_since_seen_enemy: int = 0
        self.last_move: Move = Move.STAY
        # how many steps to keep treating a lost enemy as a nearby threat
        self.danger_memory: int = 15
        # if last-seen enemy is within this manhattan range, keep fleeing
        self.flee_radius: int = 6
        # how many future rounds to simulate when choosing a flee direction
        self.flee_horizon: int = int(kwargs.get("flee_horizon", 6))
        # matches the arena's fairness rule: pacman moves 2 cells/step in a
        # straight line, but only 1 cell if it must turn (no L-shaped moves)
        self.pacman_speed: int = int(kwargs.get("pacman_speed", 2))

    #Update the map after observation
    def update_map(self, map_state: np.ndarray) -> None:
        visible_mask = map_state >= 0
        self.global_map[visible_mask] = map_state[visible_mask].astype(np.int8)

    #If agent can move to this cell
    def is_valid_position(self, pos: tuple) -> bool:
        r, c = pos
        h, w = self.global_map.shape
        return 0 <= r < h and 0 <= c < w and self.global_map[r, c] != CELL_WALL

    #Get all cells reachable from pos in one move
    def neighbors(self, pos: tuple) -> list:
        result = []
        for mv in DIRS:
            nb = next_pos(pos, mv)
            if self.is_valid_position(nb):
                result.append(nb)
        return result

    #count how many unknown cells are adjacent (useful for exploration reward)
    def unknown_neighbor_count(self, pos: tuple) -> int:
        count = 0
        for mv in DIRS:
            nr, nc = next_pos(pos, mv)
            if (0 <= nr < MAP_SIZE and 0 <= nc < MAP_SIZE
                    and self.global_map[nr, nc] == CELL_UNKNOWN):
                count += 1
        return count

    #BFS distance from a single source to every reachable cell
    def bfs_distances(self, start: tuple) -> dict:
        dist = {start: 0}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for nb in self.neighbors(cur):
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    queue.append(nb)
        return dist

    #BFS parent tree from a single source, used to walk toward a chosen target
    def bfs_tree(self, start: tuple) -> tuple:
        parent = {start: None}
        depth = {start: 0}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for nb in self.neighbors(cur):
                if nb not in parent:
                    parent[nb] = cur
                    depth[nb] = depth[cur] + 1
                    queue.append(nb)
        return parent, depth

    def reconstruct_path(self, parent: dict, goal: tuple) -> list:
        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path

    #BFS shortest path between two known cells (used to simulate pacman's pursuit)
    def bfs_path(self, start: tuple, target: tuple) -> list | None:
        if start == target:
            return [start]
        parent = {start: None}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for nb in self.neighbors(cur):
                if nb not in parent:
                    parent[nb] = cur
                    if nb == target:
                        return self.reconstruct_path(parent, nb)
                    queue.append(nb)
        return None

    #Model one enemy turn chasing `ghost_pos`: 2 cells if it can go straight,
    #otherwise only 1 cell (mirrors the arena's no-L-shaped-turn speed rule)
    def _advance_enemy(self, enemy_pos: tuple, ghost_pos: tuple) -> tuple:
        path = self.bfs_path(enemy_pos, ghost_pos)
        if not path or len(path) < 2:
            return enemy_pos
        step1 = path[1]
        if self.pacman_speed >= 2 and len(path) >= 3:
            d1 = (step1[0] - enemy_pos[0], step1[1] - enemy_pos[1])
            step2 = path[2]
            d2 = (step2[0] - step1[0], step2[1] - step1[1])
            if d1 == d2:
                return step2
        return step1

    #Cheap single-ply "run away" used only inside the lookahead simulation
    def _greedy_step_away(self, pos: tuple, threat_pos: tuple) -> tuple:
        dist_map = self.bfs_distances(threat_pos)
        best_cell, best_score = pos, float("-inf")
        for mv in DIRS:
            nb = next_pos(pos, mv)
            if not self.is_valid_position(nb):
                continue
            d = dist_map.get(nb, 0)
            score = d * 3.0 + len(self.neighbors(nb)) * 1.0
            if nb[0] == threat_pos[0] or nb[1] == threat_pos[1]:
                score -= 2.5
            if score > best_score:
                best_score, best_cell = score, nb
        return best_cell

    #Mirrors the arena's cross-shaped vision: a viewer sees `target` only if
    #they share a row/col, are within 5 cells, and no KNOWN wall sits between
    #them. Unknown cells are treated as non-blocking (we simply can't be sure),
    #so this only ever reports "visible" when we're not certain it's actually
    #hidden -- i.e. it's conservative in the direction that matters for safety.
    def _visible_to(self, viewer: tuple, target: tuple) -> bool:
        if viewer == target:
            return True
        if viewer[0] == target[0] and abs(viewer[1] - target[1]) <= 5:
            r = viewer[0]
            c0, c1 = sorted((viewer[1], target[1]))
            return not any(self.global_map[r, c] == CELL_WALL for c in range(c0 + 1, c1))
        if viewer[1] == target[1] and abs(viewer[0] - target[0]) <= 5:
            c = viewer[1]
            r0, r1 = sorted((viewer[0], target[0]))
            return not any(self.global_map[r, c] == CELL_WALL for r in range(r0 + 1, r1))
        return False

    def _dist(self, a: tuple, b: tuple) -> int:
        return self.bfs_distances(a).get(b, 999)

    #Roll a candidate move forward `horizon` rounds. Critically, the simulated
    #enemy only chases the ghost's TRUE position while it can actually see it
    def _simulate_score(self, ghost_pos: tuple, enemy_pos: tuple, horizon: int) -> float:
        sim_ghost, sim_enemy = ghost_pos, enemy_pos
        belief = ghost_pos  # what the enemy currently believes about our position
        dist0 = self._dist(sim_enemy, sim_ghost)
        if dist0 < 2:
            return -1000.0
        survived = 0
        min_dist = dist0
        hidden_bonus = 0.0
        for _ in range(horizon):
            sim_enemy = self._advance_enemy(sim_enemy, belief)
            d = self._dist(sim_enemy, sim_ghost)
            if d < 2:
                return survived * 60.0 - 500.0
            survived += 1
            min_dist = min(min_dist, d)
            if self._visible_to(sim_enemy, sim_ghost):
                belief = sim_ghost
            else:
                hidden_bonus += 8.0  # every round spent off pacman's radar is huge
            sim_ghost = self._greedy_step_away(sim_ghost, belief)
        return survived * 60.0 + min_dist * 3.0 + hidden_bonus

    #Pick the move that survives longest against a simulated pursuit
    def flee_move(self, my_pos: tuple, danger_pos: tuple) -> Move:
        candidates = [(mv, next_pos(my_pos, mv)) for mv in DIRS]
        candidates = [(mv, nb) for mv, nb in candidates if self.is_valid_position(nb)]
        if not candidates:
            return Move.STAY

        best_move, best_score = candidates[0][0], float("-inf")
        for mv, nb in candidates:
            score = self._simulate_score(nb, danger_pos, self.flee_horizon)
            branch = len(self.neighbors(nb))
            score += branch * 0.5
            if branch <= 1:
                score -= 4.0  # dead end: very risky if spotted again
            if not self._visible_to(danger_pos, nb):
                score += 6.0
            elif nb[0] == danger_pos[0] or nb[1] == danger_pos[1]:
                score -= 4.0
            if mv == self.OPPOSITE.get(self.last_move):
                score -= 0.3
            if score > best_score:
                best_score, best_move = score, mv
        return best_move

    #Local, reactive "climb toward the top" behaviour, used only before
    #Pacman has ever been spotted (it spawns below the ghost, so heading
    #up first buys distance for free). This ONLY ever looks at the
    #ghost's 4 immediate neighbours -- it never calls explore_move's
    #map-wide BFS search. That's the key fix: explore_move scores every
    #reachable cell in the whole known map and can pick a target on the
    #far side of a wall, so its planned path might have to detour DOWN
    #and around before coming back UP. This function instead just says
    #"go up; if that's blocked, keep sliding sideways along the wall
    #(preferring whichever side you were already sliding, so you don't
    #zig-zag) until up opens again; only drop down if both sides are
    #also walls".
    def escape_upward(self, my_pos: tuple) -> Move:
        up = next_pos(my_pos, Move.UP)
        if self.is_valid_position(up):
            return Move.UP

        left = next_pos(my_pos, Move.LEFT)
        right = next_pos(my_pos, Move.RIGHT)
        left_ok = self.is_valid_position(left)
        right_ok = self.is_valid_position(right)

        # keep gliding the same side we were already gliding, so we don't
        # flip-flop left/right against the same wall segment every step
        if self.last_move == Move.LEFT and left_ok:
            return Move.LEFT
        if self.last_move == Move.RIGHT and right_ok:
            return Move.RIGHT

        if left_ok and right_ok:
            # no established side yet: pick whichever is less visited
            return Move.LEFT if self.visit_count[left] <= self.visit_count[right] else Move.RIGHT
        if left_ok:
            return Move.LEFT
        if right_ok:
            return Move.RIGHT

        # both sides are walls too -- genuine dead end, down is the only option
        down = next_pos(my_pos, Move.DOWN)
        if self.is_valid_position(down):
            return Move.DOWN
        return Move.STAY

    #Patrol toward unexplored/rarely-visited ground, biased away from `threat_pos`
    def explore_move(self, my_pos: tuple, threat_pos: tuple | None) -> Move:
        parent, depth = self.bfs_tree(my_pos)
        candidates = []
        for pos in parent:
            if pos == my_pos:
                continue
            if self.unknown_neighbor_count(pos) > 0 or self.visit_count[pos] == 0:
                candidates.append(pos)
        if not candidates:
            candidates = [p for p in parent if p != my_pos]
        if not candidates:
            return Move.STAY

        def score(pos: tuple) -> float:
            branch = len(self.neighbors(pos))
            s = (- self.visit_count[pos] * 10.0 
                 - depth[pos] * 0.5 
                 + branch * 0.5)

            s -= pos[0] * 5.0
            
            if branch <= 1:
                s -= 3.0  # avoid parking the patrol route in dead ends
            if threat_pos is not None:
                s += (abs(pos[0] - threat_pos[0]) + abs(pos[1] - threat_pos[1])) * 1.5
                if not self._visible_to(threat_pos, pos):
                    s += 4.0
                elif pos[0] == threat_pos[0] or pos[1] == threat_pos[1]:
                    s -= 2.0
            return s

        best = max(candidates, key=score)
        path = self.reconstruct_path(parent, best)
        if len(path) < 2:
            return Move.STAY
        dr, dc = path[1][0] - my_pos[0], path[1][1] - my_pos[1]
        return DELTA_TO_MOVE.get((dr, dc), Move.STAY)

    def step(self, map_state: np.ndarray, my_position: tuple,
              enemy_position: tuple | None, step_number: int) -> Move:
        my_pos: tuple = (int(my_position[0]), int(my_position[1]))
        self.update_map(map_state)
        self.visit_count[my_pos] += 1

        if enemy_position is not None:
            enemy_pos = (int(enemy_position[0]), int(enemy_position[1]))
            self.last_seen_enemy = enemy_pos
            self.steps_since_seen_enemy = 0
            move = self.flee_move(my_pos, enemy_pos)
            self.last_move = move
            return move

        if self.last_seen_enemy is None:
            move = self.escape_upward(my_pos)
            self.last_move = move
            return move
        # ---------------------------------------------

        self.steps_since_seen_enemy += 1
        threat_pos = None
        
        # Chỉ lẩn trốn dựa vào vị trí cuối cùng nhìn thấy
        if (self.last_seen_enemy is not None
                and self.steps_since_seen_enemy <= self.danger_memory):
            threat_pos = self.last_seen_enemy
            dist_to_threat = abs(my_pos[0] - threat_pos[0]) + abs(my_pos[1] - threat_pos[1])
            if dist_to_threat <= self.flee_radius:
                move = self.flee_move(my_pos, threat_pos)
                self.last_move = move
                return move

        # Tuần tra (chỉ kích hoạt sau khi đã cắt đuôi thành công)
        move = self.explore_move(my_pos, threat_pos)
        self.last_move = move
        return move