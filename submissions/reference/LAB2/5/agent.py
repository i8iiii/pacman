import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
from collections import deque
import numpy as np
import random
from heapq import heappush, heappop
import time
import math

# ==================== SHARED UTILITIES ====================

def get_valid_moves(pos, map_state):
    valid_moves = []
    height, width = map_state.shape
    r, c = pos
    directions = [
        (Move.UP, -1, 0),
        (Move.DOWN, 1, 0),
        (Move.LEFT, 0, -1),
        (Move.RIGHT, 0, 1)
    ]
    for move, dr, dc in directions:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < height and 0 <= nc < width:
            if map_state[nr, nc] != 1:
                valid_moves.append((move, (nr, nc)))
    return valid_moves

def check_cross_visibility(pos1, pos2, map_state, radius=0):
    r1, c1 = pos1
    r2, c2 = pos2
    if r1 != r2 and c1 != c2:
        return False
    dist = abs(r1 - r2) + abs(c1 - c2)
    if radius > 0 and dist > radius:
        return False
    if c1 == c2:
        step = 1 if r1 < r2 else -1
        for r in range(r1 + step, r2, step):
            if map_state[r, c1] == 1:
                return False
    else:
        step = 1 if c1 < c2 else -1
        for c in range(c1 + step, c2, step):
            if map_state[r1, c] == 1:
                return False
    return True

# -------------------- BFS DISTANCE CACHE --------------------

_DISTANCE_CACHE = {}

def get_bfs_distance(start, target, map_state):
    cache_key = (start, target, map_state.shape)
    sym_key = (target, start, map_state.shape)
    if cache_key in _DISTANCE_CACHE:
        return _DISTANCE_CACHE[cache_key]
    if sym_key in _DISTANCE_CACHE:
        return _DISTANCE_CACHE[sym_key]
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        curr, dist = queue.popleft()
        _DISTANCE_CACHE[(start, curr, map_state.shape)] = dist
        if curr == target:
            return dist
        for _, (nr, nc) in get_valid_moves(curr, map_state):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
    _DISTANCE_CACHE[cache_key] = float('inf')
    return float('inf')

def flood_fill_free_space(pos, map_state, limit=15):
    queue = deque([pos])
    visited = {pos}
    free_cells = 0
    while queue and free_cells < limit:
        curr = queue.popleft()
        free_cells += 1
        for _, next_pos in get_valid_moves(curr, map_state):
            if next_pos not in visited:
                visited.add(next_pos)
                queue.append(next_pos)
    return free_cells


# ==================== PACMAN AGENT ====================

class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Ultimate Pacman"

        self.internal_map = None
        self.map_initialized = False
        self.last_move = None
        self.step_number = 0

        # Belief Tracking
        self.belief = None
        self.belief_initialized = False
        self.last_known_enemy_pos = None
        self.steps_since_seen = 0
        self.enemy_history = deque(maxlen=5)
        self.my_pos_history = deque(maxlen=8)

    def step(self, map_state: np.ndarray, my_position: tuple,
             enemy_position: tuple, step_number: int):
        self.step_number = step_number
        # Update internal map with new visible information
        self._update_map_memory(map_state)
        # Update probability distribution of where the ghost might be
        self._update_belief(map_state, my_position, enemy_position)

        self.my_pos_history.append(my_position)
        is_looping = False
        if len(self.my_pos_history) == 8 and len(set(self.my_pos_history)) <= 3:
            is_looping = True

        # Determine the current state mode based on enemy visibility
        if enemy_position is not None:
            mode = "visible" # Ghost is in our line of sight
            target = enemy_position
        elif self.last_known_enemy_pos is not None and self.steps_since_seen <= 15:
            mode = "belief"  # Ghost is hidden, use probability tracking
            target = self._get_belief_target(my_position)
        else:
            mode = "explore" # Track lost completely, need to search the map
            target = None

        chosen_move = Move.STAY
        path = None

        if mode == "visible":
            target_pos = enemy_position if is_looping else self._get_interception_target(my_position, enemy_position)
            path = self.astar(my_position, target_pos)
            if path:
                chosen_move = path[0]
            else:
                chosen_move = self._greedy_toward(my_position, enemy_position)
        elif mode == "belief":
            path = self.astar(my_position, target)
            if path:
                chosen_move = path[0]
            else:
                chosen_move = self._greedy_toward(my_position, target)
        else:
            frontier = self._find_best_frontier(my_position)
            if frontier:
                path = self.astar(my_position, frontier)
                if path:
                    chosen_move = path[0]
            if chosen_move == Move.STAY:
                chosen_move = self._random_valid_move(my_position)

        # Escape loop
        if is_looping and mode != "visible":
            chosen_move = self._random_valid_move(my_position)

        steps = 1
        if chosen_move != Move.STAY and self.pacman_speed >= 2:
            can_move_2 = self._can_move_n(my_position, chosen_move, 2)
            if path and len(path) >= 2 and path[0] == chosen_move and path[1] != chosen_move:
                can_move_2 = False
            if can_move_2:
                steps = 2
                        
        self.last_move = chosen_move
        return (chosen_move, min(steps, self.pacman_speed))

    def _init_belief(self, map_state):
        # Initialize uniform probability for all walkable cells
        h, w = map_state.shape
        self.belief = np.zeros((h, w), dtype=np.float32)
        walkable = (map_state == 0)
        n = walkable.sum()
        if n > 0:
            self.belief[walkable] = 1.0 / n
        self.belief_initialized = True

    def _update_belief(self, map_state, my_pos, enemy_pos):
        h, w = map_state.shape
        if not self.belief_initialized:
            self._init_belief(map_state)

        new_belief = np.zeros_like(self.belief)
        dirs = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        
        # Spread probability to adjacent cells
        for r in range(h):
            for c in range(w):
                if self.belief[r, c] < 1e-9:
                    continue
                p = self.belief[r, c]
                neighbors = []
                for mv in dirs:
                    nr, nc = r + mv.value[0], c + mv.value[1]
                    if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] != 1:
                        neighbors.append((nr, nc))
                total = len(neighbors) + 1
                new_belief[r, c] += p / total
                for nr, nc in neighbors:
                    new_belief[nr, nc] += p / total

        # Ghost cannot move into walls
        new_belief[map_state == 1] = 0.0

        if enemy_pos is not None:
            # If ghost is visible, sure about its position
            new_belief[:] = 0.0
            new_belief[enemy_pos] = 1.0
            self.last_known_enemy_pos = enemy_pos
            self.steps_since_seen = 0
            self.enemy_history.append(enemy_pos)
        else:
            self.steps_since_seen += 1
            
            # If ghost is not visible, it cannot be in the cells we currently see
            known_empty = (map_state == 0) | (map_state == 2)
            new_belief[known_empty] = 0.0

        total = new_belief.sum()
        if total > 1e-9:
            new_belief /= total
        else:
            self._init_belief(map_state)
            return

        self.belief = new_belief

    def _get_belief_target(self, my_pos):
        # Choose the most probable ghost location as the target
        if self.belief is None:
            return self.last_known_enemy_pos
        flat_indices = np.argsort(self.belief.flatten())[::-1][:5]
        best_target = None
        best_score = -1
        for idx in flat_indices:
            r, c = np.unravel_index(idx, self.belief.shape)
            prob = self.belief[r, c]
            if prob < 1e-9: continue
            dist = self._manhattan(my_pos, (r, c))
            score = prob / (dist + 1)
            if score > best_score:
                best_score = score
                best_target = (r, c)
        return best_target or self.last_known_enemy_pos

    def _get_interception_target(self, my_pos, ghost_pos):
        # Calculate a future position to intercept the ghost instead of just following it
        if ghost_pos not in list(self.enemy_history)[-1:]:
            self.enemy_history.append(ghost_pos)
        if self._manhattan(my_pos, ghost_pos) <= 5:
            return ghost_pos
        if len(self.enemy_history) >= 2:
            prev = self.enemy_history[-2]
            curr = self.enemy_history[-1]
            # Predict future ghost position based on its recent movement direction
            dr, dc = curr[0] - prev[0], curr[1] - prev[1]
            predicted = (curr[0] + dr * 3, curr[1] + dc * 3)
            h, w = self.internal_map.shape
            predicted = (max(0, min(h - 1, predicted[0])), max(0, min(w - 1, predicted[1])))
            if self.internal_map[predicted] == 1:
                predicted = self._find_nearest_walkable(predicted)
            dist_pac = self._bfs_distance(my_pos, predicted)
            dist_ghost = self._bfs_distance(ghost_pos, predicted)
            if dist_pac is not None and dist_ghost is not None:
                effective_pac_dist = dist_pac / 1.5 if self.pacman_speed >= 2 else dist_pac
                if effective_pac_dist <= dist_ghost + 1:
                    return predicted
        return self._find_chokepoint(my_pos, ghost_pos) or ghost_pos

    def _find_nearest_walkable(self, pos):
        if self.internal_map is None: return pos
        h, w = self.internal_map.shape
        queue = deque([pos])
        visited = {pos}
        while queue:
            curr = queue.popleft()
            if self.internal_map[curr] != 1: return curr
            for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = (curr[0] + mv.value[0], curr[1] + mv.value[1])
                if 0 <= nxt[0] < h and 0 <= nxt[1] < w and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return pos

    def _find_chokepoint(self, my_pos, ghost_pos):
        # Find a narrow passage to trap the ghost
        if self.internal_map is None: return None
        queue = deque([(ghost_pos, 0)])
        visited = {ghost_pos}
        best, best_score = None, -1
        while queue:
            curr, depth = queue.popleft()
            if depth > 6: break
            exits = self._count_exits(curr)
            dist_pac = self._manhattan(curr, my_pos)
            dist_ghost = self._manhattan(curr, ghost_pos)
            if exits <= 2 and dist_pac < dist_ghost and curr != my_pos:
                score = dist_ghost - dist_pac
                if score > best_score:
                    best_score = score
                    best = curr
            for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = (curr[0] + mv.value[0], curr[1] + mv.value[1])
                if self._is_valid(nxt) and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, depth + 1))
        return best

    def _count_exits(self, pos):
        if self.internal_map is None: return 4
        return sum(1 for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT] 
                   if self._is_valid((pos[0] + mv.value[0], pos[1] + mv.value[1])))

    def _update_map_memory(self, map_state):
        if not self.map_initialized:
            self.internal_map = np.full_like(map_state, -1)
            self.internal_map[map_state == 1] = 1
            self.map_initialized = True
        visible_mask = map_state != -1
        self.internal_map[visible_mask] = map_state[visible_mask]
        self.internal_map[(self.internal_map == 2) | (self.internal_map == 3)] = 0

    def _greedy_toward(self, my_pos, target):
        best_move, best_dist = Move.STAY, self._manhattan(my_pos, target)
        for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nxt = (my_pos[0] + mv.value[0], my_pos[1] + mv.value[1])
            if self._is_valid(nxt):
                d = self._manhattan(nxt, target)
                if d < best_dist:
                    best_dist = d
                    best_move = mv
        return best_move

    def _find_best_frontier(self, my_pos):
        if self.internal_map is None: return None
        h, w = self.internal_map.shape
        best, best_score = None, -1
        for r in range(h):
            for c in range(w):
                if self.internal_map[r, c] != 0: continue
                has_unknown = any(self.internal_map[nr, nc] == -1 
                                  for nr, nc in [(r+1,c), (r-1,c), (r,c+1), (r,c-1)]
                                  if 0 <= nr < h and 0 <= nc < w)
                if has_unknown:
                    dist = self._manhattan(my_pos, (r, c))
                    if dist == 0: continue
                    score = 1.0 / dist
                    if score > best_score:
                        best_score, best = score, (r, c)
        return best

    def _random_valid_move(self, pos):
        # Pick a random valid move from current position
        moves = [mv for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT] if self._is_valid((pos[0]+mv.value[0], pos[1]+mv.value[1]))]
        return random.choice(moves) if moves else Move.STAY

    def _can_move_n(self, pos, move, n):
        r, c = pos
        dr, dc = move.value
        for i in range(1, n + 1):
            nr, nc = r + dr * i, c + dc * i
            if not self._is_valid((nr, nc)): return False
        return True

    def _is_valid(self, pos):
        if self.internal_map is None: return False
        h, w = self.internal_map.shape
        return 0 <= pos[0] < h and 0 <= pos[1] < w and self.internal_map[pos] != 1

    def _manhattan(self, a, b):
        # Calculate Manhattan distance between two points
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _bfs_distance(self, start, goal):
        if self.internal_map is None or start == goal: return 0
        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            curr, dist = queue.popleft()
            for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = (curr[0] + mv.value[0], curr[1] + mv.value[1])
                if self._is_valid(nxt) and nxt not in visited:
                    if nxt == goal: return dist + 1
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))
        return None

    def astar(self, start, goal):
        # A* Pathfinding algorithm to find the shortest path to the goal
        if self.internal_map is None or start == goal: return []
        frontier = [(0, 0, start, [], None)] # Priority queue of paths to explore
        visited = {}
        counter = 0
        while frontier:
            _, _, current, path, last_move = heappop(frontier)
            if current == goal: return path
            state_key = (current, last_move)
            g_score = len(path)
            if state_key in visited and visited[state_key] <= g_score: continue
            visited[state_key] = g_score
            for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = (current[0] + mv.value[0], current[1] + mv.value[1])
                if self._is_valid(nxt):
                    new_path = path + [mv]
                    turn_penalty = 0.1 if last_move is not None and mv != last_move else 0
                    g = len(new_path) + turn_penalty
                    h = self._manhattan(nxt, goal)
                    counter += 1
                    heappush(frontier, (g + h, counter, nxt, new_path, mv))
        return []


# ==================== GHOST AGENT ====================

class GhostAgent(BaseGhostAgent):
    CAPTURE_THRESHOLD = 2
    TIME_LIMIT = 0.75

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Ghost Minimax Agent"
        self.pacman_speed = int(kwargs.get("pacman_speed", 2))
        self.ghost_obs_radius = int(kwargs.get("ghost_obs_radius", 0))
        self.time_limit = self.TIME_LIMIT

        self.last_known_pacman = None
        self.turns_since_seen = 0
        self.history = deque(maxlen=6)
        self._last_step = -1
        self._bfs_cache = {}
        self._eval_cache = {}

    def step(self, map_state: np.ndarray, my_position: tuple,
             enemy_position: tuple, step_number: int) -> Move:
        self.start_time = time.time()
        self.map_state = map_state
        self.timeout_flag = False
        self._eval_cache = {}

        # Detect new game (step_number went back to 1 or decreased)
        if step_number <= 1 or step_number < self._last_step:
            self._reset_game_state()
        self._last_step = step_number

        if enemy_position is not None:
            self.last_known_pacman = enemy_position
            self.turns_since_seen = 0
        else:
            self.turns_since_seen += 1

        height, width = map_state.shape
        if self.last_known_pacman is None:
            # No Pacman info yet, assume it starts from the opposite side
            self.last_known_pacman = (height - 1, my_position[1])

        if enemy_position is not None:
            # Minimax search when Pacman is visible
            best_action = self._greedy_flee(my_position, enemy_position)
            for depth in range(1, 25):
                action, score = self._minimax(
                    depth=depth,
                    ghost_pos=my_position,
                    pac_pos=enemy_position,
                    alpha=-float('inf'),
                    beta=float('inf'),
                    is_max_player=True
                )
                if self.timeout_flag:
                    break
                if action is not None:
                    best_action = action
            self._record(my_position, best_action)
            return best_action
        else:
            # Pacman not visible, use scoring-based evasion
            move = self._evasion_move(my_position)
            self._record(my_position, move)
            return move

    def _reset_game_state(self):
        self.last_known_pacman = None
        self.turns_since_seen = 0
        self.history.clear()
        # BFS cache stays valid since map is static

    def _minimax(self, depth, ghost_pos, pac_pos, alpha, beta, is_max_player):
        if time.time() - self.start_time > self.time_limit:
            self.timeout_flag = True
            return None, 0

        # Check if captured (Manhattan distance)
        manhattan = abs(ghost_pos[0] - pac_pos[0]) + abs(ghost_pos[1] - pac_pos[1])
        if manhattan < self.CAPTURE_THRESHOLD:
            return None, -99999

        bfs_dist = self._d(ghost_pos, pac_pos)

        if depth == 0:
            # use cached evaluation
            return None, self._eval_cached(ghost_pos, pac_pos, bfs_dist)

        if is_max_player:   # Ghost maximises
            max_eval = -float('inf')
            best_move = Move.STAY

            ghost_moves = get_valid_moves(ghost_pos, self.map_state)
            ghost_moves.append((Move.STAY, ghost_pos))

            # Sort: furthest from Pacman first for better pruning
            ghost_moves.sort(key=lambda x: -self._d(x[1], pac_pos))

            for move, next_ghost_pos in ghost_moves:
                _, eval_score = self._minimax(depth - 1, next_ghost_pos, pac_pos,
                                              alpha, beta, False)
                if self.timeout_flag:
                    break
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return best_move, max_eval

        else:               # Pacman minimises
            min_eval = float('inf')
            pac_moves = self._pacman_straight_line_moves(pac_pos)

            # Sort: closest to Ghost first for better pruning
            pac_moves.sort(key=lambda x: self._d(x[1], ghost_pos))

            for _, next_pac_pos in pac_moves:
                _, eval_score = self._minimax(depth - 1, ghost_pos, next_pac_pos,
                                              alpha, beta, True)
                if self.timeout_flag:
                    break
                if eval_score < min_eval:
                    min_eval = eval_score
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return None, min_eval

    def _eval_cached(self, ghost_pos, pac_pos, bfs_dist):
        key = (ghost_pos, pac_pos)
        if key not in self._eval_cache:
            self._eval_cache[key] = self._evaluate_state(ghost_pos, pac_pos, bfs_dist)
        return self._eval_cache[key]

    def _evaluate_state(self, ghost_pos, pac_pos, bfs_dist):
        # Immediate capture check
        manhattan = abs(ghost_pos[0] - pac_pos[0]) + abs(ghost_pos[1] - pac_pos[1])
        if manhattan < self.CAPTURE_THRESHOLD:
            return -99999

        # Number of turns until Pacman can reach us
        effective_turns = math.ceil(bfs_dist / self.pacman_speed)

        if effective_turns <= 1:
            # Near-certain capture next turn
            return -50000 + bfs_dist * 50

        # Primary score: BFS distance from Pacman
        score = bfs_dist * 200

        # Urgency penalty: discourage staying close to Pacman
        if effective_turns <= 2:
            score -= 3000   # Pacman arrives in 2 turns - critical danger
        elif effective_turns <= 3:
            score -= 800    # Getting close - increase distance

        # Prefer positions not in line-of-sight with Pacman
        if not check_cross_visibility(ghost_pos, pac_pos, self.map_state):
            score += 600

        # Mobility: count reachable cells (prefer open areas, avoid dead ends)
        mobility = flood_fill_free_space(ghost_pos, self.map_state, limit=30)
        if mobility <= 1:
            score -= 15000   # completely trapped
        elif mobility <= 3:
            score -= 5000    # very tight corner
        elif mobility <= 6:
            score -= 1000    # restricted space
        elif mobility <= 15:
            score += mobility * 40
        else:
            score += 15 * 40 + (mobility - 15) * 10

        # Junction bonus / dead-end penalty
        exits = len(get_valid_moves(ghost_pos, self.map_state))
        if exits >= 3:
            score += 400   # intersection
        elif exits == 2:
            score += 50
        elif exits <= 1:
            score -= 3000  # dead end is dangerous near Pacman

        return score

    def _d(self, a, b):
        # Cached BFS distance from a to b
        if a not in self._bfs_cache:
            self._bfs_cache[a] = self._get_bfs_distance_map(a)
        return self._bfs_cache[a].get(b, 999)

    def _greedy_flee(self, my_pos, danger_pos):
        # Quick greedy fallback: maximise BFS distance, avoid dead-ends
        best_move = Move.STAY
        max_eval  = -float('inf')
        for move, next_pos in get_valid_moves(my_pos, self.map_state):
            dist     = self._d(next_pos, danger_pos)
            mobility = flood_fill_free_space(next_pos, self.map_state, limit=10)
            score    = dist * 10
            if mobility <= 2:
                score -= 1000
            if score > max_eval:
                max_eval  = score
                best_move = move
        return best_move

    def _pacman_straight_line_moves(self, pac_pos):
        reachable = {pac_pos}
        h, w = self.map_state.shape
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = pac_pos
            for _ in range(self.pacman_speed):
                r, c = r + dr, c + dc
                if 0 <= r < h and 0 <= c < w and self.map_state[r, c] != 1:
                    reachable.add((r, c))
                else:
                    break
        return [(None, pos) for pos in reachable]

    def _evasion_move(self, my_position):
        valid = get_valid_moves(my_position, self.map_state)
        if not valid:
            return Move.STAY

        pacman_dist_map = self._get_bfs_distance_map(self.last_known_pacman)

        best_score = -float('inf')
        best_move  = Move.STAY
        height, width = self.map_state.shape

        for move, next_pos in valid:
            nr, nc = next_pos

            # Primary: BFS distance from last-known Pacman (larger = safer)
            real_dist = pacman_dist_map.get(next_pos, 0)
            score = real_dist * 25
            if real_dist <= 2:   score -= 3000
            elif real_dist <= 4: score -= 800
            elif real_dist <= 6: score -= 100

            # Moving out of Pacman's line-of-sight is safer
            if not check_cross_visibility(next_pos, self.last_known_pacman, self.map_state):
                score += 400

            # Junction / dead-end: prefer cells with multiple exits
            exits = len(get_valid_moves(next_pos, self.map_state))
            if exits == 1:    score -= 600
            elif exits == 2:  score += 30
            elif exits >= 3:  score += 120

            # Anti-oscillation: penalize recently visited positions
            if next_pos in self.history:
                score -= 250

            if score > best_score:
                best_score = score
                best_move  = move

        return best_move

    def _bfs_first_step(self, start, goal):
        if start == goal:
            return None
        queue   = deque([(start, None)])
        visited = {start}
        while queue:
            curr, first = queue.popleft()
            for move, nxt in get_valid_moves(curr, self.map_state):
                if nxt not in visited:
                    step_first = move if first is None else first
                    if nxt == goal:
                        return step_first
                    visited.add(nxt)
                    queue.append((nxt, step_first))
        return None

    def _get_bfs_distance_map(self, start):
        distances = {start: 0}
        queue     = deque([start])
        while queue:
            curr = queue.popleft()
            d    = distances[curr]
            for _, npos in get_valid_moves(curr, self.map_state):
                if npos not in distances:
                    distances[npos] = d + 1
                    queue.append(npos)
        return distances

    def _record(self, my_pos, move):
        # Track recent positions
        dr, dc  = move.value
        new_pos = (my_pos[0] + dr, my_pos[1] + dc)
        height, width = self.map_state.shape
        if 0 <= new_pos[0] < height and 0 <= new_pos[1] < width:
            self.history.append(new_pos)

    def _bfs_path(self, start, target):
        if start == target:
            return []
        queue   = deque([(start, [])])
        visited = {start}
        while queue:
            curr, path = queue.popleft()
            if curr == target:
                return path
            for move, next_pos in get_valid_moves(curr, self.map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))
        return []