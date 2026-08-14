import sys
import heapq
import numpy as np
from pathlib import Path
from collections import deque as Queue  # For BFS queue

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


class PacmanAgent(BasePacmanAgent):
    """
    Upgraded Pacman (Seeker) with Frontier Exploration & Memory for Blind Setting.
    """
    def __init__(self, **kwargs):
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        super().__init__(**kwargs)
        self.name = "TVS Pacman Blind"
        
        # Memory & Exploration state tracking
        self.last_known_enemy_pos = None
        self.global_map = None  # Tracks explored paths (0: empty, 1: wall, -1: unexplored)
        
        # Performance/Lookup caches
        self.valid_cells = set()
        self.neighbor_cache = {}
        self.astar_cache = {}
        self.initialized = False

    def _init_caches(self, map_state: np.ndarray):
        """Precompute the entire traversable maze structure at step 1 since walls are always visible."""
        height, width = map_state.shape
        # Walls are always 1, any cell that is not 1 is a traversable path (or will be)
        self.valid_cells = {
            (r, c) for r in range(height) for c in range(width) if map_state[r, c] != 1
        }
        self.neighbor_cache = {}
        for r, c in self.valid_cells:
            neighbors = []
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nr, nc = r + dr, c + dc
                if (nr, nc) in self.valid_cells:
                    neighbors.append(((nr, nc), move))
            self.neighbor_cache[(r, c)] = neighbors
            
        # Initialize the global map tracking: walls are 1, everything else starts as -1 (unexplored)
        self.global_map = np.where(map_state == 1, 1, -1)
        self.initialized = True
        
    def _find_nearest_unexplored(self, start: tuple) -> tuple:
        """Run BFS to find the closest unexplored empty path (-1) on the global map."""
        queue = Queue([start])
        visited = {start}
        while queue:
            curr = queue.popleft()
            if self.global_map[curr[0], curr[1]] == -1:
                return curr
            for neighbor, _ in self.neighbor_cache.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return None

    def _explore_random(self, pos: tuple):
        """Emergency random walk if exploration target is unreachable."""
        import random
        valid_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            nxt = (pos[0] + dr, pos[1] + dc)
            if nxt in self.valid_cells:
                valid_moves.append(move)
        if valid_moves:
            return (random.choice(valid_moves), 1)
        return (Move.STAY, 1)

    def A_star(self, start: tuple, goal: tuple):
        if start == goal:
            return []
        
        cache_key = (start, goal)
        if cache_key in self.astar_cache:
            return self.astar_cache[cache_key]
        
        open_set = []
        heapq.heappush(open_set, (0 + self.manhattan_distance(start, goal), 0, start))

        g_score = {start: 0}
        parent = {}
        moves = [Move.UP.value, Move.DOWN.value, Move.LEFT.value, Move.RIGHT.value]

        while open_set:
            current_f, current_g, current_pos = heapq.heappop(open_set)

            if current_pos == goal: # Path Found
                path = []
                curr = goal
                while curr != start:
                    path.append(curr)
                    curr = parent[curr]
                path.append(start)
                path.reverse()
                self.astar_cache[cache_key] = path
                return path

            if current_g > g_score.get(current_pos, float('inf')):
                continue
            
            for delta_row, delta_col in moves:
                neighbor = (current_pos[0] + delta_row, current_pos[1] + delta_col)

                if neighbor in self.valid_cells:
                    tentative_g_score = g_score[current_pos] + 1

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        parent[neighbor] = current_pos
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self.manhattan_distance(neighbor, goal)
                        heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))

        self.astar_cache[cache_key] = None  # No path found
        return None

    def _alphabeta(self, pac_pos: tuple, ghost_pos: tuple, depth: int, alpha: float, beta: float, is_max: bool) -> float:
        """Shallow depth alphabeta adversarial minimax."""
        if depth == 0 or pac_pos == ghost_pos:
            return self._evaluate_state(pac_pos, ghost_pos)

        if is_max:
            max_val = -float('inf')
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                for steps in range(1, self.pacman_speed + 1):
                    next_pos = pac_pos
                    valid = True
                    for _ in range(steps):
                        dr, dc = move.value
                        candidate = (next_pos[0] + dr, next_pos[1] + dc)
                        if candidate in self.valid_cells:
                            next_pos = candidate
                        else:
                            valid = False
                            break
                    if not valid:
                        continue
                    val = self._alphabeta(next_pos, ghost_pos, depth - 1, alpha, beta, False)
                    max_val = max(max_val, val)
                    alpha = max(alpha, val)
                    if beta <= alpha:
                        break
            return max_val
        else:
            min_val = float('inf')
            ghost_moves = self.neighbor_cache.get(ghost_pos, [])
            for neighbor, _ in ghost_moves:
                val = self._alphabeta(pac_pos, neighbor, depth - 1, alpha, beta, True)
                min_val = min(min_val, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return min_val

    def _evaluate_state(self, pac_pos: tuple, ghost_pos: tuple) -> float:
        if pac_pos == ghost_pos:
            return 100000.0

        path = self.A_star(pac_pos, ghost_pos)
        _dist = (len(path) - 1) if path else self.manhattan_distance(pac_pos, ghost_pos)

        ghost_moves = self.neighbor_cache.get(ghost_pos, [])
        ghost_mobility = len(ghost_moves)

        distance_penalty = _dist * 50.0
        close_bonus = 0.0
        if _dist <= 2:
            close_bonus = (3 - _dist) * 2000.0
        elif _dist <= 4:
            close_bonus = (5 - _dist) * 200.0
        
        trap_score = (4 - ghost_mobility) * 25.0
        score = - distance_penalty + trap_score * 0.9 + close_bonus
        return score

    def manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_move_direction(self, from_pos: tuple, to_pos: tuple) -> Move:
        row_diff = to_pos[0] - from_pos[0]
        col_diff = to_pos[1] - from_pos[1]
        if row_diff == -1 and col_diff == 0:
            return Move.UP
        if row_diff == 1 and col_diff == 0:
            return Move.DOWN
        if row_diff == 0 and col_diff == -1:
            return Move.LEFT
        if row_diff == 0 and col_diff == 1:
            return Move.RIGHT
        return Move.STAY

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        
        if not self.initialized:
            self._init_caches(map_state)

        # Clear temporal path cache to keep search updated
        self.astar_cache.clear()
        
        # Update global map with newly explored areas
        self.global_map[map_state == 0] = 0
        my_position = tuple(my_position)

        if enemy_position is not None:
            self.last_known_enemy_pos = tuple(enemy_position)
            target = self.last_known_enemy_pos
        else:
            target = self.last_known_enemy_pos

        # If we reached the target but the enemy is no longer there, clear memory
        if target is not None and my_position == target:
            self.last_known_enemy_pos = None
            target = None

        # --- EXPLORATION & TARGET RECOVERY PHASE ---
        if target is None:
            # Active Search: BFS to find the closest unexplored empty path (-1)
            explore_target = self._find_nearest_unexplored(my_position)
            if explore_target is not None:
                path = self.A_star(my_position, explore_target)
                if path and len(path) > 1:
                    next_pos = path[1]
                    move = self._get_move_direction(my_position, next_pos)
                    steps = 1
                    if len(path) >= 3 and self.pacman_speed >= 2:
                        next_next_pos = path[2]
                        if move == self._get_move_direction(next_pos, next_next_pos):
                            steps = 2
                    return (move, steps)
            return self._explore_random(my_position)

        # If enemy is not currently visible but we have their memory: Sprint directly to the memory spot
        if enemy_position is None and target is not None:
            path = self.A_star(my_position, target)
            if path and len(path) > 1:
                next_pos = path[1]
                move = self._get_move_direction(my_position, next_pos)
                steps = 1
                if len(path) >= 3 and self.pacman_speed >= 2:
                    next_next_pos = path[2]
                    if move == self._get_move_direction(next_pos, next_next_pos):
                        steps = 2
                return (move, steps)
            return self._explore_random(my_position)

        # --- ACTIVE ENGAGEMENT PHASE (Enemy is currently visible) ---
        best_action = (Move.STAY, 1)
        best_score = -float('inf')

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            for steps in range(1, self.pacman_speed + 1):
                next_pos = my_position
                valid = True
                
                for _ in range(steps):
                    dr, dc = move.value
                    candidate = (next_pos[0] + dr, next_pos[1] + dc)
                    if candidate in self.valid_cells:
                        next_pos = candidate
                    else:
                        valid = False
                        break
                        
                if not valid:
                    continue

                if next_pos == target:
                    return (move, steps)
                
                score = self._alphabeta(next_pos, target, depth=3, alpha=-float('inf'), beta=float('inf'), is_max=False)

                if score > best_score or (score == best_score and steps > best_action[1] and best_action[0] == move):
                    best_score = score
                    best_action = (move, steps)
                    
        return best_action
    


class GhostAgent(BaseGhostAgent):
    """
    1. Prefers Right Pocket (5, 12) unless Left Pocket (5, 8) is strictly closer 
       by more than 2 steps (dist_left + 2 < dist_right).
    2. Runs pure BFS to guarantee shortest distance path without lower-map detours.
    3. Adapts BFS move preferences based on spawn row (prioritizing UP when spawned below Row 5).
    4. Stays permanently at the chosen pocket once reached.
    """

    TARGET_ROW = 5
    LEFT_POCKET = (5, 8)
    RIGHT_POCKET = (5, 12)
    
    # Margin of tolerance: Right pocket is chosen unless Left is > 2 tiles closer
    RIGHT_BIAS_MARGIN = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Right-Biased Pocket Ghost v2.1"

        # Navigation state precomputed on step 1
        self.target_pocket = None
        self.planned_moves = None
        self.current_move_idx = 0

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple, 
             step_number: int) -> Move:
        """
        Executes one environment tick for the Ghost Agent.
        
        Ignores enemy_position. Performs single-pass dual-path calculation on step 1,
        follows the optimized path to the chosen pocket, and stays forever.
        """
        # Step 1: Initialize target pocket choice and precompute path
        if self.planned_moves is None:
            self.planned_moves = self._plan_route_to_best_pocket(my_position, map_state)

        # Reached hiding pocket or finished path -> stay forever
        if self.current_move_idx >= len(self.planned_moves):
            return Move.STAY

        # Follow planned path step-by-step
        next_move = self.planned_moves[self.current_move_idx]
        self.current_move_idx += 1
        return next_move

    def _plan_route_to_best_pocket(self, start_pos: tuple, map_state: np.ndarray) -> list:
        """
        Calculates shortest paths to both pockets using standard BFS.
        Applies a bias toward the Right Pocket unless the Left Pocket is strictly closer
        by more than RIGHT_BIAS_MARGIN steps.
        """
        path_left = self._bfs_shortest_path(start_pos, self.LEFT_POCKET, map_state)
        path_right = self._bfs_shortest_path(start_pos, self.RIGHT_POCKET, map_state)

        dist_left = len(path_left)
        dist_right = len(path_right)

        # Only select Left Pocket if it is strictly closer by more than RIGHT_BIAS_MARGIN (2 tiles)
        if dist_left + self.RIGHT_BIAS_MARGIN < dist_right:
            self.target_pocket = self.LEFT_POCKET
            return path_left
        else:
            # Commit 100% to Right Pocket when equal, closer, or up to 2 steps further
            self.target_pocket = self.RIGHT_POCKET
            return path_right

    def _bfs_shortest_path(self, start_pos: tuple, target_pos: tuple, map_state: np.ndarray) -> list:
        """
        Runs pure unweighted BFS to find the shortest path from start_pos to target_pos.
        
        Dynamically adjusts neighbor expansion order based on spawn row:
        - Spawned below Row 5: Prioritizes UP early, and RIGHT over LEFT.
        - Spawned at or above Row 5: Prioritizes RIGHT/LEFT and UP before DOWN
          to delay downward moves into Pacman's threat zone.
        """
        if start_pos == target_pos:
            return []

        spawn_row = start_pos[0]

        # Determine directional search order based on spawn elevation relative to Row 5
        if spawn_row > self.TARGET_ROW:
            # Below target: Move UP early, prefer RIGHT over LEFT
            directions = [
                ((-1, 0), Move.UP),
                ((0, 1), Move.RIGHT),
                ((0, -1), Move.LEFT),
                ((1, 0), Move.DOWN)
            ]
        else:
            # At or above target: Move RIGHT/LEFT first, delay DOWN as late as possible
            directions = [
                ((0, 1), Move.RIGHT),
                ((0, -1), Move.LEFT),
                ((-1, 0), Move.UP),
                ((1, 0), Move.DOWN)
            ]

        queue = Queue([start_pos])
        visited = {start_pos}
        parent = {}  # Maps child_pos -> (parent_pos, Move)

        found = False
        while queue:
            curr_pos = queue.popleft()

            if curr_pos == target_pos:
                found = True
                break

            for (dr, dc), move in directions:
                next_pos = (curr_pos[0] + dr, curr_pos[1] + dc)

                if self._is_valid_cell(next_pos, map_state) and next_pos not in visited:
                    visited.add(next_pos)
                    parent[next_pos] = (curr_pos, move)
                    queue.append(next_pos)

        if not found:
            return []

        # Reconstruct sequence of moves from target back to start
        path_moves = []
        curr = target_pos
        while curr != start_pos:
            prev_pos, move = parent[curr]
            path_moves.append(move)
            curr = prev_pos

        path_moves.reverse()
        return path_moves

    def _is_valid_cell(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Checks if a position is within grid boundaries and non-wall."""
        r, c = pos
        rows, cols = map_state.shape
        if 0 <= r < rows and 0 <= c < cols:
            return map_state[r, c] != 1
        return False