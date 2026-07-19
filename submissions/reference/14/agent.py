import sys
from pathlib import Path
import heapq

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np

import sys
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from collections import deque as Queue # For BFS queue
import heapq # For A* priority queue


class PacmanAgent(BasePacmanAgent):

    #  Pacman (Seeker) - fog setting will NOT be used in this lab
    #  Some kind of like A* pathfinding + Alpha-Beta Minimax
    
    def __init__(self, **kwargs):
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        super().__init__(**kwargs)

        # - self.visited = set()  # Track visited positions
        self.name = "TVS Pacman"
        
        # State Tracking
        self.prev_enemy_pos = None
        self.enemy_velocity = (0, 0)
        self.visited_cells = set()
        self.last_known_enemy_pos = None
    
        # Performance/Lookup caches
        self.valid_cells = set()
        self.neighbor_cache = {}
        self.astar_cache = {} # Store planned path
        self.initialized = False

    def _init_caches(self, map_state: np.ndarray):
        """Precompute traversable map structure once."""
        height, width = map_state.shape
        self.valid_cells = {
            (r, c) for r in range(height) for c in range(width) if map_state[r, c] == 0
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
        self.initialized = True
    
    ######################### Search Algorithm Implementation #########################

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

                if self._is_valid_position_fast(neighbor):
                    tentative_g_score = g_score[current_pos] + 1

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        parent[neighbor] = current_pos
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self.manhattan_distance(neighbor, goal)
                        heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))

        self.astar_cache[cache_key] = None  # No path found
        return None

            ######################################################################
            
        ############################### Prediction Seeker Move  ###################################
    def _alphabeta(self, pac_pos: tuple, ghost_pos: tuple, depth: int, alpha: float, beta: float, is_max: bool) -> float:
        """Shallow depth alphabeta adversarial minimax."""
        if depth == 0 or pac_pos == ghost_pos:
            return self._evaluate_state(pac_pos, ghost_pos)

        if is_max:
            #### SEEKER ####
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
            #### Hider ####
            min_val = float('inf')
            ghost_moves = self.neighbor_cache.get(ghost_pos, [])

            for neighbor, _ in ghost_moves:
                val = self._alphabeta(pac_pos, neighbor, depth - 1, alpha, beta, True)
                min_val = min(min_val, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return min_val
    ###################################################################################

    ############################### Choice making Move  ###################################
    def _evaluate_state(self, pac_pos: tuple, ghost_pos: tuple) -> float:
        """Evaluation function reflecting distance, capture status, and topological traps."""
        if pac_pos == ghost_pos:
            return 100000.0

        path = None
        path = self.A_star(pac_pos, ghost_pos)
        _dist = (len(path) - 1) if path else self.manhattan_distance(pac_pos, ghost_pos)

        # Calculate ghost mobility and trap structural score
        ghost_moves = self.neighbor_cache.get(ghost_pos, [])
        ghost_mobility = len(ghost_moves)

        # Close distance (the closer the better)
        distance_penalty = _dist * 50.0
        
        close_bonus = 0.0
        if _dist <= 2:
            close_bonus = (3 - _dist) * 2000.0
        elif _dist <= 4:
            close_bonus = (5 - _dist) * 200.0
        
        trap_score = (4 - ghost_mobility) * 25.0
        score = - distance_penalty + trap_score * 0.9 + close_bonus
        return score
    
    ###################################################################################

    ############################### Helper methods  ###################################

    def manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _choose_action_and_step_number(self, path):
        if len(path) < 2:
            return (Move.STAY, 1)

        if len(path) >= 3:
            pos0 = path[0]
            pos1 = path[1]
            pos2 = path[2]

            mov1 = self._get_move_direction(pos0, pos1)
            mov2 = self._get_move_direction(pos1, pos2)

            if mov1 == mov2:
                path.pop(0)
                path.pop(0)
                return (mov1, 2)
        
        mov = self._get_move_direction(path[0], path[1])
        path.pop(0)
        return (mov, 1)

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

    def _choose_action(self, pos: tuple, moves, map_state: np.ndarray, desired_steps: int):
        for move in moves:
            max_steps = min(self.pacman_speed, max(1, desired_steps))
            steps = self._max_valid_steps(pos, move, map_state, max_steps)
            if steps > 0:
                return (move, steps)
        return None

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:
        steps = 0
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid for at least one step."""
        return self._max_valid_steps(pos, move, map_state, 1) == 1
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0
    
    def _is_valid_position_fast(self, pos: tuple) -> bool:
        return pos in self.valid_cells
    
    ############################### H_M  ###################################

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        
        """Main step strategy wrapper."""
        if not self.initialized:
            self._init_caches(map_state)

        #In case smth gone wrong
        if enemy_position is None:
            return (Move.STAY, 1)
        
        target = enemy_position
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
    Ghost (Hider) Agent - Goal: Avoid being caught
    Optimized: Minimax + Alpha-Beta Pruning + Territory-Control Evaluation
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.name = "Territory Minimax Ghost"
        
        # State Tracking & Navigation Cache
        self.last_known_enemy_pos = None
        self.valid_cells = set()
        self.neighbor_cache = {}
        self.dead_ends = set()
        self.initialized = False

    def _init_caches(self, map_state: np.ndarray):
        """Precompute navigation graph, adjacency lists, and dead-ends."""
        height, width = map_state.shape
        self.valid_cells = {
            (r, c) for r in range(height) for c in range(width) if map_state[r, c] == 0
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

        # Iteratively identify structural dead-ends (corridors with 1 escape route)
        degrees = {cell: len(neighs) for cell, neighs in self.neighbor_cache.items()}
        dead_ends = set()
        queue = [cell for cell, deg in degrees.items() if deg <= 1]
        while queue:
            curr = queue.pop(0)
            dead_ends.add(curr)
            for neighbor, _ in self.neighbor_cache.get(curr, []):
                if neighbor not in dead_ends:
                    valid_neighs = [n for n, _ in self.neighbor_cache[neighbor] if n not in dead_ends]
                    if len(valid_neighs) <= 1:
                        queue.append(neighbor)
        self.dead_ends = dead_ends
        self.initialized = True

    def _get_distance(self, start: tuple, goal: tuple) -> int:
        """Fast BFS shortest-path distance."""
        if start == goal:
            return 0
        queue = [(start, 0)]
        visited = {start}
        while queue:
            curr, dist = queue.pop(0)
            if curr == goal:
                return dist
            for neighbor, _ in self.neighbor_cache.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return 999

    def _analyze_territory(self, ghost_pos: tuple, pac_pos: tuple) -> tuple:
        """Perform a custom flood-fill to get reachable territory and branching metric."""
        queue = [ghost_pos]
        visited = {ghost_pos}
        total_branching = 0
        
        while queue:
            curr = queue.pop(0)
            neighbors = self.neighbor_cache.get(curr, [])
            total_branching += len(neighbors)
            for neighbor, _ in neighbors:
                if neighbor != pac_pos and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
        territory_size = len(visited)
        branching_factor = total_branching / max(1, territory_size)
        return territory_size, branching_factor

    def _evaluate_state(self, ghost_pos: tuple, pac_pos: tuple) -> float:
        """Robust multi-criteria territory escape evaluation."""
        if ghost_pos == pac_pos:
            return -10000.0

        pac_dist = self._get_distance(ghost_pos, pac_pos)
        territory_size, branching_factor = self._analyze_territory(ghost_pos, pac_pos)
        
        dead_end_penalty = 1.0 if ghost_pos in self.dead_ends else 0.0
        trap_risk = 1.0 if pac_dist <= 3 else 0.0

        # Weighted survival metric formula
        score = (
            territory_size * 6.0
            + branching_factor * 10.0
            + pac_dist * 4.0
            - dead_end_penalty * 20.0
            - trap_risk * 25.0
        )
        return score

    def _alphabeta(self, ghost_pos: tuple, pac_pos: tuple, depth: int, alpha: float, beta: float, is_max: bool) -> float:
        """Alpha-Beta Minimax search optimized for high survival evasion."""
        if depth == 0 or ghost_pos == pac_pos:
            return self._evaluate_state(ghost_pos, pac_pos)

        if is_max:
            # Ghost Turn - maximize escape paths and distance
            max_val = -float('inf')
            candidates = []
            for neighbor, move in self.neighbor_cache.get(ghost_pos, []):
                dist = abs(neighbor[0] - pac_pos[0]) + abs(neighbor[1] - pac_pos[1])
                candidates.append((dist, neighbor, move))
            # Sort candidates to check the most promising moves away from Pacman first
            candidates.sort(key=lambda x: x[0], reverse=True)
            candidates.append((abs(ghost_pos[0] - pac_pos[0]) + abs(ghost_pos[1] - pac_pos[1]), ghost_pos, Move.STAY))

            for _, next_pos, _ in candidates:
                val = self._alphabeta(next_pos, pac_pos, depth - 1, alpha, beta, False)
                max_val = max(max_val, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return max_val
        else:
            # Pacman Turn - minimize Ghost's survival options
            min_val = float('inf')
            pac_moves = []
            # Calculate all potential target positions Pacman could move to using its speed
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
                    if not valid and next_pos == pac_pos:
                        continue
                    dist = abs(next_pos[0] - ghost_pos[0]) + abs(next_pos[1] - ghost_pos[1])
                    pac_moves.append((dist, next_pos))
                    
            # Sort: Pacman chooses paths minimizing the distance to Ghost
            pac_moves.sort(key=lambda x: x[0])
            for _, next_pos in pac_moves[:4]:  # Prune search space branching factor
                val = self._alphabeta(ghost_pos, next_pos, depth - 1, alpha, beta, True)
                min_val = min(min_val, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return min_val

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """Determine optimal survival action using Minimax with dynamic depth control."""
        if not self.initialized:
            self._init_caches(map_state)
            
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            
        threat = enemy_position or self.last_known_enemy_pos

        # Fallback Behavior: patrol/maximize space if threat is unseen
        if threat is None:
            best_move = Move.STAY
            best_score = -float('inf')
            for neighbor, move in self.neighbor_cache.get(my_position, []):
                score = len(self.neighbor_cache.get(neighbor, []))
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_move

        best_move = Move.STAY
        best_score = -float('inf')

        # Generate, rank and filter Ghost candidate moves
        candidates = []
        for neighbor, move in self.neighbor_cache.get(my_position, []):
            dist = abs(neighbor[0] - threat[0]) + abs(neighbor[1] - threat[1])
            candidates.append((dist, neighbor, move))
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates.append((abs(my_position[0] - threat[0]) + abs(my_position[1] - threat[1]), my_position, Move.STAY))

        # Shallow 4-depth lookahead evaluation (2 moves for Ghost, 2 moves for Pacman)
        for _, next_pos, move in candidates:
            score = self._alphabeta(next_pos, threat, depth=4, alpha=-float('inf'), beta=float('inf'), is_max=False)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move
