import heapq
import numpy as np
from collections import deque
from typing import Tuple, Optional
from environment import Move
from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from functools import wraps


class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "A* Pacman"

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int):
        """
        Decide the next move.

        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Ghost's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)

        Returns:
            Move or (Move, steps): Direction to move (optionally with step count)
        """
        try:
            path = self.a_star(my_position, enemy_position, map_state)

            move = Move.STAY
            step = 1

            if len(path) > 1:
                step1 = path[1]

                if step1[0] == my_position[0]:
                    if step1[1] < my_position[1]:
                        move = Move.LEFT
                    elif step1[1] > my_position[1]:
                        move = Move.RIGHT
                elif step1[1] == my_position[1]:
                    if step1[0] < my_position[0]:
                        move = Move.UP
                    elif step1[0] > my_position[0]:
                        move = Move.DOWN

                if len(path) > 2:
                    step2 = path[2]
                    if (step1[0] == my_position[0] and step2[0] == step1[0]) or (step1[1] == my_position[1] and step2[1] == step1[1]):
                        step = 2

            return (move, step)
        except Exception as e:
            # Fallback
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                next_pos = self._apply_move(my_position, move)
                if self._is_valid_position(next_pos, map_state):
                    return (move, 1)

            return (Move.STAY, 1)


    # A* PATHFINDING ALGORITHM
    def a_star(self, start_pos: tuple, end_pos: tuple, map_state: np.ndarray):
        frontier_heap = []  # Priority queue (heapq)
        frontier = set()  # Sort of a tracker for the frontier. Needed because we use lazy deletion when updating the heapq.
        explored = set()

        # Track parents and costs using dictionaries (i don't know if we're allowed to add a Node class)
        parent = {start_pos: None}
        g_cost = {start_pos: 0}
        h_cost = {start_pos: self._manhattan_distance(start_pos, end_pos)}
        f_cost = {start_pos: g_cost[start_pos] + h_cost[start_pos]}

        # Push start pos into frontier. Use tuple for priority order (f-cost -> h-cost -> coordinate itself as fallback)
        heapq.heappush(frontier_heap, (f_cost[start_pos], h_cost[start_pos], start_pos))
        frontier.add(start_pos)

        # Loop through frontier
        iterations = 0
        while frontier:
            current_node = heapq.heappop(frontier_heap)[2]

            # Handle old duplicates left behind by lazy deletion.
            if current_node not in frontier:
                continue

            # Found target or reached max iterations
            if current_node == end_pos or iterations >= 128:
                path = []
                while current_node in parent:
                    path.append(current_node)
                    current_node = parent[current_node]
                path.reverse()
                return path

            # Move node to explored
            explored.add(current_node)
            frontier.remove(current_node)

            # Process neighbors
            neighbors = self._get_neighbors(current_node, map_state)
            for neighbor in neighbors:
                if neighbor in explored:
                    continue

                new_g_cost = g_cost[current_node] + 1
                if neighbor not in frontier:
                    h_cost[neighbor] = self._manhattan_distance(neighbor, end_pos)
                    frontier.add(neighbor)
                    g_cost[neighbor] = 1024 # Placeholder g-cost for the second if

                if new_g_cost < g_cost[neighbor]:
                    parent[neighbor] = current_node
                    g_cost[neighbor] = new_g_cost
                    f_cost[neighbor] = g_cost[neighbor] + h_cost[neighbor]
                    heapq.heappush(frontier_heap, (f_cost[neighbor], h_cost[neighbor], neighbor))

            iterations += 1

        # If no path is found
        print("PACMAN A* FAILED")
        return []

    # Helper methods
    def _apply_move(self, pos, move):
        """Apply a move to a position, return new position."""
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)

    def _get_neighbors(self, pos, map_state):
        """Get all valid neighboring positions and their moves."""
        neighbors = []

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append(next_pos)

        return neighbors

    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape

        if row < 0 or row >= height or col < 0 or col >= width:
            return False

        return map_state[row, col] == 0

class GhostAgent(BaseGhostAgent):
    """
    Uses BFS for instant distance calculations, Minimax for perfect 
    2-step Pacman prediction, and Topology to avoid distant dead ends.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_depth = 4 
        self.map_shape = None
        self.last_pos = None    

    def _get_pacman_distances(self, pacman_pos: Tuple[int, int], map_state: np.ndarray) -> np.ndarray:
        """Precomputes EXACT maze distances from Pacman to EVERY empty tile on the map."""
        h, w = self.map_shape
        dist = np.full((h, w), 9999, dtype=int)
        dist[pacman_pos[0], pacman_pos[1]] = 0
        queue = deque([pacman_pos])
        
        while queue:
            curr_r, curr_c = queue.popleft()
            d = dist[curr_r, curr_c]
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nr, nc = curr_r + move.value[0], curr_c + move.value[1]
                if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] == 0:
                    if dist[nr, nc] == 9999:
                        dist[nr, nc] = d + 1
                        queue.append((nr, nc))
        return dist

    def _get_topology_score(self, pos: Tuple[int, int], map_state: np.ndarray) -> float:
        """Evaluates the shape of the current tile to encourage turning and avoid traps."""
        valid_moves = 0
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            r, c = pos[0] + move.value[0], pos[1] + move.value[1]
            if 0 <= r < self.map_shape[0] and 0 <= c < self.map_shape[1] and map_state[r, c] == 0:
                valid_moves += 1
                
        if valid_moves >= 3:
            return 20  # Bonus for intersections
        elif valid_moves <= 1:
            return -100 # Huge penalty for dead ends
        return 0

    def _minimax(self, ghost_pos, pacman_pos, depth, is_maximizing, alpha, beta, map_state, dist_matrix):
        dist = dist_matrix[ghost_pos[0], ghost_pos[1]]
        sim_dist = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
        
        # Check if the SIMULATED Pacman caught us
        # Minus (depth * 1000). Dying immediately (Depth 4) gives -104,000. 
        # Dying later (Depth 2) gives -102,000. 
        # It will prefer to survive
        if sim_dist <= 1:
            return -99999 - (depth * 1000) + sim_dist
            
        # Base case: Reached depth limit. 
        if depth == 0:
            return (dist * 10) + self._get_topology_score(ghost_pos, map_state)
            
        if is_maximizing: 
            max_eval = -float('inf')
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                r, c = ghost_pos[0] + move.value[0], ghost_pos[1] + move.value[1]
                if 0 <= r < self.map_shape[0] and 0 <= c < self.map_shape[1] and map_state[r, c] == 0:
                    eval_score = self._minimax((r, c), pacman_pos, depth - 1, False, alpha, beta, map_state, dist_matrix)
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha: 
                        break
            return max_eval if max_eval != -float('inf') else -99999
            
        else:
            min_eval = float('inf')
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                r1, c1 = pacman_pos[0] + move.value[0], pacman_pos[1] + move.value[1]
                if 0 <= r1 < self.map_shape[0] and 0 <= c1 < self.map_shape[1] and map_state[r1, c1] == 0:
                    
                    # Pacman takes 1 step 
                    eval_score1 = self._minimax(ghost_pos, (r1, c1), depth - 1, True, alpha, beta, map_state, dist_matrix)
                    min_eval = min(min_eval, eval_score1)
                    
                    # Pacman takes 2 steps in a straight line
                    r2, c2 = r1 + move.value[0], c1 + move.value[1]
                    if 0 <= r2 < self.map_shape[0] and 0 <= c2 < self.map_shape[1] and map_state[r2, c2] == 0:
                        eval_score2 = self._minimax(ghost_pos, (r2, c2), depth - 1, True, alpha, beta, map_state, dist_matrix)
                        min_eval = min(min_eval, eval_score2)
                        
                beta = min(beta, min_eval)
                if beta <= alpha: 
                    break
            return min_eval

    def step(self, map_state: np.ndarray, 
             my_position: Tuple[int, int], 
             enemy_position: Optional[Tuple[int, int]],
             step_number: int) -> Move:
             
        if enemy_position is None:
            return Move.STAY
            
        self.map_shape = map_state.shape
        dist_matrix = self._get_pacman_distances(enemy_position, map_state)
            
        best_move = Move.STAY
        best_score = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            r, c = my_position[0] + move.value[0], my_position[1] + move.value[1]
            if 0 <= r < self.map_shape[0] and 0 <= c < self.map_shape[1] and map_state[r, c] == 0:
                
                score = self._minimax((r, c), enemy_position, self.max_depth - 1, False, alpha, beta, map_state, dist_matrix)
                
                if self.last_pos is not None and (r, c) == self.last_pos:
                    score -= 0.1 
                
                if score > best_score:
                    best_score = score
                    best_move = move
                    
                alpha = max(alpha, best_score)
                
        self.last_pos = my_position
        return best_move