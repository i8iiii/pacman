# Nguyễn Hoàng Tiến - 24127308
# Nguyễn Trần Phương Thuý - 24127248
# Trần Hoàng Khánh My - 24127086

import sys
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import random
from collections import deque

class PacmanAgent(BasePacmanAgent):
    """
    Minimax Pacman with BFS evaluation
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Pacman agent.
        Students can set up any data structures they need here.
        """
        super().__init__(**kwargs)
        self.name = "Minimax Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
        self.valid_tiles = {}
        self.map_BFS_cache = {}
        
    def get_BFS_map(self, start: tuple):
        """Retrieves or calculates BFS map from start position"""
        if start in self.map_BFS_cache:
            return self.map_BFS_cache[start]
        
        distances = {start: 0}
        parent = {start: None}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                x, y = move.value
                next = (current[0] + x, current[1] + y)
                if self.is_valid_position_precomputed(next) and next not in distances:
                    distances[next] = distances[current] + 1
                    parent[next] = (current, move)
                    queue.append(next)
        self.map_BFS_cache[start] = (distances, parent)
        return self.map_BFS_cache[start]
    
    def minimax(self, ghost_position: tuple, pacman_position: tuple, map_state: np.ndarray, depth: int, alpha: int, beta: int, pacman_turn: bool):
        # base case
        if ghost_position == pacman_position:
           return 0 - depth
        if depth == 0: 
            distances, parent = self.get_BFS_map(pacman_position)
            if ghost_position not in distances:
                return float('inf')
            
            # reconstruct path
            path = []
            current = ghost_position
            while current != pacman_position:
                previous, move_taken = parent[current]
                path.append(move_taken)
                current = previous
            path.reverse() 
            # calculate ticks
            ticks = self.pacman_tick(path)
            return (ticks * 10) + len(path) # use path length as tiebreaker
        
        # minimizing for pacman
        if pacman_turn:
            min_eval = float('inf')
            pacman_next_positions = self.get_sprint_options(pacman_position)
            for _, _, next_pacman_pos in pacman_next_positions:
                eval = self.minimax(ghost_position, next_pacman_pos, map_state, depth - 1, alpha, beta, False)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
            
        # maximizing for ghost
        else:
            max_eval = -float('inf')
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                next_ghost_pos = (ghost_position[0] + move.value[0], ghost_position[1] + move.value[1])
                
                if self.is_valid_position_precomputed(next_ghost_pos):
                    eval = self.minimax(next_ghost_pos, pacman_position, map_state, depth - 1, alpha, beta, True)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            return max_eval
        
    def precompute_valid_tiles(self, map_state: np.ndarray):
        """Precompute valid tiles for quick access during minimax."""
        self.valid_tiles = {}
        height, width = map_state.shape
        for row in range(height):
            for col in range(width):
                if map_state[row, col] == 0:
                    self.valid_tiles[(row, col)] = True
    
    def is_valid_position_precomputed(self, pos: tuple):
        """Check if a position is valid using precomputed tiles."""
        return pos in self.valid_tiles
    
    def pacman_tick(self, path: list) -> int:
        if not path:
            return 0
        ticks = 0
        current_move = path[0]
        steps = 0
        
        for move in path:
            if move == current_move and steps < self.pacman_speed:
                steps += 1
            else:
                ticks += 1
                current_move = move
                steps = 1
        if steps > 0:
            ticks += 1
        return ticks
    
    def get_sprint_options(self, pos: tuple):
        """
        Generates all valid moves and step counts for Pacman.
        Returns a list of tuples: (Move, step_count, landing_position)
        """
        options = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            delta_row, delta_col = move.value
            current_pos = pos        
            for steps in range(1, self.pacman_speed + 1):
                next_pos = (current_pos[0] + delta_row, current_pos[1] + delta_col)
                # Stop projecting if we hit a wall
                if not self.is_valid_position_precomputed(next_pos):
                    break 
                current_pos = next_pos
                options.append((move, steps, current_pos))
                
        return options
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        # Use current sighting, fallback to last known, or explore
        target = enemy_position or self.last_known_enemy_pos
        
        if target is None:
            # No information about enemy - explore randomly
            return self._explore(my_position)
        
        if len(self.valid_tiles) == 0:
            self.precompute_valid_tiles(map_state)
        
        best_move = Move.STAY
        best_steps = 1
        min_eval = float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        possible_moves = self.get_sprint_options(my_position)
        for move, steps, landing_pos in possible_moves:
            eval = self.minimax(enemy_position, landing_pos, map_state, 3, alpha, beta, False)
            if eval < min_eval:
                min_eval = eval
                best_move = move
                best_steps = steps
            beta = min(beta, eval)
            
        if best_steps > 1:
            return (best_move, best_steps)
        return (best_move, 1)

    def _explore(self, my_position: tuple):
        """Random exploration when enemy position is unknown."""
        all_moves = self.get_sprint_options(my_position)
        if all_moves:
            move, steps, _ = random.choice(all_moves)
            if steps > 1:
                return (move, steps)
            return (move, 1)
        return (Move.STAY, 1)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0

class GhostAgent(BaseGhostAgent):
    
    def __init__(self, **kwargs):
        """
        Initialize the Ghost agent.
        """
        super().__init__(**kwargs)
        self.name = "Minimax Ghost"
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.valid_tiles = {}
        self.map_BFS_cache = {}
        self.dead_ends = set()
        
    def get_BFS_map(self, start: tuple):
        """Retrieves or calculates BFS map from start positiom"""
        if start in self.map_BFS_cache:
            return self.map_BFS_cache[start]
        
        distances = {start: 0}
        parent = {start: None}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                x, y = move.value
                next = (current[0] + x, current[1] + y)
                if self.is_valid_position_precomputed(next) and next not in distances:
                    distances[next] = distances[current] + 1
                    parent[next] = (current, move)
                    queue.append(next)
        self.map_BFS_cache[start] = (distances, parent)
        return self.map_BFS_cache[start]
    
    def minimax(self, ghost_position: tuple, pacman_position: tuple, map_state: np.ndarray, depth: int, alpha: int, beta: int, ghost_turn: bool):
        # base case
        if ghost_position == pacman_position:
            return -1000 - depth
        
        if depth == 0:
            distances, parent = self.get_BFS_map(pacman_position)
            if ghost_position not in distances:
                return float('inf')
            
            # reconstruct path
            path = []
            current = ghost_position
            while current != pacman_position:
                previous, move_taken = parent[current]
                path.append(move_taken)
                current = previous
            path.reverse()
            
            # calculate score
            ticks = self.pacman_tick(path)
            score = (ticks * 10) + len(path) # use path length as tiebreaker
            if ghost_position in self.dead_ends:
                score -= 0.5
            return score
        
        # maximizing for ghost
        if ghost_turn:
            max_eval = -float('inf')
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
                next_ghost_pos = (ghost_position[0] + move.value[0], ghost_position[1] + move.value[1])
                
                if self.is_valid_position_precomputed(next_ghost_pos):
                    eval = self.minimax(next_ghost_pos, pacman_position, map_state, depth - 1, alpha, beta, False)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            return max_eval
        
        # minimizing for pacman
        else:
            min_eval = float('inf')
            pacman_next_positions = self.pacman_simulated_positions(pacman_position, map_state)
            for next_pacman_pos in pacman_next_positions:
                eval = self.minimax(ghost_position, next_pacman_pos, map_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
        
    def precompute_valid_tiles(self, map_state: np.ndarray):
        """Precompute valid tiles for quick access during minimax."""
        self.valid_tiles = {}
        self.dead_ends = set()
        
        # map all valid tiles
        height, width = map_state.shape
        for row in range(height):
            for col in range(width):
                if map_state[row, col] == 0:
                    self.valid_tiles[(row, col)] = True
                    
        # build adjacency graph with valid tiles
        graph = {pos: [] for pos in self.valid_tiles}
        for pos in self.valid_tiles:
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                x, y = move.value
                next_pos = (pos[0] + x, pos[1] + y)
                if next_pos in self.valid_tiles:
                    graph[pos].append(next_pos)
              
        # find dead-ends
        initial_dead_ends = [pos for pos, neighbours in graph.items() if len(neighbours) == 1]
        queue = deque(initial_dead_ends)
        
        while queue:
            current = queue.popleft()
            self.dead_ends.add(current)
            
            for neighbour in graph[current]:
                if neighbour not in self.dead_ends:
                    exits = sum(1 for n in graph[neighbour] if n not in self.dead_ends)
                    if exits <= 1:
                        queue.append(neighbour) 
    def is_valid_position_precomputed(self, pos: tuple):
        """Check if a position is valid using precomputed tiles."""
        return pos in self.valid_tiles
        
    def pacman_tick(self, path: list) -> int:
        if not path:
            return 0
        ticks = 0
        current_move = path[0]
        steps = 0
        
        for move in path:
            if move == current_move and steps < self.pacman_speed:
                steps += 1
            else:
                ticks += 1
                current_move = move
                steps = 1
        if steps > 0:
            ticks += 1
        return ticks
    
    def pacman_simulated_positions(self, pacman_position: tuple, map_state: np.ndarray):      
        """Generate sensible next moves for Pacman based on current position and map state."""
        simulated_positions = []
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            row, col = move.value
            current_position = pacman_position
            
            for step in range(self.pacman_speed):
                next_position = (current_position[0] + row, current_position[1] + col)
                
                if not self.is_valid_position_precomputed(next_position):
                    break
                current_position = next_position
                
            if current_position != pacman_position:
                simulated_positions.append(current_position)
        return simulated_positions
            
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        # Use current sighting, fallback to last known, or move randomly
        threat = enemy_position or self.last_known_enemy_pos
        
        if threat is None:
            # No information about enemy - move randomly
            return self._random_move(my_position, map_state)
        
        if len(self.valid_tiles) == 0:
            self.precompute_valid_tiles(map_state)
        
        best_move = Move.STAY
        max_eval = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
            next_pos = (my_position[0] + move.value[0], my_position[1] + move.value[1])
            
            if self.is_valid_position_precomputed(next_pos):
                eval = self.minimax(next_pos, threat, map_state, 3, alpha, beta, False)
                
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                    
                alpha = max(alpha, eval)
        
        return best_move
    
    def _random_move(self, my_position: tuple, map_state: np.ndarray) -> Move:
        """Random movement when enemy position is unknown."""
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        
        for move in all_moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            if self._is_valid_position(new_pos, map_state):
                return move
        
        return Move.STAY