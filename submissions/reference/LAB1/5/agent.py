"""
Template for student agent implementation.

INSTRUCTIONS:
1. Copy this file to submissions/<your_student_id>/agent.py
2. Implement the PacmanAgent and/or GhostAgent classes
3. Replace the simple logic with your search algorithm
4. Test your agent using: python arena.py --seek <your_id> --hide example_student

IMPORTANT:
- Do NOT change the class names (PacmanAgent, GhostAgent)
- Do NOT change the method signatures (step, __init__)
- Pacman step must return either a Move or a (Move, steps) tuple where
    1 <= steps <= pacman_speed (provided via kwargs)
- Ghost step must return a Move enum value
- You CAN add your own helper methods
- You CAN import additional Python standard libraries
"""

import sys
from collections import deque
from pathlib import Path

# Add the ``src`` directory to the front of the module search path so this file
# can import shared interface classes when run from a submission directory.
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


class PacmanAgent(BasePacmanAgent):
    """
    Pacman uses Minimax with Alpha-Beta pruning to intercept and catch Ghost.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.capture_distance_threshold = int(kwargs.get("capture_distance_threshold", 2))
        self.name = "Minimax Pacman"

        # Initialize the BFS distance cache.
        self.bfs_cache = {}
        # Previous position kept for interface compatibility.
        self.previous_ghost_position = None

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int):
        """
        Choose the best move with a 4-ply Minimax search (2 full turns).
        """
        if enemy_position is None:
            return (Move.STAY, 1)

        # Clear the cache every turn to use the latest map state.
        self.bfs_cache = {}
        self.previous_ghost_position = enemy_position

        best_score = -float('inf')
        best_action = (Move.STAY, 1)

        # Iterate over Pacman's possible directions.
        moves = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
        for move in moves:
            dr, dc = move.value
            # Try longer steps first.
            for steps in range(self.pacman_speed, 0, -1):
                next_p = (my_position[0] + dr * steps, my_position[1] + dc * steps)
                if not self._is_straight_action_valid(my_position, move, steps, map_state):
                    continue

                # Run Minimax for the next 3 plies (total depth is 4).
                score = self._minimax(
                    next_p, enemy_position, 3, False, -float('inf'), float('inf'), map_state
                )
                if score > best_score:
                    best_score = score
                    best_action = (move, steps)

        return best_action

    def _minimax(self, p_pos: tuple, g_pos: tuple, depth: int, is_max: bool,
                 alpha: float, beta: float, map_state: np.ndarray) -> float:
        # Check the early win/loss condition: Ghost has been caught.
        manhattan_dist = abs(p_pos[0] - g_pos[0]) + abs(p_pos[1] - g_pos[1])
        if manhattan_dist < self.capture_distance_threshold:
            return 10000 - depth  # Very large score, preferring earlier wins.

        if depth == 0:
            # Evaluation function: penalize by the real BFS distance.
            return -self._get_bfs_distance(p_pos, g_pos, map_state)

        if is_max:
            # Pacman's turn (max node).
            max_val = -float('inf')
            moves = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
            for move in moves:
                dr, dc = move.value
                for steps in range(self.pacman_speed, 0, -1):
                    next_p = (p_pos[0] + dr * steps, p_pos[1] + dc * steps)
                    if not self._is_straight_action_valid(p_pos, move, steps, map_state):
                        continue

                    val = self._minimax(next_p, g_pos, depth - 1, False, alpha, beta, map_state)
                    max_val = max(max_val, val)
                    alpha = max(alpha, val)
                    if beta <= alpha:
                        break
                if beta <= alpha:
                    break
            
            if max_val == -float('inf'):
                # Force Pacman to stay still if no legal move exists.
                max_val = self._minimax(p_pos, g_pos, depth - 1, False, alpha, beta, map_state)
            return max_val
        else:
            # Ghost's turn (min node).
            min_val = float('inf')
            moves = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY)
            has_valid_move = False
            for move in moves:
                dr, dc = move.value
                next_g = (g_pos[0] + dr, g_pos[1] + dc)
                if self._is_valid_position(next_g, map_state):
                    has_valid_move = True
                    val = self._minimax(p_pos, next_g, depth - 1, True, alpha, beta, map_state)
                    min_val = min(min_val, val)
                    beta = min(beta, val)
                    if beta <= alpha:
                        break
            
            if not has_valid_move:
                min_val = self._minimax(p_pos, g_pos, depth - 1, True, alpha, beta, map_state)
            return min_val

    def _get_bfs_distance(self, p_pos: tuple, g_pos: tuple, map_state: np.ndarray) -> int:
        """Get the real shortest-path distance, using a cache."""
        if g_pos not in self.bfs_cache:
            self.bfs_cache[g_pos] = self._bfs_distance_map(g_pos, map_state)
        return self.bfs_cache[g_pos].get(p_pos, 999)

    def _bfs_distance_map(self, start: tuple, map_state: np.ndarray) -> dict:
        """Build a BFS distance map from the start position to all positions."""
        if start is None or not self._is_valid_position(start, map_state):
            return {}

        queue = deque([start])
        distances = {start: 0}

        while queue:
            current = queue.popleft()
            for move in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
                neighbor = (current[0] + move.value[0], current[1] + move.value[1])
                if (neighbor not in distances
                        and self._is_valid_position(neighbor, map_state)):
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

        return distances

    def _predict_ghost_position(self, current_position: tuple,
                                map_state: np.ndarray):
        """
        Predict Ghost continuing one more cell in its latest movement direction.

        For example, if Ghost moves from ``(5, 3)`` to ``(5, 4)``, the predicted
        position is ``(5, 5)``. The prediction is discarded if the data does not
        represent a legal one-step move, or if the cell ahead is a wall/outside
        the map.
        """
        if self.previous_ghost_position is None:
            return None

        delta_row = current_position[0] - self.previous_ghost_position[0]
        delta_col = current_position[1] - self.previous_ghost_position[1]

        # Ghost may only stay still or move one cell in the four cardinal directions.
        if abs(delta_row) + abs(delta_col) > 1:
            return None

        predicted_position = (
            current_position[0] + delta_row,
            current_position[1] + delta_col,
        )
        if not self._is_valid_position(predicted_position, map_state):
            return None

        # At junctions, Ghost has many ways to turn, so a straight prediction is
        # unreliable. Only predict in corridors with exactly two exits.
        valid_neighbors = 0
        for move in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
            neighbor = (
                current_position[0] + move.value[0],
                current_position[1] + move.value[1],
            )
            if self._is_valid_position(neighbor, map_state):
                valid_neighbors += 1

        if valid_neighbors == 2:
            return predicted_position
        return None

    def _bfs_by_turns(self, start: tuple, target: tuple,
                      map_state: np.ndarray):
        """
        Run BFS for the path with the fewest turns, not just the fewest cells.

        From each position, an edge can move straight for 1, 2, ...,
        ``pacman_speed`` cells. Every edge costs one turn, so BFS still
        guarantees the optimal solution by turn count. The result is a list of
        ``(Move, steps)`` actions.
        """
        if not self._is_valid_position(target, map_state):
            return []
        if start == target:
            return []

        queue = deque([start])
        # parent[position] = (previous_position, action_used)
        parent = {start: None}
        moves = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)

        while queue:
            current = queue.popleft()
            for move in moves:
                dr, dc = move.value
                # Try longer steps first. When multiple paths use the same number
                # of turns, this better uses Pacman's speed without losing optimality.
                for steps in range(self.pacman_speed, 0, -1):
                    neighbor = (
                        current[0] + dr * steps,
                        current[1] + dc * steps,
                    )
                    if not self._is_straight_action_valid(
                            current, move, steps, map_state):
                        continue
                    if neighbor in parent:
                        continue

                    action = (move, steps)
                    parent[neighbor] = (current, action)
                    if neighbor == target:
                        return self._reconstruct_actions(
                            start, target, parent
                        )
                    queue.append(neighbor)

        return []

    def _reconstruct_actions(self, start: tuple, target: tuple, parent: dict):
        """Backtrack through ``parent`` to build the action list from start to target."""
        actions = []
        current = target
        while current != start:
            previous, action = parent[current]
            actions.append(action)
            current = previous
        actions.reverse()
        return actions

    def _is_straight_action_valid(self, start: tuple, move: Move, steps: int,
                                  map_state: np.ndarray) -> bool:
        """Check every cell crossed by one straight action."""
        dr, dc = move.value
        for distance in range(1, steps + 1):
            position = (
                start[0] + dr * distance,
                start[1] + dc * distance,
            )
            if not self._is_valid_position(position, map_state):
                return False
        return True

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check that a position is inside the map and is not a wall."""
        row, col = pos
        height, width = map_state.shape

        return (
            0 <= row < height
            and 0 <= col < width
            and map_state[row, col] == 0
        )


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    
    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo
    """

    DISTANCE_WEIGHT = 10
    SAFE_AREA_WEIGHT = 2
    EXIT_WEIGHT = 3
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Store the most recently seen Pacman position for limited-visibility mode.
        self.last_known_enemy_pos = None
        # The cell Ghost occupied last turn; avoid moving back to it.
        self.previous_position = None

        # Read Pacman's speed from command-line arguments (sys.argv).
        self.pacman_speed = 2  # Default is 2.
        if '--pacman-speed' in sys.argv:
            try:
                idx = sys.argv.index('--pacman-speed')
                self.pacman_speed = max(1, int(sys.argv[idx + 1]))
            except (ValueError, IndexError):
                pass
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Decide Ghost's one-cell movement direction for the current turn.
        
        Args:
            map_state: A 2D NumPy array; 1=wall, 0=empty cell, -1=unseen.
            my_position: Ghost's current ``(row, col)`` position.
            enemy_position: Pacman's position if visible, otherwise None.
            step_number: The current turn number, starting at 1.
            
        Returns:
            A direction from Move.UP, DOWN, LEFT, RIGHT, or STAY.
        """
        # Update memory whenever Pacman appears in the visible area.
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        threat = enemy_position or self.last_known_enemy_pos

        distance_map = self._bfs_distance_map(threat, map_state)

        # When Pacman and Ghost have a clear line of sight on the same row/column,
        # first try to leave that line through the nearest perpendicular turn.
        aligned_axis = self._aligned_axis(my_position, threat, map_state)
        if aligned_axis is not None:
            turn_move = self._fastest_turn_move(
                my_position, threat, aligned_axis, map_state
            )
            if turn_move is not None:
                self.previous_position = my_position
                return turn_move

        # Score each move by distance, safe area, and exit count.
        evaluated_moves = []
        for move in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
            next_position = self._apply_move(my_position, move)
            if next_position == self.previous_position:
                continue
            if self._is_valid_position(next_position, map_state):
                exit_count = self._count_exits(next_position, map_state)
                distance = distance_map.get(next_position, 0)
                safe_area = self._safe_flood_fill_area(
                    next_position, distance_map, map_state
                )
                score = (
                    distance * self.DISTANCE_WEIGHT
                    + safe_area * self.SAFE_AREA_WEIGHT
                    + exit_count * self.EXIT_WEIGHT
                )
                evaluated_moves.append(
                    (move, exit_count, distance, safe_area, score)
                )

        if not evaluated_moves:
            self.previous_position = my_position
            return Move.STAY

        # Filter out dead ends if any cell has at least two exits. Only enter a
        # dead end when every available choice has one exit.
        safe_moves = [item for item in evaluated_moves if item[1] > 1]
        candidates = safe_moves or evaluated_moves

        # The highest score balances distance from Pacman, open escape area, and
        # exit count. Class-level weights make the policy easy to explain/tune.
        highest_score = max(item[4] for item in candidates)
        best_moves = [
            move for move, _, _, _, score in candidates
            if score == highest_score
        ]

        # On ties, keep the old strategy: prefer moving away along the dominant axis.
        preferred_move = self._move_away_from_threat(my_position, threat)
        if preferred_move in best_moves:
            best_move = preferred_move
        else:
            best_move = best_moves[0]

        self.previous_position = my_position
        return best_move

    def _aligned_axis(self, position: tuple, threat: tuple,
                      map_state: np.ndarray):
        """Return the clear line-of-sight axis: ``row`` or ``col``."""
        if threat is None:
            return None

        if position[0] == threat[0]:
            row = position[0]
            left, right = sorted((position[1], threat[1]))
            if all(map_state[row, col] == 0
                   for col in range(left + 1, right)):
                return "row"

        if position[1] == threat[1]:
            col = position[1]
            top, bottom = sorted((position[0], threat[0]))
            if all(map_state[row, col] == 0
                   for row in range(top + 1, bottom)):
                return "col"

        return None

    def _fastest_turn_move(self, start: tuple, threat: tuple, axis: str,
                           map_state: np.ndarray):
        """Use BFS to find the first action toward the nearest perpendicular turn."""
        perpendicular_moves = (
            (Move.UP, Move.DOWN)
            if axis == "row"
            else (Move.LEFT, Move.RIGHT)
        )
        all_moves = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
        queue = deque([(start, None)])
        visited = {start}

        while queue:
            current, first_move = queue.popleft()
            for move in all_moves:
                neighbor = self._apply_move(current, move)
                if (neighbor in visited
                        or neighbor == self.previous_position
                        or neighbor == threat
                        or not self._is_valid_position(neighbor, map_state)):
                    continue

                path_first_move = first_move or move
                if move in perpendicular_moves:
                    return path_first_move

                visited.add(neighbor)
                queue.append((neighbor, path_first_move))

        return None

    def _bfs_distance_map(self, start: tuple,
                          map_state: np.ndarray) -> dict:
        """Compute shortest-path distances from Pacman to every empty cell."""
        if start is None or not self._is_valid_position(start, map_state):
            return {}

        queue = deque([start])
        distances = {start: 0}

        while queue:
            current = queue.popleft()
            for move in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
                neighbor = self._apply_move(current, move)
                if (neighbor not in distances
                        and self._is_valid_position(neighbor, map_state)):
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

        return distances

    def _safe_flood_fill_area(self, start: tuple, distance_map: dict,
                              map_state: np.ndarray) -> int:
        """
        Count how many cells Ghost can reach before Pacman.

        Ghost starts at the input position at time 0. A cell is considered safe
        only when the number of Ghost steps needed to reach it is smaller than
        Pacman's BFS distance.
        """
        if not self._is_valid_position(start, map_state):
            return 0
        if distance_map.get(start, float("inf")) <= 0:
            return 0

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            current, ghost_distance = queue.popleft()
            for move in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
                neighbor = self._apply_move(current, move)
                next_ghost_distance = ghost_distance + 1
                pacman_distance = distance_map.get(neighbor, float("inf"))

                if (neighbor not in visited
                        and self._is_valid_position(neighbor, map_state)
                        and next_ghost_distance * self.pacman_speed < pacman_distance):
                    visited.add(neighbor)
                    queue.append((neighbor, next_ghost_distance))

        return len(visited)

    def _move_away_from_threat(self, position: tuple, threat: tuple):
        """Choose the direction away from Pacman along the larger-difference axis."""
        if threat is None:
            return None

        row_diff = position[0] - threat[0]
        col_diff = position[1] - threat[1]
        if abs(row_diff) > abs(col_diff):
            return Move.DOWN if row_diff > 0 else Move.UP
        return Move.RIGHT if col_diff > 0 else Move.LEFT

    def _count_exits(self, position: tuple, map_state: np.ndarray) -> int:
        """Count exits: 1=dead end, 2=corridor, 3-4=junction."""
        count = 0
        for move in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
            next_position = self._apply_move(position, move)
            if self._is_valid_position(next_position, map_state):
                count += 1
        return count

    def _apply_move(self, position: tuple, move: Move) -> tuple:
        """Compute the coordinates after one step in the chosen direction."""
        return (
            position[0] + move.value[0],
            position[1] + move.value[1],
        )

    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check whether the cell reached after one step is valid."""
        return self._is_valid_position(
            self._apply_move(pos, move), map_state
        )
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check that a position is inside the map and is an observed empty cell."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        # Ghost also cannot pass through walls or enter unknown fog-of-war cells.
        return map_state[row, col] == 0
