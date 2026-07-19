import sys
from pathlib import Path
from collections import deque

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import random


class PacmanAgent(BasePacmanAgent):
    """
    Seeker agent using BFS for optimal shortest-path navigation.

    Strategy:
      • When the Ghost is visible → BFS to its current position, follow
        the first move of the path, use Pacman's speed multiplier to move
        as many straight-line steps as allowed.
      • When the Ghost is not visible → BFS toward the last known position;
        if never seen, fall back to random exploration.

    Path caching:
      The BFS result is cached and only recomputed when the Ghost moves
      ≥ 2 cells from the position we planned for, keeping each step well
      within the 1-second time limit even on the full 21×21 grid.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "BFS Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))

        # Path cache
        self._cached_path: list = []       # remaining moves in current plan
        self._cached_target: tuple = None  # ghost pos used to build cache

        # Memory for fog-of-war
        self.last_known_enemy_pos = None

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int):

        # Update fog-of-war memory
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        target = enemy_position or self.last_known_enemy_pos

        if target is None:
            # No information at all — explore randomly
            return self._explore(my_position, map_state)

        # Invalidate cache when the target has moved significantly
        if (self._cached_target is None
                or self._manhattan(target, self._cached_target) >= 2
                or not self._cached_path):
            self._cached_path = self._bfs_path(my_position, target, map_state)
            self._cached_target = target

        # Check if the next move is valid; if not, pop it until a valid one
        while self._cached_path:
            move = self._cached_path[0]
            new_pos = (my_position[0] + move.value[0], my_position[1] + move.value[1])
            if self._is_valid_position(new_pos, map_state):
                break
            self._cached_path.pop(0)

        if not self._cached_path:
            # Cache empty after stale-prefix removal — replan from scratch
            self._cached_path = self._bfs_path(my_position, target, map_state)
            self._cached_target = target

        if not self._cached_path:
            # Truly no path (shouldn't happen on a connected map)
            return self._explore(my_position, map_state)

        first_move = self._cached_path[0]

        # Use speed multiplier: walk as many straight-line steps as allowed
        steps = self._max_valid_steps(my_position, first_move, map_state,
                                      self._straight_run(first_move, self._cached_path))

        # Consume the steps we are actually taking from the cache
        self._cached_path = self._cached_path[steps:]

        return (first_move, steps)

    def _bfs_path(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        """
        Return the shortest list of Moves from start to goal.
        Returns [] if start == goal or no path exists.
        """
        if start == goal:
            return []

        queue = deque()
        queue.append((start, []))
        visited = {start}

        while queue:
            current, path = queue.popleft()

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                delta_row, delta_col = move.value
                nxt = (current[0] + delta_row, current[1] + delta_col)

                if nxt == goal:
                    return path + [move]

                if nxt not in visited and self._is_valid_position(nxt, map_state):
                    visited.add(nxt)
                    queue.append((nxt, path + [move]))

        return []


    def _straight_run(self, move: Move, path: list) -> int:
        """Count how many leading moves in path share the same direction as move."""
        count = 0
        for m in path:
            if m != move:
                break
            count += 1
        return max(1, count)

    @staticmethod
    def _manhattan(a: tuple, b: tuple) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _explore(self, my_position: tuple, map_state: np.ndarray):
        """Random exploration when enemy position is unknown."""
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)

        for move in all_moves:
            steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
            if steps > 0:
                return (move, steps)

        return (Move.STAY, 1)

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape

        if row < 0 or row >= height or col < 0 or col >= width:
            return False

        return map_state[row, col] == 0

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, desired_steps: int) -> int:
        steps = 0
        max_steps = min(self.pacman_speed, max(1, desired_steps))
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps
 
 
class GhostAgent(BaseGhostAgent):
    """
    Hider agent using Minimax with alpha-beta pruning.
 
    Minimax models the game as a two-player zero-sum tree:
      - Ghost (maximiser) — wants to stay alive as long as possible.
      - Pacman (minimiser) — plays optimally to catch the Ghost fastest.
 
    At each node the evaluation score is the BFS distance from Pacman to
    the Ghost. A high score means the Ghost is far from Pacman — good for
    the Ghost. Pacman tries to minimise this; Ghost tries to maximise it.
 
    Alpha-beta pruning cuts branches that cannot affect the final decision,
    allowing a deeper search within the 1-second time budget.
 
    Depth is set to 4 (2 Ghost moves + 2 Pacman moves lookahead), which
    is safe on the 21x21 grid given the small branching factor (<=4 moves).
    """
 
    DEPTH = 4   # search depth (plies)
 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Minimax Ghost"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.last_known_enemy_pos = None
 
 
    def step(self, map_state, my_position, enemy_position, step_number):
 
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
 
        threat = enemy_position or self.last_known_enemy_pos
 
        if threat is None:
            return self._random_move(my_position, map_state)
 
        best_move = None
        best_score = -1
 
        # Ghost is the maximiser — try every valid move and pick the best
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            delta_row, delta_col = move.value
            next_ghost = (my_position[0] + delta_row, my_position[1] + delta_col)
 
            if not self._is_valid_position(next_ghost, map_state):
                continue
 
            # After Ghost moves, it is Pacman's turn (minimiser)
            score = self._minimax(
                ghost_pos=next_ghost,
                pacman_pos=threat,
                map_state=map_state,
                depth=self.DEPTH - 1,
                is_ghost_turn=False,
                alpha=-1,
                beta=float('inf')
            )
 
            if score > best_score:
                best_score = score
                best_move = move
 
        return best_move if best_move is not None else self._random_move(my_position, map_state)

 
    def _minimax(self, ghost_pos, pacman_pos, map_state, depth, is_ghost_turn, alpha, beta):
 
        # Terminal: Pacman catches Ghost (Manhattan distance < 2)
        dist = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
        if dist < 2:
            return 0   # Ghost is caught — worst outcome
 
        # Leaf node: evaluate using BFS distance (true shortest path)
        if depth == 0:
            return self._bfs_distance(pacman_pos, ghost_pos, map_state)
 
        if is_ghost_turn:
            # Ghost maximises score
            max_score = -1
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                next_ghost = (ghost_pos[0] + dr, ghost_pos[1] + dc)
 
                if not self._is_valid_position(next_ghost, map_state):
                    continue
 
                score = self._minimax(next_ghost, pacman_pos, map_state,
                                      depth - 1, False, alpha, beta)
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break   # beta cut-off
 
            return max_score if max_score != -1 else 0
 
        else:
            # Pacman minimises score — simulate Pacman moving optimally
            min_score = float('inf')
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
 
                # Pacman can move up to PACMAN_SPEED steps in a straight line
                next_pacman = pacman_pos
                moved = False
                for _ in range(self.pacman_speed):
                    candidate = (next_pacman[0] + dr, next_pacman[1] + dc)
                    if not self._is_valid_position(candidate, map_state):
                        break
                    next_pacman = candidate

                    # Evaluate Pacman at EACH step, not just the final one
                    score = self._minimax(ghost_pos, next_pacman, map_state,
                              depth - 1, True, alpha, beta)
                    min_score = min(min_score, score)
                    beta = min(beta, score)
                    if beta <= alpha:
                        break
            
            return min_score if min_score != float('inf') else 0
 
    def _bfs_distance(self, start, goal, map_state):
        """
        BFS shortest-path distance from start to goal.
        Returns 0 if start == goal, -1 if unreachable.
        Used as the Minimax leaf evaluation function.
        """
        if start == goal:
            return 0
 
        queue = deque([(start, 0)])
        visited = {start}
 
        while queue:
            current, dist = queue.popleft()
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                delta_row, delta_col = move.value
                nxt = (current[0] + delta_row, current[1] + delta_col)
 
                if nxt == goal:
                    return dist + 1
 
                if nxt not in visited and self._is_valid_position(nxt, map_state):
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))
 
        return -1   # unreachable
 
    def _random_move(self, my_position, map_state):
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
 
        for move in all_moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            if self._is_valid_position(new_pos, map_state):
                return move
 
        return Move.STAY
 
    def _is_valid_position(self, pos, map_state):
        row, col = pos
        height, width = map_state.shape
 
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
 
        return map_state[row, col] == 0