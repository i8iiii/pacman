"""
Template for student agent implementation.
Refactored Version with Performance Tracker (Time & Memory).
"""

import sys
import time
import collections
import tracemalloc  # Thư viện tiêu chuẩn đo bộ nhớ RAM
from pathlib import Path
import numpy as np

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    
    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "bfs Pacman",
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # TODO: Initialize any data structures you need
        # Examples:
        # - self.path = []  # Store planned path
        # - self.visited = set()  # Track visited positions
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
    
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
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        # Use current sighting, fallback to last known, or explore
        target = enemy_position or self.last_known_enemy_pos
        
        if target is None:
            # No information about enemy
            return (Move.STAY, 1)
        
        path = self._bfs_path(my_position, target, map_state)
        
        if not path:
            # No path found, stay in place
            return (Move.STAY, 1)
        
        # --- collapse leading straight run into one fast jump ---
        first = path[0]
        steps = 1
        while steps < len(path) and path[steps] == first and steps < self.pacman_speed:
            steps += 1
        return (first, steps)        
        
    def _bfs_path(self, start: tuple, target: tuple, map_state: np.ndarray) -> Move:
        if start == target:
            return []
        
        queue = [(start, [])]
        visited = set()
        visited.add(start)
        
        while queue:
            cell, path = queue.pop(0)
            
            for neighbor in self._neighbors(cell, map_state):
                if neighbor in visited:
                    continue
                
                new_path = path + [neighbor]
                
                if neighbor == target:
                    # Convert path to moves
                    moves = []
                    current = start
                    for next_cell in new_path:
                        delta_row = next_cell[0] - current[0]
                        delta_col = next_cell[1] - current[1]
                        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                            if (delta_row, delta_col) == move.value:
                                moves.append(move)
                                break
                        current = next_cell
                    return moves
                visited.add(neighbor)
                queue.append((neighbor, new_path))
        
    def _neighbors(self, pos:tuple, map_state: np.ndarray):
        """return valid neighbors of pos"""
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(pos, move, map_state):
                delta_row, delta_col = move.value
                new_pos = (pos[0] + delta_row, pos[1] + delta_col)
                neighbors.append(new_pos)
        return neighbors
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        r, c = pos
        delta_row, delta_col = move.value
        new_r, new_c = r + delta_row, c + delta_col
        if not (0 <= new_r < map_state.shape[0] and 0 <= new_c < map_state.shape[1]):
            return False
        return map_state[new_r, new_c] == 0

class TimeoutException(Exception):
    pass

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = 2
        self.MAX_TIME = 0.85
        self.name = "My Logic Ghost"

    def step(self, map_state, my_position, enemy_position, step_number):
        start_time = time.time()

        seeker_pos = enemy_position
        if seeker_pos is None:
            return Move.STAY

        # 1. Tính toán khoảng cách thực tế (BFS)
        seeker_dist_map = self._bfs_seeker_distances(seeker_pos, map_state)
        current_dist = seeker_dist_map.get(my_position, 20)

        # 2. KIỂM TRA ĐỊA HÌNH: Xem có bao nhiêu hướng đi khả thi
        valid_moves = []
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nxt = self._get_next_pos(my_position, m, map_state)
            if nxt != my_position:
                valid_moves.append(m)

        if current_dist >= 5 and len(valid_moves) <= 2:
            best_move = valid_moves[0] if valid_moves else Move.STAY
            max_nxt_dist = -1

            for m in valid_moves:
                nxt = self._get_next_pos(my_position, m, map_state)
                d = seeker_dist_map.get(nxt, 0)
                if d > max_nxt_dist:
                    max_nxt_dist = d
                    best_move = m

            if max_nxt_dist >= current_dist:
                return best_move
        best_move = Move.STAY

        try:
            for depth in range(1, 20):
                score, move = self._minimax(
                    map_state, my_position, seeker_pos, depth, True,
                    float('-inf'), float('inf'), start_time, seeker_dist_map
                )
                if move is not None:
                    best_move = move
        except TimeoutException:
            pass

        return best_move

    def _minimax(self, map_state, g_pos, p_pos, depth, is_max, alpha, beta, start_time, seeker_dist_map):
        if time.time() - start_time > self.MAX_TIME:
            raise TimeoutException()

        if g_pos == p_pos or abs(g_pos[0] - p_pos[0]) + abs(g_pos[1] - p_pos[1]) < 2:
            return -20000 + (10 - depth), None

        if depth == 0:
            return self._evaluate(g_pos, p_pos, map_state, seeker_dist_map), None

        if is_max:
            val = float('-inf')
            best_m = Move.STAY

            moves = []
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
                nxt = g_pos if m == Move.STAY else self._get_next_pos(g_pos, m, map_state)
                if m != Move.STAY and nxt == g_pos: continue

                dist = self._manhattan(nxt, p_pos)
                moves.append((dist, m, nxt))

            moves.sort(key=lambda x: x[0], reverse=True)

            for _, m, nxt in moves:
                res, _ = self._minimax(map_state, nxt, p_pos, depth - 1, False, alpha, beta, start_time,
                                       seeker_dist_map)

                if m == Move.STAY:
                    res -= 50  # Phạt lười biếng

                if res > val:
                    val = res
                    best_m = m

                alpha = max(alpha, val)
                if beta <= alpha: break
            return val, best_m

        else:
            val = float('inf')
            p_options = self._get_seeker_options(p_pos, g_pos, map_state)

            for nxt_p in p_options:
                res, _ = self._minimax(map_state, g_pos, nxt_p, depth - 1, True, alpha, beta, start_time,
                                       seeker_dist_map)
                if res < val:
                    val = res

                beta = min(beta, val)
                if beta <= alpha: break
            return val, None

    def _evaluate(self, g_pos, p_pos, map_state, seeker_dist_map):
        true_dist = seeker_dist_map.get(g_pos, 20)
        sim_dist = self._manhattan(g_pos, p_pos)

        dist_score = (true_dist * 500) + (sim_dist * 200)

        safe_area = self._flood_fill_safe_area(g_pos, seeker_dist_map, map_state)

        if true_dist < 6:
            dist_score = true_dist * 2000

        return dist_score + (safe_area * 50)

    def _bfs_seeker_distances(self, p_pos, map_state):
        """BFS chuẩn single-pass từ vị trí Pacman.
        Trả về {pos: số bước tối thiểu để Pacman đến pos}.
        Không nhân đôi vòng (không s2), tránh ghi sai khoảng cách.
        """
        dist = {p_pos: 0}
        q = collections.deque([p_pos])
        while q:
            curr = q.popleft()
            d = dist[curr]
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = (curr[0] + m.value[0], curr[1] + m.value[1])
                if self._is_valid(nxt, map_state) and nxt not in dist:
                    dist[nxt] = d + 1
                    q.append(nxt)
        return dist

    def _flood_fill_safe_area(self, g_pos, seeker_dist_map, map_state):
        """Flood Fill nhẹ để tránh ngõ cụt"""
        safe_area = 0
        q = collections.deque([(g_pos, 0)])
        visited = {g_pos}
        while q:
            curr, dist = q.popleft()
            if dist > 4: break

            if dist * self.pacman_speed < seeker_dist_map.get(curr, 99):
                safe_area += 1
                for n in self._get_neighbors(curr, map_state):
                    if n not in visited:
                        visited.add(n)
                        q.append((n, dist + 1))
        return safe_area

    def _get_seeker_options(self, p_pos, g_pos, map_state):
        """BFS mở rộng đúng pacman_speed bước từ p_pos trong 1 lượt.
        Bắt đủ mọi hướng di chuyển (thẳng + rẽ), không bỏ sót L-shape.
        Sắp xếp theo manhattan đến Ghost (gần nhất trước = nguy hiểm nhất cho Ghost).
        """
        visited = {p_pos}
        frontier = [p_pos]
        for _ in range(self.pacman_speed):
            nxt_frontier = []
            for pos in frontier:
                for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    nxt = (pos[0] + m.value[0], pos[1] + m.value[1])
                    if self._is_valid(nxt, map_state) and nxt not in visited:
                        visited.add(nxt)
                        nxt_frontier.append(nxt)
            frontier = nxt_frontier
        options = sorted(visited, key=lambda x: self._manhattan(x, g_pos))
        return options[:6]

    def _manhattan(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _get_neighbors(self, pos, map_state):
        return [(pos[0] + m.value[0], pos[1] + m.value[1]) for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
                if self._is_valid((pos[0] + m.value[0], pos[1] + m.value[1]), map_state)]

    def _is_valid(self, pos, map_state):
        r, c = pos
        return 0 <= r < map_state.shape[0] and 0 <= c < map_state.shape[1] and map_state[r, c] == 0

    def _get_next_pos(self, pos, move, map_state):
        nxt = (pos[0] + move.value[0], pos[1] + move.value[1])
        return nxt if self._is_valid(nxt, map_state) else pos