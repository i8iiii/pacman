import sys
from pathlib import Path
import time
from collections import deque
import random
import numpy as np

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

ALL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

PRECOMPUTE_TIME_LIMIT = 0.4
FALLBACK_TRIGGER      = 0.6

# Cost for weighted BFS navigation through unknown territory
COST_KNOWN  = 1   # confirmed open cell
COST_UNKNOWN = 2  # unknown -1 cell

class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    
    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.catch_dist   = int(kwargs.get("capture_distance_threshold", 1))
        self.name = "Precomputed Pacman"
        self._minimax_threshold   = self.pacman_speed * 3
    
        # Memory for limited observation mode
        self.known_map     = None   # np.int8 (h, w): best knowledge so far
        self.last_known_enemy_pos = None
        self.steps_since_seen      = 0
        self._lost_track_limit    = max(6, 6 * self.pacman_speed)
        

        # Precomputed cache on known_
        self.last_map_hash = None
        self.precomputed       = False
        self._precompute_idx   = 0
        self.open_cells        = []
        self.open_set          = set()
        self._h                = 0
        self._w                = 0
        self.dist_matrix       = None   # np.int16 (n, n)
        self.first_move_matrix = None   # np.int8  (n, n)
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):

        t0 = time.time()

        self._update_known_map(map_state)
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.steps_since_seen     = 0
        else:
            self.steps_since_seen += 1

        # Recompute cache if know_map has changed
        current_hash = self.known_map.tobytes()
        if current_hash != self.last_map_hash:
            self.precomputed     = False
            self._precompute_idx = 0
            self.last_map_hash   = current_hash
        
        if not self.precomputed:
            self._precompute(time_limit=PRECOMPUTE_TIME_LIMIT)

        # Safety fallback if precompute took too long
        if time.time() - t0 > FALLBACK_TRIGGER:
            target = self.last_known_enemy_pos or my_position
            return self._fallback_greedy(my_position, target)

        # Choose strategy
        return self._decide(my_position, enemy_position)


    # ---------------- Known map accumulation ----------------
    def _update_known_map(self, map_state: np.ndarray):
        """Merge current FOV observation into persistent known_map."""
        if self.known_map is None:
            self.known_map = np.full(map_state.shape, -1, dtype=np.int8)
            self._h, self._w = map_state.shape
        revealed = map_state != -1
        self.known_map[revealed] = map_state[revealed]

    
    # ---------------- Search algorithm ----------------

    # Precompute: BFS all-pairs shortest path on confirmed open cells 
    def _precompute(self, time_limit: float = PRECOMPUTE_TIME_LIMIT):
        h, w = self._h, self._w
        n    = h * w

        # Full reinit when restarting
        if self._precompute_idx == 0:
            self.open_cells = [
                (r, c)
                for r in range(h) for c in range(w)
                if self.known_map[r, c] == 0
            ]
            self.open_set          = set(self.open_cells)
            self.dist_matrix       = np.full((n, n), -1, dtype=np.int16)
            self.first_move_matrix = np.full((n, n), -1, dtype=np.int8)

        t0 = time.time()
        while self._precompute_idx < len(self.open_cells):
            self._bfs_flat(self.open_cells[self._precompute_idx])
            self._precompute_idx += 1
            if time.time() - t0 > time_limit:
                return   # interrupted — retry next step

        self.precomputed = True

    def _bfs_flat(self, src: tuple):
        """BFS from src on known open cells only."""
        h, w = self._h, self._w
        si   = self._idx(src)

        dist_row  = self.dist_matrix[si]
        fmove_row = self.first_move_matrix[si]
        dist_row[si] = 0

        queue = [si]
        head  = 0

        while head < len(queue):
            idx  = queue[head]
            head += 1
            r, c = divmod(idx, w)
            d    = dist_row[idx] + 1

            for mi, move in enumerate(ALL_MOVES):
                dr, dc = move.value
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    continue
                if self.known_map[nr, nc] != 0:   # only confirmed open
                    continue
                nidx = nr * w + nc
                if dist_row[nidx] >= 0:
                    continue
                dist_row[nidx]  = d
                fmove_row[nidx] = fmove_row[idx] if fmove_row[idx] >= 0 else mi
                queue.append(nidx)


    # Choosing strategy
    def _decide(self, my_pos, enemy_pos):
        if enemy_pos is not None:
            # Ghost visible — use precomputed paths
            dist_to_enemy = self._get_dist(my_pos, enemy_pos)

            if dist_to_enemy <= self.catch_dist:
                # Already in catch range
                return self._fallback_greedy(my_pos, enemy_pos)

            if dist_to_enemy <= self._minimax_threshold:
                # Close range — minimax to cut off escape
                return self._minimax_move(my_pos, enemy_pos)

            # Far range — optimal precomputed path
            return self._lookup(my_pos, enemy_pos)

        else: 
            target = self.last_known_enemy_pos
            if target is None:
                # Never seen ghost
                return self._weighted_bfs_move(my_pos, target=None)
            
            dist_via_known = self._get_dist(my_pos, target)

            if dist_via_known != float('inf') and self.steps_since_seen <= self._lost_track_limit:
                # Confirmed path to last known position
                return self._lookup(my_pos, target)

            # No confirmed path or lost too long — venture through unknowns
            return self._weighted_bfs_move(my_pos, target=target)

    # Weighted BFS for Ghost not visible / path through unknown territory
    def _weighted_bfs_move(self, start, target):
        """Weighted BFS treating -1 cells as passable with higher cost.
        Navigates toward target if given, otherwise toward nearest -1 cell."""
        h, w = self._h, self._w

        MAX_EDGE   = COST_UNKNOWN
        NUM_BUCKET = MAX_EDGE + 1
        INF        = h * w * MAX_EDGE + 1

        buckets  = [[] for _ in range(NUM_BUCKET)]
        buckets[0].append((start, None))
        best_cost = {start: 0}
        current   = 0
        limit     = INF

        while current < limit:
            b = current % NUM_BUCKET
            if not buckets[b]:
                current += 1
                continue

            pos, first_action = buckets[b].pop()
            cost = best_cost.get(pos, INF)

            # Skip stale entries
            if cost < current:
                continue

            r, c = pos

            # Goal check
            if target is not None:
                if pos == target:
                    return first_action if first_action else self._fallback_greedy(start, target)
            else:
                # No target — find nearest -1 cell adjacent to known area
                if self.known_map[r, c] == -1:
                    return first_action if first_action else self._fallback_greedy(start, pos)

            for move in ALL_MOVES:
                cur = pos
                for steps in range(1, self.pacman_speed + 1):
                    dr, dc = move.value
                    nr, nc = cur[0] + dr, cur[1] + dc

                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        break
                    cell_val = self.known_map[nr, nc]
                    if cell_val == 1:   # confirmed wall
                        break

                    nxt      = (nr, nc)
                    # Cost based on cell type
                    step_cost = COST_KNOWN if cell_val == 0 else COST_UNKNOWN
                    new_cost  = t + step_cost

                    if new_cost < best_cost.get(nxt, INF):
                        best_cost[nxt] = new_cost
                        fa = first_action if first_action else (move, steps)
                        buckets[new_cost % NUM_BUCKET].append((nxt, fa))

                    cur = nxt

        return self._fallback_greedy(start, target or start)

    # Minimax 1-ply for close range, ghost visible
    def _minimax_move(self, my_pos, ghost_pos):
        """1-ply minimax: minimises worst-case distance after ghost's best escape"""
        best_action = None
        best_worst  = float('inf')

        for move in ALL_MOVES:
            max_s = self._max_valid_steps(my_pos, move, self.pacman_speed)
            if max_s == 0:
                continue

            # Among all steps in this direction, pick landing closest to ghost
            dr, dc     = move.value
            best_steps = 1
            best_land  = my_pos
            best_d     = float('inf')
            cur        = my_pos

            for s in range(1, max_s + 1):
                cur = (cur[0] + dr, cur[1] + dc)
                d   = self._get_dist(cur, ghost_pos)
                if d < best_d:
                    best_d     = d
                    best_steps = s
                    best_land  = cur

            # Ghost picks worst case
            worst_dist = -float('inf')
            for gmove in ALL_MOVES + [Move.STAY]:
                gdr, gdc = gmove.value
                g_land   = (ghost_pos[0] + gdr, ghost_pos[1] + gdc)
                if not self._is_valid_position(g_land):
                    g_land = ghost_pos
                d = self._get_dist(best_land, g_land)
                if d > worst_dist:
                    worst_dist = d

            # Picks move that minimises ghost's best escape distance
            if worst_dist < best_worst:
                best_worst  = worst_dist
                best_action = (move, best_steps)

        return best_action if best_action else self._fallback_greedy(my_pos, ghost_pos)
    
    def _lookup(self, src, dst):
        """Compute optimal (Move, steps) from precomputed matrices.
        Falls back to weighted BFS if path not in known map."""
        if self.first_move_matrix is None:
            return self._weighted_bfs_move(src, dst)

        si = self._idx(src)
        di = self._idx(dst)
        mi = int(self.first_move_matrix[si, di])
        if mi < 0:
            # No confirmed path — go through unknown territory
            return self._weighted_bfs_move(src, dst)

        first_move = ALL_MOVES[mi]
        max_s = self._max_valid_steps(src, first_move, self.pacman_speed)
        if max_s == 0:
            return self._weighted_bfs_move(src, dst)

        dr, dc     = first_move.value
        best_steps = 1
        best_dist  = float('inf')
        cur        = src

        for s in range(1, max_s + 1):
            cur = (cur[0] + dr, cur[1] + dc)
            d   = self._get_dist(cur, dst)
            if d < best_dist:
                best_dist  = d
                best_steps = s

        return (first_move, best_steps)

    # Fallback greedy (when cache missing or timeout)
    def _fallback_greedy(self, start, goal):
        """One-step greedy minimises Manhattan to goal."""
        best_move, best_steps, best_dist = None, 1, float('inf')
        for move in ALL_MOVES:
            cur = start
            for steps in range(1, self.pacman_speed + 1):
                dr, dc = move.value
                nxt    = (cur[0] + dr, cur[1] + dc)
                if not self._is_valid_position(nxt):
                    break
                d = self._manhattan(nxt, goal)
                if d < best_dist:
                    best_dist, best_move, best_steps = d, move, steps
                cur = nxt
        return (best_move, best_steps) if best_move else (Move.STAY, 1)

    # ---------------- Helper methods ----------------
    
    def _idx(self, pos):
        return pos[0] * self._w + pos[1]

    def _get_dist(self, src, dst):
        if self.dist_matrix is None:
            return self._manhattan(src, dst)
        n  = self._h * self._w
        si = self._idx(src)
        di = self._idx(dst)
        if si < 0 or si >= n or di < 0 or di >= n:
            return float('inf')
        d = int(self.dist_matrix[si, di])
        return d if d >= 0 else self._manhattan(src, dst)

    def _is_valid_position(self, pos):
        """Check against known_map — unknown treated as invalid for confirmed moves."""
        r, c = pos
        h, w = self._h, self._w
        if r < 0 or r >= h or c < 0 or c >= w:
            return False
        return self.known_map[r, c] == 0

    def _max_valid_steps(self, pos, move, max_steps):
        """Count consecutive confirmed-open steps in move direction."""
        steps, cur = 0, pos
        for _ in range(max_steps):
            dr, dc = move.value
            nxt    = (cur[0] + dr, cur[1] + dc)
            if not self._is_valid_position(nxt):
                break
            steps += 1
            cur    = nxt
        return steps
 
    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_known_enemy_pos = None

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        
        # 1. Cập nhật vị trí kẻ thù nếu nhìn thấy[cite: 5]
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            
        threat = self.last_known_enemy_pos
        
        valid_moves = []
        for move in ALL_MOVES:
            dr, dc = move.value
            nxt = (my_position[0] + dr, my_position[1] + dc)
            if self._is_valid_position(nxt, map_state):
                valid_moves.append((move, nxt))
                
        if not valid_moves:
            return Move.STAY
            
        # 2. Nếu chưa từng thấy Pacman, đi tìm các ngã rẽ an toàn
        if threat is None:
            valid_moves.sort(key=lambda x: self._get_degree(x[1], map_state), reverse=True)
            best_deg = self._get_degree(valid_moves[0][1], map_state)
            best_moves = [m for m, nxt in valid_moves if self._get_degree(nxt, map_state) == best_deg]
            return random.choice(best_moves)

        # 3. Tính BFS từ vị trí cuối cùng nhìn thấy Pacman (xuyên qua cả vùng -1)[cite: 5]
        pacman_distances = self._bfs_from(threat, map_state)

        best_move = Move.STAY
        best_score = -float('inf')

        for move, nxt_pos in valid_moves:
            dist = pacman_distances.get(nxt_pos, 0)
            degree = self._get_degree(nxt_pos, map_state)
            
            score = dist * 10 + degree
            
            # Phạt nặng ngõ cụt
            if degree <= 1:
                score -= 100 
                
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _bfs_from(self, start: tuple, map_state: np.ndarray) -> dict:
        queue = deque([(start, 0)])
        distances = {start: 0}
        
        while queue:
            curr, dist = queue.popleft()
            for move in ALL_MOVES:
                dr, dc = move.value
                nxt = (curr[0] + dr, curr[1] + dc)
                if self._is_valid_position(nxt, map_state) and nxt not in distances:
                    distances[nxt] = dist + 1
                    queue.append((nxt, dist + 1))
        return distances

    def _get_degree(self, pos: tuple, map_state: np.ndarray) -> int:
        count = 0
        for move in ALL_MOVES:
            dr, dc = move.value
            if self._is_valid_position((pos[0] + dr, pos[1] + dc), map_state):
                count += 1
        return count

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        # Chấp nhận đi vào cả ô 0 và ô -1 (vùng mù), chỉ chặn tường 1
        return map_state[row, col] != 1