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
    
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
        self.steps_since_seen      = 0      # number of steps since last saw the ghost
        self.last_move             = None   # last Move enum used
        self._lost_track_limit    = max(6, 6 * self.pacman_speed)
        self._minimax_threshold   = self.pacman_speed * 3

        # Precomputed cache
        self.last_map_hash = None
        self.dist          = {}   # dist[src][dst] = shortest path in cells
        self.next_step     = {}   # next_step[(src, dst)] = (Move, steps)
        self.open_cells    = []
    
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
        t0 = time.time()

        # Recompute cache if map has changed
        current_hash = map_state.tobytes()
        if current_hash != self.last_map_hash:
            self._precompute(map_state)
            self.last_map_hash = current_hash

        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.steps_since_seen     = 0
        else:
            self.steps_since_seen += 1

        # Safety fallback if precompute took too long
        if time.time() - t0 > 0.8:
            target = self.last_known_enemy_pos or my_position
            action = self._fallback_greedy(map_state, my_position, target)
            self.last_move = action[0]
            return action

        # Choose strategy
        action = self._decide(map_state, my_position, enemy_position)
        self.last_move = action[0]
        return action
    
    # ---------------- Search algorithm ----------------

    # BFS from src to all reachable open cells
    def _bfs_from(self, src: tuple, map_state: np.ndarray):
        dist_map       = {src: 0}
        first_move_map = {}
        queue          = [src]
        head           = 0

        while head < len(queue):
            pos  = queue[head]
            head += 1
            for move in ALL_MOVES:
                dr, dc = move.value
                nxt    = (pos[0] + dr, pos[1] + dc)
                if nxt in dist_map:
                    continue
                if not self._is_valid_position(nxt, map_state):
                    continue
                dist_map[nxt]       = dist_map[pos] + 1
                first_move_map[nxt] = first_move_map.get(pos, move)
                queue.append(nxt)

        return dist_map, first_move_map

    # Precompute: BFS all-pairs shortest path + next_step table
    def _precompute(self, map_state: np.ndarray):
        h, w = map_state.shape
        self.open_cells = [
            (r, c)
            for r in range(h) for c in range(w)
            if map_state[r, c] == 0
        ]
        self.dist      = {}
        self.next_step = {}

        for src in self.open_cells:
            dist_from_src, first_move_from_src = self._bfs_from(src, map_state)
            self.dist[src] = dist_from_src
            for dst, fmove in first_move_from_src.items():
                self.next_step[(src, dst)] = self._best_action(
                    src, dst, fmove, dist_from_src, map_state
                )

    def _best_action(self, src: tuple, dst: tuple, first_move: Move,
                     dist_from_src: dict, map_state: np.ndarray):
        """Compute optimal (Move, steps) using pacman_speed.
        Walk straight in first_move direction up to pacman_speed steps,
        pick the landing position closest to dst by real distance."""
        max_s = self._max_valid_steps(src, first_move, map_state, self.pacman_speed)
        if max_s == 0:
            return (first_move, 1)

        dr, dc     = first_move.value
        best_steps = 1
        best_dist  = float('inf')
        cur        = src

        for s in range(1, max_s + 1):
            cur = (cur[0] + dr, cur[1] + dc)
            # Use cached real distance
            # fallback to Manhattan if not yet computed
            d = self.dist.get(cur, {}).get(dst, self._manhattan(cur, dst))
            if d < best_dist:
                best_dist  = d
                best_steps = s
        return (first_move, best_steps)

    # Choosing strategy
    def _decide(self, map_state, my_pos, enemy_pos):
        target = enemy_pos or self.last_known_enemy_pos
 
        if target is None:
            # No information about ghost -> BFS-explore unseen cells
            return self._bfs_explore(map_state, my_pos)
 
        # Snap target to nearest open cell if it is on a wall or fog cell
        target = self._nearest_open(target, map_state)
        dist_to_target = self.dist.get(my_pos, {}).get(target, self._manhattan(my_pos, target))

        if dist_to_target <= self.catch_dist:
            # Already within capture range
            return self._step_toward(map_state, my_pos, target)
 
        if enemy_pos is not None:
            # Ghost is visible
            if dist_to_target <= self._minimax_threshold:
                # Close range: minimax
                return self._minimax_move(my_pos, enemy_pos, map_state)
            else:
                # Long range: precomputed optimal path
                return self._lookup(map_state, my_pos, target)
        else:
            # Ghost not visible
            if self.steps_since_seen <= self._lost_track_limit:
                # Recently lost sight -> Head to last known position
                return self._lookup(map_state, my_pos, target)
            else:
                # Lost track too long —> switch to exploration
                return self._bfs_explore(map_state, my_pos)

    # Minimax 1-ply
    def _minimax_move(self, my_pos, ghost_pos, map_state):
        """1-ply minimax: Pacman minimises worst-case distance after ghost's best escape"""
        best_action = None
        best_worst  = float('inf')

        for move in ALL_MOVES:
            max_s = self._max_valid_steps(my_pos, move, map_state, self.pacman_speed)
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
                d   = self.dist.get(cur, {}).get(ghost_pos, self._manhattan(cur, ghost_pos))
                if d < best_d:
                    best_d     = d
                    best_steps = s
                    best_land  = cur

            # Ghost picks worst case
            worst_dist = -float('inf')
            for gmove in ALL_MOVES + [Move.STAY]:
                gdr, gdc = gmove.value
                g_land   = (ghost_pos[0] + gdr, ghost_pos[1] + gdc)
                if not self._is_valid_position(g_land, map_state):
                    g_land = ghost_pos
                d = self.dist.get(best_land, {}).get(g_land, self._manhattan(best_land, g_land))
                if d > worst_dist:
                    worst_dist = d

            # Picks move that minimises ghost's best escape distance
            if worst_dist < best_worst:
                best_worst  = worst_dist
                best_action = (move, best_steps)

        return best_action if best_action else self._fallback_greedy(map_state, my_pos, ghost_pos)
    
    def _lookup(self, map_state, src, dst):
        """Look up precomputed next step. Fall back to greedy if entry missing."""
        action = self.next_step.get((src, dst))
        if action:
            return action
        return self._fallback_greedy(map_state, src, dst)

    # BFS exploration
    def _bfs_explore(self, map_state, start):
        """
        BFS to reach the nearest unseen cell (-1).
        Falls back to any valid move if no fog cells remain.
        """
        queue   = [(start, None)]
        head    = 0
        visited = {start}

        while head < len(queue):
            pos, first_action = queue[head]
            head += 1
            r, c  = pos
            h, w  = map_state.shape

            if 0 <= r < h and 0 <= c < w and map_state[r, c] == -1:
                return first_action if first_action else self._step_toward(map_state, start, pos)

            for move in ALL_MOVES:
                cur = pos
                for steps in range(1, self.pacman_speed + 1):
                    dr, dc = move.value
                    nxt    = (cur[0] + dr, cur[1] + dc)
                    if not self._is_valid_position(nxt, map_state):
                        break
                    if nxt not in visited:
                        visited.add(nxt)
                        fa = first_action if first_action else (move, steps)
                        queue.append((nxt, fa))
                    cur = nxt

        return self._any_valid_move(map_state, start)

    # Fallback greedy (when cache missing or timeout)
    def _fallback_greedy(self, map_state, start, goal):
        """One-step greedy: pick the move+steps that minimises Manhattan to goal."""
        best_move, best_steps, best_dist = None, 1, float('inf')
        for move in ALL_MOVES:
            cur = start
            for steps in range(1, self.pacman_speed + 1):
                dr, dc = move.value
                nxt    = (cur[0] + dr, cur[1] + dc)
                if not self._is_valid_position(nxt, map_state):
                    break
                d = self._manhattan(nxt, goal)
                if d < best_dist:
                    best_dist, best_move, best_steps = d, move, steps
                cur = nxt
        return (best_move, best_steps) if best_move else (Move.STAY, 1)

    # ---------------- Helper methods ----------------
    
    def _step_toward(self, map_state, pos, target):
        return self._fallback_greedy(map_state, pos, target)

    def _any_valid_move(self, map_state, pos):
        for move in ALL_MOVES:
            steps = self._max_valid_steps(pos, move, map_state, self.pacman_speed)
            if steps > 0:
                return (move, steps)
        return (Move.STAY, 1)

    def _nearest_open(self, pos, map_state):
        """Snap a position to the nearest open cell (used when target is wall or fog)."""
        r, c = pos
        h, w = map_state.shape
        if 0 <= r < h and 0 <= c < w and map_state[r, c] == 0:
            return pos
        best, best_d = pos, float('inf')
        for cell in self.open_cells:
            d = self._manhattan(pos, cell)
            if d < best_d:
                best_d, best = d, cell
        return best
 
    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

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


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    
    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo
    """
    DEFAULT_CAPTURE_DISTANCE = 2
    DEFAULT_BURST_SPEED = 2.0
    DEFAULT_AVG_SPEED = 1.6
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Initialize any data structures you need
        self.capture_distance = kwargs.get("capture_distance_threshold", self.DEFAULT_CAPTURE_DISTANCE)
        self._enemy_history = deque(maxlen=10)
        self._speed_samples = deque(maxlen=10)
        self.burst_speed = self.DEFAULT_BURST_SPEED
        self.avg_speed = self.DEFAULT_AVG_SPEED
        self.prev_move = None
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
    
    def _update_speed_estimate(self, enemy_position):
        if enemy_position is None:
            return
        if self._enemy_history:
            prev = self._enemy_history[-1]
            step_dist = abs(enemy_position[0] - prev[0]) + abs(enemy_position[1] - prev[1])
            if step_dist > 0:
                self._speed_samples.append(step_dist)
                self.avg_speed = max(1.0, sum(self._speed_samples) / len(self._speed_samples))
                self.burst_speed = max(self.DEFAULT_BURST_SPEED, step_dist, self.avg_speed)
        self._enemy_history.append(enemy_position)

    def _get_bfs_distances(self, start_pos, map_state, walkable_flat=None):
        height, width = map_state.shape
        if walkable_flat is None:
            walkable_flat = (map_state == 0).ravel().tolist()

        dist_flat = [-1] * (height * width)
        sidx = start_pos[0] * width + start_pos[1]
        if not walkable_flat[sidx]:
            return np.array(dist_flat, dtype=np.int32).reshape(height, width)

        dist_flat[sidx] = 0
        queue = deque([sidx])
        while queue:
            idx = queue.popleft()
            r, c = divmod(idx, width)
            d = dist_flat[idx] + 1
            if r > 0:
                nidx = idx - width
                if walkable_flat[nidx] and dist_flat[nidx] == -1:
                    dist_flat[nidx] = d
                    queue.append(nidx)
            if r < height - 1:
                nidx = idx + width
                if walkable_flat[nidx] and dist_flat[nidx] == -1:
                    dist_flat[nidx] = d
                    queue.append(nidx)
            if c > 0:
                nidx = idx - 1
                if walkable_flat[nidx] and dist_flat[nidx] == -1:
                    dist_flat[nidx] = d
                    queue.append(nidx)
            if c < width - 1:
                nidx = idx + 1
                if walkable_flat[nidx] and dist_flat[nidx] == -1:
                    dist_flat[nidx] = d
                    queue.append(nidx)
        return np.array(dist_flat, dtype=np.int32).reshape(height, width)

    def _apply_move(self, pos, move, map_state):
        if move == Move.STAY:
            return pos
        dr, dc = move.value
        new_pos = (pos[0] + dr, pos[1] + dc)
        return new_pos if self._is_valid_position(new_pos, map_state) else pos

    def _degree(self, pos, map_state):
        r, c = pos
        height, width = map_state.shape
        count = 0
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and map_state[nr, nc] == 0:
                count += 1
        return count

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Pacman's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)
            
        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        # TODO: Implement your search algorithm here
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        self._update_speed_estimate(enemy_position)
        # Use current sighting, fallback to last known, or move randomly
        threat = enemy_position or self.last_known_enemy_pos
        
        valid_moves = [m for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
                       if self._is_valid_move(my_position, m, map_state)]
        if not valid_moves:
            return Move.STAY
        if threat is None:
            return random.choice(valid_moves)
        
        # Example: Simple evasive approach (replace with your algorithm)
        height, width = map_state.shape
        walkable = (map_state == 0)
        walkable_flat = walkable.ravel().tolist()

        p_dist = self._get_bfs_distances(threat, map_state, walkable_flat)

        candidates = []
        for move in valid_moves:
            next_pos = self._apply_move(my_position, move, map_state)
            pd_next = p_dist[next_pos[0], next_pos[1]]
            if pd_next == -1:
                margin = float('inf')
            else:
                margin = pd_next - self.burst_speed
            deg = self._degree(next_pos, map_state)
            candidates.append({"move": move, "pos": next_pos, "margin": margin, "deg": deg})

        safe = [c for c in candidates if c["margin"] >= self.capture_distance]
        pool = safe if safe else candidates

        if len(pool) == 1:
            chosen = pool[0]
            self.prev_move = chosen["move"]
            return chosen["move"]

        p_dist_valid = (p_dist != -1)
        p_dist_over_speed = p_dist / self.avg_speed
        base_mask = walkable & p_dist_valid

        for c in pool:
            g_dist = self._get_bfs_distances(c["pos"], map_state, walkable_flat)
            reachable_first = base_mask & (g_dist != -1)
            faster_than_pacman = reachable_first & (g_dist < p_dist_over_speed)
            if np.any(faster_than_pacman):
                c["haven"] = float(p_dist[faster_than_pacman].max())
            else:
                c["haven"] = -1.0

        best_haven = max(c["haven"] for c in pool)

        if best_haven > -1:
            tolerance = 1.0
            top = [c for c in pool if c["haven"] >= best_haven - tolerance]
        else:
            best_margin = max(c["margin"] for c in pool)
            top = [c for c in pool if c["margin"] >= best_margin - 1e-6]

        top.sort(key=lambda c: (c["margin"], c["deg"]), reverse=True)
        best_margin_tier = top[0]["margin"]
        best_deg_tier = max(c["deg"] for c in top if c["margin"] == best_margin_tier)
        finalists = [c for c in top if c["margin"] == best_margin_tier and c["deg"] == best_deg_tier]

        if self.prev_move is not None:
            for c in finalists:
                if c["move"] == self.prev_move:
                    self.prev_move = c["move"]
                    return c["move"]

        chosen = random.choice(finalists)
        self.prev_move = chosen["move"]
        return chosen["move"]
    
    # Helper methods
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0