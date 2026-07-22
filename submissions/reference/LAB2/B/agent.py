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
# performance benchmark
import time
_DIRS  = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
_INF   = 999


# ─────────────────────────────────────────────────────────────────────────────
# Shared utility functions
# ─────────────────────────────────────────────────────────────────────────────

def _nxt(pos, mv):
    dr, dc = mv.value
    return (pos[0] + dr, pos[1] + dc)

def _mh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _ok(pos, km):
    """In bounds and not a confirmed wall."""
    r, c = pos
    h, w = km.shape
    return 0 <= r < h and 0 <= c < w and km[r, c] != 1

def _bfs(start, km):
    """BFS from start on km. Returns {pos: dist}."""
    dist = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        cd  = dist[cur]
        for mv in _DIRS:
            nb = _nxt(cur, mv)
            if nb not in dist and _ok(nb, km):
                dist[nb] = cd + 1
                q.append(nb)
    return dist

def _build_apsp(km):
    """All-pairs BFS on traversable cells of km."""
    h, w   = km.shape
    cells  = [(r, c) for r in range(h) for c in range(w) if km[r, c] != 1]
    n      = len(cells)
    ix     = {p: i for i, p in enumerate(cells)}
    D      = np.full((n, n), _INF, dtype=np.int32)
    for i in range(n):
        D[i, i] = 0
    for i, start in enumerate(cells):
        q   = deque([start])
        vis = {start: 0}
        while q:
            cur = q.popleft()
            cd  = vis[cur]
            for mv in _DIRS:
                nb = _nxt(cur, mv)
                if nb not in vis and _ok(nb, km):
                    vis[nb] = cd + 1
                    q.append(nb)
        for pos, d in vis.items():
            if pos in ix:
                D[i, ix[pos]] = d
    return D, ix, cells

def _ray_visible(obs, tgt, km):
    """
    True if tgt is visible from obs in cross-shaped FOV (range 5, wall-blocked).
    Wall occlusion uses km (known_map).
    """
    r0, c0 = obs
    r1, c1 = tgt
    if r0 == r1 and c0 == c1:
        return True
    if r0 != r1 and c0 != c1:
        return False   # not same row or column
    if r0 == r1:
        if abs(c1 - c0) > 5:
            return False
        step = 1 if c1 > c0 else -1
        for dc in range(1, abs(c1 - c0)):
            if km[r0, c0 + dc * step] == 1:
                return False
    else:
        if abs(r1 - r0) > 5:
            return False
        step = 1 if r1 > r0 else -1
        for dr in range(1, abs(r1 - r0)):
            if km[r0 + dr * step, c0] == 1:
                return False
    return True
class PacmanAgent(BasePacmanAgent):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        
        self.map_shape = (21, 21) 
        self.learned_map = np.full(self.map_shape, -1, dtype=int)
        self.ghost_belief = np.ones(self.map_shape, dtype=float)
        
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        
        self.map_shape = map_state.shape
        self._update_memory(map_state)
        self._update_belief(my_position, enemy_position, map_state)
        
        # 1. Run BFS to find shortest paths to everywhere on our known map
        distances, parents = self._run_bfs(my_position)
        
        target_pos = None
        
        # 2. Decide Target
        if enemy_position is not None:
            target_pos = enemy_position
        else:
            max_prob = np.max(self.ghost_belief)
            if max_prob > 0.05:
                # Find the closest high-probability tile
                candidates = np.argwhere(self.ghost_belief == max_prob)
                target_pos = min([tuple(c) for c in candidates], key=lambda c: distances.get(c, 9999))
            else:
                # Hunt the closest fog/frontier
                unexplored = np.argwhere(self.learned_map == -1)
                if len(unexplored) > 0:
                    target_pos = min([tuple(c) for c in unexplored], key=lambda c: distances.get(c, 9999))
                else:
                    target_pos = my_position
                    
        # 3. Extract Path & Execute
        path = self._get_path_from_bfs(my_position, target_pos, parents)
        
        if not path:
            valid = self._get_optimistic_neighbors(my_position, include_stay=False)
            if valid:
                return (random.choice(valid)[2], 1)
            return (Move.STAY, 1)
            
        first_move = path[0]
        
        # Optimize step count based on pacman_speed
        steps = 0
        for m in path:
            if m == first_move and steps < self.pacman_speed:
                steps += 1
            else:
                break
                
        return (first_move, steps)

    def _update_memory(self, map_state: np.ndarray):
        visible_mask = map_state != -1
        self.learned_map[visible_mask] = map_state[visible_mask]
        
    def _update_belief(self, my_pos, enemy_pos, map_state):
        if enemy_pos is not None:
            self.ghost_belief.fill(0.0)
            self.ghost_belief[enemy_pos[0], enemy_pos[1]] = 1.0
        else:
            visible_mask = map_state != -1
            self.ghost_belief[visible_mask] = 0.0
            
            total_prob = np.sum(self.ghost_belief)
            if total_prob > 0:
                self.ghost_belief /= total_prob
            else:
                self.ghost_belief[self.learned_map != 1] = 1.0
                self.ghost_belief[visible_mask] = 0.0
                self.ghost_belief /= np.sum(self.ghost_belief)
                
        new_belief = np.zeros_like(self.ghost_belief)
        for r in range(self.map_shape[0]):
            for c in range(self.map_shape[1]):
                if self.ghost_belief[r, c] > 0:
                    neighbors = self._get_optimistic_neighbors((r, c), include_stay=True)
                    prob_per_neighbor = self.ghost_belief[r, c] / len(neighbors)
                    for nr, nc, _ in neighbors:
                        new_belief[nr, nc] += prob_per_neighbor
        self.ghost_belief = new_belief

    def _run_bfs(self, start):
        queue = deque([start])
        distances = {start: 0}
        parents = {start: None}
        while queue:
            curr = queue.popleft()
            for nr, nc, move in self._get_optimistic_neighbors(curr, include_stay=False):
                nxt = (nr, nc)
                if nxt not in distances:
                    distances[nxt] = distances[curr] + 1
                    parents[nxt] = (curr, move)
                    queue.append(nxt)
        return distances, parents

    def _get_path_from_bfs(self, start, target, parents):
        if target not in parents or target == start:
            return []
        path =[]
        curr = target
        while curr != start:
            curr, move = parents[curr]
            path.append(move)
        path.reverse()
        return path

    def _get_optimistic_neighbors(self, pos, include_stay=False):
        neighbors =[]
        if include_stay:
            neighbors.append((pos[0], pos[1], Move.STAY))
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = pos[0] + move.value[0], pos[1] + move.value[1]
            if 0 <= nr < self.map_shape[0] and 0 <= nc < self.map_shape[1]:
                if self.learned_map[nr, nc] != 1:
                    neighbors.append((nr, nc, move))
        return neighbors
    

# ============================================================================
# GHOST AGENT (The Hider - SuperGhost_V21x7)
# Merges Particle Filter & MCTS with Velocity Prediction
# ============================================================================

class GhostAgent(BaseGhostAgent):
    """
    Hider agent for limited-vision arena.

    Visible mode  → Deep Alpha-Beta Minimax evasion (Project 1 competition ghost v2).
    Hidden mode   → Flat MCTS over immediate moves, sampling Pacman positions
                    from a particle filter belief state (Enhanced with Velocity Prediction).
    """

    # ── tunable constants ──────────────────────────────────────────────────
    N_PARTICLES   = 250
    MCTS_C        = 1.41
    ROLLOUT_DEPTH = 10
    AB_BASE_DEPTH = 4
    APSP_INTERVAL = 20
    TIME_LIMIT    = 0.85

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed     = max(1, int(kwargs.get("pacman_speed",     2)))
        self.capture_distance = max(1, int(kwargs.get("capture_distance", 2)))
        self.name             = "SuperGhost_V21x7"

        # Map belief
        self.km              = None
        self._map_dirty      = False
        self._last_apsp_step = -100

        # APSP cache
        self.D        = None
        self.IX       = None
        self.cells    = None
        self.farthest = None
        self.exits_a  = None

        # Particle filter
        self.particles  = None
        self.last_seen  = None
        self.since_seen = 0
        
        # --- Velocity Memory (Ghost 21 integration) ---
        self.pacman_velocity = (0, 0)

        self._danger_dist = self.capture_distance + self.pacman_speed  # 4

    # ── public interface ───────────────────────────────────────────────────

    def step(self, map_state, my_position, enemy_position, step_number):
        t0 = time.time()

        self._merge_map(map_state)
        self._maybe_rebuild_apsp(step_number)
        
        # Calculate velocity vector if enemy just seen
        if enemy_position is not None:
            if self.last_seen is not None and self.since_seen == 0:
                vr = enemy_position[0] - self.last_seen[0]
                vc = enemy_position[1] - self.last_seen[1]
                self.pacman_velocity = (
                    (vr // abs(vr)) if vr != 0 else 0,
                    (vc // abs(vc)) if vc != 0 else 0
                )
            else:
                self.pacman_velocity = (0, 0)

        self._pf_update(enemy_position, my_position)

        if enemy_position is not None:
            self.last_seen  = enemy_position
            self.since_seen = 0
            return self._ab_evade(my_position, enemy_position, t0)
        else:
            self.since_seen += 1
            return self._mcts_hide(my_position, t0)

    # ── map maintenance ────────────────────────────────────────────────────

    def _merge_map(self, map_state):
        if self.km is None:
            self.km = np.full((21, 21), -1, dtype=np.int8)
        mask = map_state >= 0
        if np.any(self.km[mask] != map_state[mask].astype(np.int8)):
            self.km[mask]   = map_state[mask].astype(np.int8)
            self._map_dirty = True

    def _maybe_rebuild_apsp(self, step):
        if not self._map_dirty:
            return
        if step - self._last_apsp_step < self.APSP_INTERVAL:
            return
        self.D, self.IX, self.cells = _build_apsp(self.km)
        n = len(self.cells)
        self.farthest = np.max(self.D, axis=1).astype(np.int32)
        self.exits_a  = np.array(
            [sum(1 for mv in _DIRS if _ok(_nxt(p, mv), self.km)) for p in self.cells],
            dtype=np.int32
        )
        self._map_dirty      = False
        self._last_apsp_step = step

    # ── particle filter ────────────────────────────────────────────────────

    def _pf_update(self, enemy_pos, my_pos):
        if self.km is None:
            return
        traversable = [(r, c) for r in range(21) for c in range(21)
                       if self.km[r, c] != 1]
        if not traversable:
            return

        if self.particles is None:
            self.particles = (
                [enemy_pos] * self.N_PARTICLES if enemy_pos is not None
                else random.choices(traversable, k=self.N_PARTICLES)
            )
            return

        if enemy_pos is not None:
            # Hard reset
            core   = [enemy_pos] * (self.N_PARTICLES * 3 // 4)
            spread = []
            for _ in range(self.N_PARTICLES - len(core)):
                p = enemy_pos
                for _ in range(random.randint(1, self.pacman_speed * 2)):
                    opts = []
                    for mv in _DIRS:
                        cur = p
                        for _ in range(self.pacman_speed):
                            nxt = _nxt(cur, mv)
                            if not _ok(nxt, self.km):
                                break
                            opts.append(nxt)
                            cur = nxt
                    if opts:
                        p = random.choice(opts)
                spread.append(p)
            self.particles = core + spread
        else:
            # Propagate particles
            new_particles = []
            target = my_pos  
            
            for p in self.particles:
                rand_val = random.random()
                
                # 1. (Ghost 21 Logic) 40% chance Pacman continues moving with inertia
                if rand_val < 0.40 and self.pacman_velocity != (0, 0):
                    cur = p
                    for _ in range(self.pacman_speed):
                        nxt = (cur[0] + self.pacman_velocity[0], cur[1] + self.pacman_velocity[1])
                        if _ok(nxt, self.km):
                            cur = nxt
                        else:
                            break 
                    p = cur

                # 2. (Ghost 7 Logic) 40% chance Pacman chases ghost directly
                elif rand_val < 0.80:
                    best, best_d = p, _mh(p, target)
                    for mv in _DIRS:
                        cur = p
                        for _ in range(self.pacman_speed):
                            nxt = _nxt(cur, mv)
                            if not _ok(nxt, self.km):
                                break
                            d = _mh(nxt, target)
                            if d < best_d:
                                best_d, best = d, nxt
                            cur = nxt
                    p = best
                    
                # 3. 20% random exploration/drift
                else:
                    opts = [_nxt(p, mv) for mv in _DIRS if _ok(_nxt(p, mv), self.km)]
                    if opts:
                        p = random.choice(opts)

                # Rejection: particle visible from my_pos but Pacman not observed
                if _ray_visible(my_pos, p, self.km):
                    p = random.choice(self.particles)
                new_particles.append(p)
                
            self.particles = new_particles

    def _pf_best(self):
        if not self.particles:
            return self.last_seen
        return Counter(self.particles).most_common(1)[0][0]

    # ── Alpha-Beta evasion (visible Pacman) ────────────────────────────────

    def _ab_evade(self, ghost, pacman, t0):
        d = self._d(ghost, pacman)

        if d <= self._danger_dist:
            depth = self.AB_BASE_DEPTH + 1
        elif d > 14:
            depth = max(self.AB_BASE_DEPTH - 1, 2)
        else:
            depth = self.AB_BASE_DEPTH

        moves = self._ghost_legal(ghost)
        if not moves:
            return Move.STAY

        # Move ordering: distance-maximising first
        moves.sort(key=lambda x: -self._d(x[1], pacman))

        best_mv  = Move.STAY
        best_val = -float("inf")
        alpha    = -float("inf")
        beta     =  float("inf")

        for mv, new_ghost in moves:
            if time.time() - t0 > self.TIME_LIMIT:
                break
            val = self._ab(new_ghost, pacman, depth - 1, alpha, beta, is_ghost=False)
            if val > best_val:
                best_val, best_mv = val, mv
            alpha = max(alpha, best_val)

        return best_mv

    def _ab(self, ghost, pacman, depth, alpha, beta, is_ghost):
        if _mh(ghost, pacman) < self.capture_distance:
            return -100_000.0
        if depth == 0:
            return self._eval(ghost, pacman)

        if is_ghost:
            moves = self._ghost_legal(ghost)
            if not moves:
                return self._eval(ghost, pacman)
            moves.sort(key=lambda x: -self._d(x[1], pacman))
            best = -float("inf")
            for _, new_ghost in moves:
                val  = self._ab(new_ghost, pacman, depth - 1, alpha, beta, False)
                best = max(best, val)
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
            return best
        else:
            # Pacman MIN: consider all reachable positions (worst case)
            pac_positions = self._pac_next(pacman)
            pac_positions.sort(key=lambda p: self._d(p, ghost))
            best = float("inf")
            for new_pac in pac_positions:
                val  = self._ab(ghost, new_pac, depth - 1, alpha, beta, True)
                best = min(best, val)
                beta = min(beta, best)
                if beta <= alpha:
                    break
            return best

    def _eval(self, ghost, pacman):
        if _mh(ghost, pacman) < self.capture_distance:
            return -100_000.0

        gi = self.IX.get(ghost)  if self.IX else None
        pi = self.IX.get(pacman) if self.IX else None

        if gi is None:
            return float(_mh(ghost, pacman) * 10)

        d        = int(self.D[gi, pi]) if pi is not None else _mh(ghost, pacman)
        exits    = int(self.exits_a[gi])
        farthest = int(self.farthest[gi])

        score  = d        * 10.0
        score += exits    * 5.0
        score += farthest * 0.5

        if d <= self._danger_dist:
            score += d * 15.0     # amplify distance term in danger zone

        if exits == 1 and d < self._danger_dist + 2:
            score -= 500.0
        if exits == 0:
            score -= 2000.0

        return score

    # ── MCTS hiding (hidden Pacman) ────────────────────────────────────────

    def _mcts_hide(self, ghost, t0):
        """
        Flat UCB1 bandit over ghost's legal moves.
        Each iteration samples a Pacman hypothesis from particles and
        simulates a rollout to estimate survival probability.
        """
        if not self.particles:
            return self._flee_fallback(ghost)

        moves = self._ghost_legal(ghost)
        if not moves:
            return Move.STAY

        n_arms = len(moves)
        wins   = [0.0] * n_arms
        visits = [0]   * n_arms
        total  = 0

        while time.time() - t0 < self.TIME_LIMIT * 0.9:
            pac_hyp = random.choice(self.particles)

            # UCB1 selection
            if total < n_arms:
                arm = total
            else:
                best_ucb = -float("inf")
                arm      = 0
                log_t    = math.log(total)
                for i in range(n_arms):
                    if visits[i] == 0:
                        arm = i
                        break
                    ucb = wins[i] / visits[i] + self.MCTS_C * math.sqrt(log_t / visits[i])
                    if ucb > best_ucb:
                        best_ucb, arm = ucb, i

            _, new_ghost = moves[arm]
            reward       = self._rollout(new_ghost, pac_hyp)

            wins[arm]   += reward
            visits[arm] += 1
            total       += 1

        best_arm = max(range(n_arms), key=lambda i: wins[i] / visits[i] if visits[i] > 0 else 0.0)
        return moves[best_arm][0]

    def _rollout(self, ghost, pacman):
        """
        Simulate ghost evasion vs. greedy Pacman for ROLLOUT_DEPTH steps.
        Returns survival score in [0, 1].
        """
        for step in range(self.ROLLOUT_DEPTH):
            if _mh(ghost, pacman) < self.capture_distance:
                # Penalise early capture more heavily
                return step / self.ROLLOUT_DEPTH * 0.2

            # Ghost evades greedily (max distance from Pacman)
            ghost_opts = self._ghost_legal(ghost)
            if ghost_opts:
                ghost_opts.sort(key=lambda x: -_mh(x[1], pacman))
                _, ghost = ghost_opts[0]

            # Pacman chases greedily (speed-aware)
            pac_opts = []
            for mv in _DIRS:
                cur = pacman
                for _ in range(self.pacman_speed):
                    nxt = _nxt(cur, mv)
                    if not _ok(nxt, self.km):
                        break
                    pac_opts.append(nxt)
                    cur = nxt
            if pac_opts:
                pac_opts.sort(key=lambda p: _mh(p, ghost))
                pacman = pac_opts[0]

        # Survived all steps — reward proportional to final distance
        d = _mh(ghost, pacman)
        return min(1.0, 0.4 + d / 20.0)

    def _flee_fallback(self, ghost):
        """Fallback when no particle info: flee from last known Pacman position."""
        if self.last_seen is None:
            moves = self._ghost_legal(ghost)
            return random.choice(moves)[0] if moves else Move.STAY
        moves = self._ghost_legal(ghost)
        if not moves:
            return Move.STAY
        moves.sort(key=lambda x: -self._d(x[1], self.last_seen))
        return moves[0][0]

    # ── move generators ────────────────────────────────────────────────────

    def _ghost_legal(self, pos):
        """All (move, new_pos) for ghost including STAY."""
        result = [(Move.STAY, pos)]
        for mv in _DIRS:
            nxt = _nxt(pos, mv)
            if _ok(nxt, self.km):
                result.append((mv, nxt))
        return result

    def _pac_next(self, start):
        """All positions Pacman can reach in one turn (speed-aware)."""
        positions = {start}
        for mv in _DIRS:
            cur = start
            for _ in range(self.pacman_speed):
                nxt = _nxt(cur, mv)
                if not _ok(nxt, self.km):
                    break
                positions.add(nxt)
                cur = nxt
        return list(positions)

    def _d(self, a, b):
        if self.D is None or a not in self.IX or b not in self.IX:
            return _mh(a, b)
        return int(self.D[self.IX[a], self.IX[b]])