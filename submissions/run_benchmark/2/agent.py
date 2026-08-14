"""
Student submission for the "Blind Adversary" lab (partial observability
Hide and Seek Arena).

PacmanAgent (Seeker): Belief-State tracking + Monte Carlo Tree Search.
GhostAgent (Hider): kept as the reference evasive strategy — only the
seeker was requested to be upgraded. Feel free to replace this with
your own hider algorithm.
"""

import sys
import time
import random
import math
from pathlib import Path
from collections import deque

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


DIRECTIONS = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
ALL_ACTIONS = DIRECTIONS + [Move.STAY]

TIME_BUDGET = 0.85     # seconds; keep margin under the 1s hard limit
MAX_PARTICLES = 5      # how many candidate ghost locations to consider
ROLLOUT_DEPTH = 6      # plies (real turns) simulated per MCTS rollout
UCB_C = 1.4            # UCB1 exploration constant


class MCTSNode:
    """One node of a per-particle search tree.

    A node represents "it is our turn, we are at pacman_pos, and (in
    this particle's hypothetical world) the ghost is at ghost_pos,
    `depth` turns from now".
    """
    __slots__ = ("pacman_pos", "ghost_pos", "depth", "N", "W", "children", "untried")

    def __init__(self, pacman_pos, ghost_pos, depth):
        self.pacman_pos = pacman_pos
        self.ghost_pos = ghost_pos
        self.depth = depth
        self.N = 0            # visit count
        self.W = 0.0           # total backed-up reward
        self.children = {}     # action -> MCTSNode
        self.untried = list(ALL_ACTIONS)


class PacmanAgent(BasePacmanAgent):
    """
    Seeker agent for the Blind Adversary lab.

    Strategy: Belief-State tracking + Monte Carlo Tree Search (MCTS).

    Because the ghost is only intermittently visible, we keep a
    probability distribution ("belief") over where it currently is.
    Every step we:

      1. Update our persistent memory of the maze. Walls/paths we have
         already observed stay known even after we look away (the raw
         map_state given each step only reflects our *current* field
         of view; we merge it into `self.memory` so knowledge
         accumulates over the game).
      2. Predict how the belief would have spread since last step
         (the ghost may have taken one step, in any direction, or
         stayed put).
      3. Correct the belief with this step's observation: collapse it
         to the seen cell if the ghost is visible, otherwise zero out
         probability mass on every cell we can currently see (since
         the ghost is clearly not there).
      4. Sample a handful of the most likely ghost locations
         ("particles") from the belief.
      5. Run MCTS from each particle to score the 5 candidate actions
         (UP / DOWN / LEFT / RIGHT / STAY), simulating a simple
         evasive ghost as our opponent model.
      6. Combine the belief-weighted scores from every particle and
         take the best-scoring action.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Belief-MCTS Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))

        self.shape = None
        self.memory = None          # persistent known map: -1/0/1
        self.belief = None          # persistent probability grid
        self.last_known_enemy_pos = None
        self.last_seen_step = -1
        self.recent_positions = deque(maxlen=8)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def step(self, map_state: np.ndarray, my_position: tuple,
              enemy_position: tuple, step_number: int):
        t0 = time.time()
        self.recent_positions.append(my_position)

        self._ensure_initialized(map_state)
        self._update_memory(map_state)
        self._update_belief(map_state, my_position, enemy_position)

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.last_seen_step = step_number

        particles = self._sample_particles(my_position)

        if not particles:
            # No idea at all where the ghost could be (shouldn't really
            # happen once the belief has been initialized) -> explore.
            return self._explore(my_position, map_state)

        action = self._plan(my_position, particles, t0)
        return self._materialize_action(my_position, action, map_state)

    # ------------------------------------------------------------------
    # Memory / belief maintenance
    # ------------------------------------------------------------------
    def _ensure_initialized(self, map_state):
        if self.shape is None:
            self.shape = map_state.shape
            self.memory = np.full(self.shape, -1, dtype=int)
            self.belief = np.zeros(self.shape, dtype=float)

    def _update_memory(self, map_state):
        seen = map_state != -1
        self.memory[seen] = map_state[seen]

    def _update_belief(self, map_state, my_position, enemy_position):
        visible_mask = map_state != -1

        if self.belief.sum() <= 0:
            # First-ever belief: uniform over everything we currently
            # know to be empty (falls back to the visible cells).
            known_empty = (self.memory == 0)
            if known_empty.sum() == 0:
                known_empty = visible_mask  # extreme fallback
            self.belief[:] = 0.0
            self.belief[known_empty] = 1.0
        else:
            # 1) Predict: diffuse belief to neighboring passable cells,
            #    modelling "the ghost may have taken one step".
            self.belief = self._diffuse(self.belief)

        # 2) Correct with this step's observation.
        if enemy_position is not None:
            self.belief[:] = 0.0
            self.belief[enemy_position] = 1.0
        else:
            self.belief[visible_mask] = 0.0
            # The ghost cannot occupy our own current cell either.
            self.belief[my_position] = 0.0

        total = self.belief.sum()
        if total > 0:
            self.belief /= total
        else:
            # We lost track completely: reset to uniform over all known
            # empty, currently-invisible cells.
            known_empty = (self.memory == 0) & (~visible_mask)
            if known_empty.sum() == 0:
                known_empty = (self.memory == 0)
            self.belief[:] = 0.0
            if known_empty.sum() > 0:
                self.belief[known_empty] = 1.0 / known_empty.sum()

    def _diffuse(self, belief):
        """One step of belief propagation assuming the ghost takes a
        single random step (including possibly staying put)."""
        new_belief = np.zeros_like(belief)
        rows, cols = np.nonzero(belief)
        for r, c in zip(rows, cols):
            p = belief[r, c]
            if p <= 0:
                continue
            targets = [(r, c)]  # staying is one possibility
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if self._passable_for_planning((nr, nc)):
                    targets.append((nr, nc))
            share = p / len(targets)
            for (tr, tc) in targets:
                new_belief[tr, tc] += share
        return new_belief

    # ------------------------------------------------------------------
    # Particle sampling
    # ------------------------------------------------------------------
    def _sample_particles(self, my_position):
        flat_idx = np.argsort(self.belief, axis=None)[::-1]
        particles = []
        weights = []
        for idx in flat_idx:
            if len(particles) >= MAX_PARTICLES:
                break
            r, c = np.unravel_index(idx, self.belief.shape)
            p = self.belief[r, c]
            if p <= 1e-6:
                break
            if (r, c) == tuple(my_position):
                continue
            particles.append((int(r), int(c)))
            weights.append(float(p))

        if not particles and self.last_known_enemy_pos is not None:
            particles = [self.last_known_enemy_pos]
            weights = [1.0]

        total_w = sum(weights) or 1.0
        weights = [w / total_w for w in weights]
        return list(zip(particles, weights))

    # ------------------------------------------------------------------
    # Planning: belief-weighted MCTS
    # ------------------------------------------------------------------
    def _plan(self, my_position, particles, t0):
        scores = {a: 0.0 for a in ALL_ACTIONS}
        remaining = TIME_BUDGET - (time.time() - t0)
        if remaining <= 0 or not particles:
            return self._greedy_action(my_position, particles)

        per_particle_budget = max(remaining / len(particles), 0.01)

        for (ghost_pos, weight) in particles:
            particle_deadline = time.time() + per_particle_budget
            root = MCTSNode(tuple(my_position), ghost_pos, 0)
            self._run_mcts(root, particle_deadline)
            for action, child in root.children.items():
                if child.N > 0:
                    next_pos = self._pacman_result(my_position, action)
                    penalty = 0.0
                    if next_pos in self.recent_positions:
                        penalty = 0.3

                    scores[action] += weight * (child.W / child.N) - penalty

        best_action = max(scores, key=scores.get)
        return best_action

    def _run_mcts(self, root, deadline):
        # Always run at least a handful of iterations even if the
        # clock is razor-thin, so every root action gets expanded once.
        iterations = 0
        while time.time() < deadline or iterations < len(ALL_ACTIONS):
            path = self._tree_policy(root)
            leaf = path[-1]
            reward = self._rollout(leaf.pacman_pos, leaf.ghost_pos, leaf.depth)
            for node in path:
                node.N += 1
                node.W += reward
            iterations += 1
            if iterations > 2000:
                break  # hard safety cap

    def _tree_policy(self, node):
        path = [node]
        while node.depth < ROLLOUT_DEPTH and not self._is_capture(node.pacman_pos, node.ghost_pos):
            if node.untried:
                action = node.untried.pop(random.randrange(len(node.untried)))
                new_pacman = self._pacman_result(node.pacman_pos, action)
                new_ghost = self._ghost_reaction(node.ghost_pos, new_pacman)
                child = MCTSNode(new_pacman, new_ghost, node.depth + 1)
                node.children[action] = child
                path.append(child)
                return path
            else:
                if not node.children:
                    break
                action = self._best_ucb_action(node)
                node = node.children[action]
                path.append(node)
        return path

    def _best_ucb_action(self, node):
        best_action, best_value = None, -float("inf")
        log_parent = math.log(node.N + 1)
        for action, child in node.children.items():
            if child.N == 0:
                value = float("inf")
            else:
                exploit = child.W / child.N
                explore = UCB_C * math.sqrt(log_parent / child.N)
                value = exploit + explore
            if value > best_value:
                best_value, best_action = value, action
        return best_action

    def _rollout(self, pacman_pos, ghost_pos, depth):
        p, g = pacman_pos, ghost_pos
        d = depth
        while d < ROLLOUT_DEPTH:
            if self._is_capture(p, g):
                return 1.0 - 0.05 * d
            action = self._greedy_direction(p, g)
            p = self._pacman_result(p, action)
            if self._is_capture(p, g):
                return 1.0 - 0.05 * (d + 1)
            g = self._ghost_reaction(g, p)
            d += 1
        dist = self._manhattan(p, g)
        max_dist = sum(self.shape)
        return -dist / max_dist

    # ------------------------------------------------------------------
    # Simple heuristic policies used inside simulation only
    # ------------------------------------------------------------------
    def _greedy_direction(self, pos, target):
        row_diff = target[0] - pos[0]
        col_diff = target[1] - pos[1]
        prefs = []
        if row_diff > 0:
            prefs.append(Move.DOWN)
        elif row_diff < 0:
            prefs.append(Move.UP)
        if col_diff > 0:
            prefs.append(Move.RIGHT)
        elif col_diff < 0:
            prefs.append(Move.LEFT)
        for move in prefs:
            nr, nc = pos[0] + move.value[0], pos[1] + move.value[1]
            if self._passable_for_planning((nr, nc)):
                return move
        for move in DIRECTIONS:
            nr, nc = pos[0] + move.value[0], pos[1] + move.value[1]
            if self._passable_for_planning((nr, nc)):
                return move
        return Move.STAY

    def _ghost_reaction(self, ghost_pos, pacman_pos):
        """Assumed opponent model used only for planning: a simple
        evasive ghost that moves away from Pacman (mirrors the
        reference GhostAgent behaviour)."""
        row_diff = ghost_pos[0] - pacman_pos[0]
        col_diff = ghost_pos[1] - pacman_pos[1]
        prefs = []
        if row_diff > 0:
            prefs.append(Move.DOWN)
        elif row_diff < 0:
            prefs.append(Move.UP)
        if col_diff > 0:
            prefs.append(Move.RIGHT)
        elif col_diff < 0:
            prefs.append(Move.LEFT)
        for move in prefs:
            nr, nc = ghost_pos[0] + move.value[0], ghost_pos[1] + move.value[1]
            if self._passable_for_planning((nr, nc)):
                return (nr, nc)
        for move in DIRECTIONS:
            nr, nc = ghost_pos[0] + move.value[0], ghost_pos[1] + move.value[1]
            if self._passable_for_planning((nr, nc)):
                return (nr, nc)
        return ghost_pos

    def _pacman_result(self, pos, action):
        if action == Move.STAY:
            return pos
        cur = pos
        for _ in range(self.pacman_speed):
            nxt = (cur[0] + action.value[0], cur[1] + action.value[1])
            if not self._passable_for_planning(nxt):
                break
            cur = nxt
        return cur

    def _greedy_action(self, my_position, particles):
        if not particles:
            return Move.STAY
        target, _ = particles[0]
        return self._greedy_direction(my_position, target)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _is_capture(self, pacman_pos, ghost_pos):
        return self._manhattan(pacman_pos, ghost_pos) < 2

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _passable_for_planning(self, pos):
        """Optimistic passability used for lookahead beyond current
        vision: known walls are blocked, everything else (known-empty
        or still-unknown/foggy) is treated as passable. This is only
        ever used for hypothetical planning, never for the actual move
        we execute."""
        r, c = pos
        h, w = self.shape
        if r < 0 or r >= h or c < 0 or c >= w:
            return False
        return self.memory[r, c] != 1

    def _is_valid_position(self, pos, map_state):
        """Strict validity check used for the actual move we execute:
        only cells we can currently confirm as empty (from the live,
        freshly-observed map_state) are considered safe."""
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0

    def _max_valid_steps(self, pos, move, map_state, desired_steps):
        steps = 0
        max_steps = min(self.pacman_speed, max(1, desired_steps))
        current = pos
        for _ in range(max_steps):
            dr, dc = move.value
            nxt = (current[0] + dr, current[1] + dc)
            if not self._is_valid_position(nxt, map_state):
                break
            steps += 1
            current = nxt
        return steps

    def _materialize_action(self, my_position, action, map_state):
        """Turn the abstract direction chosen by MCTS into a concrete
        (Move, steps) tuple, re-validated against the live map_state
        (immediate neighbours are always within our field of view, so
        this check is always based on fresh information)."""
        if action == Move.STAY:
            return (Move.STAY, 1)
        steps = self._max_valid_steps(my_position, action, map_state, self.pacman_speed)
        if steps > 0:
            return (action, steps)
        # Chosen direction turned out to be immediately blocked
        # -> fall back to any other valid direction.
        for move in DIRECTIONS:
            steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
            if steps > 0:
                return (move, steps)
        return (Move.STAY, 1)

    def _explore(self, my_position, map_state):
        all_moves = list(DIRECTIONS)
        random.shuffle(all_moves)
        for move in all_moves:
            steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
            if steps > 0:
                return (move, steps)
        return (Move.STAY, 1)


class GhostAgent(BaseGhostAgent):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Kinetic APSP Hider"
        self.p_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.cap_dist = max(1, int(kwargs.get("capture_distance", 2)))
        self.danger_zone = self.cap_dist + self.p_speed

        # Environment & Precomputations
        self.world_map = None
        self.needs_cache_update = False
        self.last_cache_step = -999
        self.dist_matrix = None
        self.coord_to_idx = None
        self.mobility_scores = None
        self.max_clearance = None

        # Tracking (Particle Filter)
        self.hypotheses = []
        self.adversary_momentum = (0, 0)
        self.last_known_adversary = None
        self.turns_unseen = 0

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        clock_start = time.time()

        self._integrate_vision(map_state)
        self._refresh_pathing_cache(step_number)
        self._calculate_momentum(enemy_position)
        self._update_belief_system(enemy_position, my_position)

        if enemy_position is not None:
            self.last_known_adversary = enemy_position
            self.turns_unseen = 0
            return self._minimax_evasion(my_position, enemy_position, clock_start)
        else:
            self.turns_unseen += 1
            return self._bandit_survival(my_position, clock_start)

    # -- Core Map & Cache Logic --

    def _integrate_vision(self, map_state):
        if self.world_map is None:
            self.world_map = np.full(map_state.shape, -1, dtype=np.int8)
        mask = map_state >= 0
        if (self.world_map[mask] != map_state[mask]).any():
            self.world_map[mask] = map_state[mask].astype(np.int8)
            self.needs_cache_update = True

    def _refresh_pathing_cache(self, step):
        if not self.needs_cache_update or (step - self.last_cache_step) < 20:
            return
            
        h, w = self.world_map.shape
        walkable = [(r, c) for r in range(h) for c in range(w) if self.world_map[r, c] != 1]
        n_nodes = len(walkable)
        
        self.coord_to_idx = {pos: i for i, pos in enumerate(walkable)}
        self.dist_matrix = np.full((n_nodes, n_nodes), 999, dtype=np.int32)
        
        # APSP Construction
        for idx, start_node in enumerate(walkable):
            self.dist_matrix[idx, idx] = 0
            queue = deque([start_node])
            visited = {start_node: 0}
            
            while queue:
                curr = queue.popleft()
                curr_dist = visited[curr]
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    neighbor = (curr[0] + dr, curr[1] + dc)
                    if neighbor not in visited and self._cell_ok(neighbor):
                        visited[neighbor] = curr_dist + 1
                        queue.append(neighbor)
                        
            for pos, d in visited.items():
                if pos in self.coord_to_idx:
                    self.dist_matrix[idx, self.coord_to_idx[pos]] = d

        # Derived Analytics (Dead-ends and open space)
        self.max_clearance = np.max(self.dist_matrix, axis=1).astype(np.int32)
        self.mobility_scores = np.zeros(n_nodes, dtype=np.int32)
        for pos, idx in self.coord_to_idx.items():
            exits = sum(1 for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)] 
                        if self._cell_ok((pos[0]+dr, pos[1]+dc)))
            self.mobility_scores[idx] = exits

        self.needs_cache_update = False
        self.last_cache_step = step

    # -- Particle Filter --

    def _calculate_momentum(self, current_enemy_pos):
        if current_enemy_pos and self.last_known_adversary and self.turns_unseen == 0:
            dr = current_enemy_pos[0] - self.last_known_adversary[0]
            dc = current_enemy_pos[1] - self.last_known_adversary[1]
            self.adversary_momentum = (dr // abs(dr) if dr else 0, dc // abs(dc) if dc else 0)
        else:
            self.adversary_momentum = (0, 0)

    def _update_belief_system(self, enemy_pos, my_pos):
        if self.world_map is None: return
        
        if not self.hypotheses:
            if enemy_pos:
                self.hypotheses = [enemy_pos] * 200
            else:
                spaces = [(r, c) for r in range(21) for c in range(21) if self.world_map[r, c] != 1]
                self.hypotheses = random.choices(spaces, k=200) if spaces else [my_pos]
            return

        if enemy_pos:
            # Resample heavily around absolute truth
            base = [enemy_pos] * 150
            scatter = []
            for _ in range(50):
                sim_pos = enemy_pos
                for _ in range(random.randint(1, self.p_speed * 2)):
                    options = self._simulate_seeker_reach(sim_pos)
                    if options: sim_pos = random.choice(options)
                scatter.append(sim_pos)
            self.hypotheses = base + scatter
        else:
            # Advance particles through time
            advanced = []
            for particle in self.hypotheses:
                behavior_roll = random.random()
                
                # Momentum-based movement
                if behavior_roll <= 0.40 and self.adversary_momentum != (0, 0):
                    curr = particle
                    for _ in range(self.p_speed):
                        nxt = (curr[0] + self.adversary_momentum[0], curr[1] + self.adversary_momentum[1])
                        if self._cell_ok(nxt): curr = nxt
                        else: break
                    particle = curr
                    
                # Aggressive chasing
                elif behavior_roll <= 0.80:
                    best_dst = abs(particle[0] - my_pos[0]) + abs(particle[1] - my_pos[1])
                    best_pos = particle
                    for reach in self._simulate_seeker_reach(particle):
                        d = abs(reach[0] - my_pos[0]) + abs(reach[1] - my_pos[1])
                        if d < best_dst:
                            best_dst, best_pos = d, reach
                    particle = best_pos
                    
                # Random drift
                else:
                    opts = [(particle[0]+dr, particle[1]+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)] 
                            if self._cell_ok((particle[0]+dr, particle[1]+dc))]
                    if opts: particle = random.choice(opts)

                # Filter out impossible particles (ones we should be able to see)
                if self._line_of_sight(my_pos, particle):
                    particle = random.choice(self.hypotheses)
                advanced.append(particle)
                
            self.hypotheses = advanced

    # -- Visible Strategy (Minimax) --

    def _minimax_evasion(self, ghost_pos, pacman_pos, clock_start):
        dist = self._fast_dist(ghost_pos, pacman_pos)
        lookahead = 5 if dist <= self.danger_zone else (2 if dist > 14 else 4)
        
        legal_moves = self._get_ghost_options(ghost_pos)
        if not legal_moves: return Move.STAY

        # Optimizer: Evaluate moves that increase distance first
        legal_moves.sort(key=lambda m: self._fast_dist(m[1], pacman_pos), reverse=True)

        best_action = Move.STAY
        best_eval = -float("inf")
        alpha, beta = -float("inf"), float("inf")

        for action, next_g in legal_moves:
            if time.time() - clock_start > 0.82: break
            score = self._pacman_turn_min(next_g, pacman_pos, lookahead - 1, alpha, beta)
            if score > best_eval:
                best_eval, best_action = score, action
            alpha = max(alpha, best_eval)

        return best_action

    def _ghost_turn_max(self, ghost, pacman, depth, alpha, beta):
        if abs(ghost[0] - pacman[0]) + abs(ghost[1] - pacman[1]) < self.cap_dist: return -99999.0
        if depth == 0: return self._assess_survival_state(ghost, pacman)

        moves = self._get_ghost_options(ghost)
        if not moves: return self._assess_survival_state(ghost, pacman)
        
        moves.sort(key=lambda m: self._fast_dist(m[1], pacman), reverse=True)
        max_score = -float("inf")
        
        for _, nxt_ghost in moves:
            score = self._pacman_turn_min(nxt_ghost, pacman, depth - 1, alpha, beta)
            max_score = max(max_score, score)
            alpha = max(alpha, max_score)
            if beta <= alpha: break
        return max_score

    def _pacman_turn_min(self, ghost, pacman, depth, alpha, beta):
        if abs(ghost[0] - pacman[0]) + abs(ghost[1] - pacman[1]) < self.cap_dist: return -99999.0
        if depth == 0: return self._assess_survival_state(ghost, pacman)

        pac_reaches = self._simulate_seeker_reach(pacman)
        pac_reaches.sort(key=lambda p: self._fast_dist(p, ghost))
        
        min_score = float("inf")
        for nxt_pacman in pac_reaches:
            score = self._ghost_turn_max(ghost, nxt_pacman, depth - 1, alpha, beta)
            min_score = min(min_score, score)
            beta = min(beta, min_score)
            if beta <= alpha: break
        return min_score

    def _assess_survival_state(self, ghost, pacman):
        manhattan = abs(ghost[0] - pacman[0]) + abs(ghost[1] - pacman[1])
        if manhattan < self.cap_dist: return -99999.0

        g_idx = self.coord_to_idx.get(ghost) if self.coord_to_idx else None
        p_idx = self.coord_to_idx.get(pacman) if self.coord_to_idx else None

        if g_idx is None: return float(manhattan * 10)

        true_dist = int(self.dist_matrix[g_idx, p_idx]) if p_idx is not None else manhattan
        exits = int(self.mobility_scores[g_idx])
        clearance = int(self.max_clearance[g_idx])

        # Mathematical weighting
        evaluation = (true_dist * 10.0) + (exits * 5.0) + (clearance * 0.5)

        if true_dist <= self.danger_zone:
            evaluation += true_dist * 15.0

        # Dead-end penalties
        if exits == 1 and true_dist < self.danger_zone + 2: evaluation -= 500.0
        if exits == 0: evaluation -= 2000.0

        return evaluation

    # -- Hidden Strategy (Bandits) --

    def _bandit_survival(self, ghost, clock_start):
        if not self.hypotheses:
            return self._emergency_flee(ghost)

        options = self._get_ghost_options(ghost)
        if not options: return Move.STAY

        stats = {act: {"rew": 0.0, "vis": 0} for act, _ in options}
        total_sims = 0

        while time.time() - clock_start < 0.75:
            assumed_pacman = random.choice(self.hypotheses)
            
            # Select Arm
            selected_act, selected_pos = None, None
            if total_sims < len(options):
                selected_act, selected_pos = options[total_sims]
            else:
                best_ucb, log_t = -float("inf"), math.log(total_sims)
                for act, pos in options:
                    v = stats[act]["vis"]
                    if v == 0:
                        selected_act, selected_pos = act, pos
                        break
                    val = (stats[act]["rew"] / v) + 1.41 * math.sqrt(log_t / v)
                    if val > best_ucb:
                        best_ucb, selected_act, selected_pos = val, act, pos

            # Simulate
            outcome = self._simulate_chase_sequence(selected_pos, assumed_pacman)

            # Backpropagate
            stats[selected_act]["rew"] += outcome
            stats[selected_act]["vis"] += 1
            total_sims += 1

        return max(stats.keys(), key=lambda k: stats[k]["rew"] / max(1, stats[k]["vis"]))

    def _simulate_chase_sequence(self, ghost_pos, pacman_pos):
        for step in range(10):
            if abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1]) < self.cap_dist:
                return (step / 10.0) * 0.2

            g_moves = self._get_ghost_options(ghost_pos)
            if g_moves:
                g_moves.sort(key=lambda m: -(abs(m[1][0] - pacman_pos[0]) + abs(m[1][1] - pacman_pos[1])))
                ghost_pos = g_moves[0][1]

            p_moves = self._simulate_seeker_reach(pacman_pos)
            if p_moves:
                p_moves.sort(key=lambda p: abs(p[0] - ghost_pos[0]) + abs(p[1] - ghost_pos[1]))
                pacman_pos = p_moves[0]

        d = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
        return min(1.0, 0.4 + (d / 20.0))

    def _emergency_flee(self, ghost):
        moves = self._get_ghost_options(ghost)
        if not moves: return Move.STAY
        if not self.last_known_adversary: return random.choice(moves)[0]
        moves.sort(key=lambda m: self._fast_dist(m[1], self.last_known_adversary), reverse=True)
        return moves[0][0]

    # -- Utility Helpers --

    def _get_ghost_options(self, pos):
        valid = [(Move.STAY, pos)]
        for dr, dc, act in [(-1,0,Move.UP), (1,0,Move.DOWN), (0,-1,Move.LEFT), (0,1,Move.RIGHT)]:
            if self._cell_ok((pos[0]+dr, pos[1]+dc)):
                valid.append((act, (pos[0]+dr, pos[1]+dc)))
        return valid

    def _simulate_seeker_reach(self, start):
        reaches = {start}
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            curr = start
            for _ in range(self.p_speed):
                nxt = (curr[0] + dr, curr[1] + dc)
                if not self._cell_ok(nxt): break
                reaches.add(nxt)
                curr = nxt
        return list(reaches)

    def _cell_ok(self, pos):
        if 0 <= pos[0] < self.world_map.shape[0] and 0 <= pos[1] < self.world_map.shape[1]:
            return self.world_map[pos] != 1
        return False

    def _fast_dist(self, p1, p2):
        if self.dist_matrix is not None and p1 in self.coord_to_idx and p2 in self.coord_to_idx:
            return int(self.dist_matrix[self.coord_to_idx[p1], self.coord_to_idx[p2]])
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _line_of_sight(self, observer, target):
        r0, c0 = observer
        r1, c1 = target
        if r0 != r1 and c0 != c1: return False
        if r0 == r1:
            if abs(c1 - c0) > 5: return False
            step = 1 if c1 > c0 else -1
            for dc in range(1, abs(c1 - c0)):
                if self.world_map[r0, c0 + dc * step] == 1: return False
        else:
            if abs(r1 - r0) > 5: return False
            step = 1 if r1 > r0 else -1
            for dr in range(1, abs(r1 - r0)):
                if self.world_map[r0 + dr * step, c0] == 1: return False
        return True
