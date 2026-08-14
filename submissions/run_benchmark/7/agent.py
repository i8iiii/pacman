import sys
from collections import deque
from pathlib import Path
import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move

MOVES = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
DELTAS = tuple(move.value for move in MOVES)
CAPTURE_DISTANCE = 2

class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Heuristic search with Belief + Greedy Chase"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.floor = None
        self.belief = None
        self.seen_count = None
        self.last_seen = None
        self.post_visits = {}
        self.turn = 0
        self.radius = 0
        self.scan_target = None
        self.opening_hubs = []
        self.opening_visits = 0
        self.opening_target = None
        self.opening_done = False

    def step(self, map_state, my_position, enemy_position, step_number):
        me = (int(my_position[0]), int(my_position[1]))
        self._initialize(map_state)
        self.turn += 1
        visible = self._visible_cells(me)
        
        for cell in visible:
            self.seen_count[cell] += 1
            self.last_seen[cell] = self.turn # stores the whole map with the current position

        # lock belief to enemy when in sight
        if enemy_position is not None:
            enemy = (int(enemy_position[0]), int(enemy_position[1]))
            self.belief = {enemy}
            self.last_known_ghost_pos = enemy
            self.scan_target = None
            self.opening_done = True
            return self._visible_chase(me, enemy)

        # if the ghost is not seen yet, pacman's belief that ghost could be in any visible steps of its sight
        if self.belief is None:
            self.belief = set(self._all_floor())
        else:
            self.belief = self._ghost_successors(self.belief)
            
        self.belief.difference_update(visible)
        
        if not self.belief:
            self.belief = set(self._all_floor()) - visible

        # Check if the opening state has finished and the number of encounters towards beginning spot < 2
        if not self.opening_done and self.opening_visits < 2:
            closest_cell = None
            min_distance = float('inf')
            # if the target is not initialized
            if self.opening_target is None:
                for cell in self.opening_hubs:
                    dist = self._grid_distance(me, cell)
                    if dist < min_distance:
                        min_distance = dist
                        closest_cell = cell
                self.opening_target = closest_cell
            target = self.opening_target
            

            if target is not None:
                if me == target:
                    self.opening_visits += 1
                    if self.opening_visits < 2:
                        self.opening_target = max(
                            (cell for cell in self.opening_hubs if cell != me),
                            key=lambda cell: self._grid_distance(me, cell),
                            default=None,
                        )
                    else:
                        self.opening_target = None
                    target = self.opening_target
                    
                if target is not None and target != me:
                    return self._action_toward(me, target)

        if self.scan_target is None or self.scan_target == me:
            if self.scan_target == me:
                self.post_visits[me] = self.post_visits.get(me, 0) + 1
            self.scan_target = self._best_scan_post(me)
            
        target = self.scan_target
        if target is not None:
            action = self._action_toward(me, target)
            if action is not None:
                return action
                
        return self._best_local_scan_move(me)

    def _initialize(self, map_state):
        if self.floor is None:
            self.floor = map_state != 1
            self.seen_count = np.zeros(map_state.shape, dtype=np.int32)
            self.last_seen = np.full(map_state.shape, -30, dtype=np.int32)
            self.opening_hubs = self._central_hubs()
        self._learn_radius(map_state)

    # Find possible junctions and choose one with highest distance
    def _central_hubs(self):
        junctions = [cell for cell in self._all_floor() if len(self._neighbors(cell)) >= 3]
        if not junctions:
            return []
            
        centrality = []
        for cell in junctions:
            distances = self._distance_map(cell)
            centrality.append((sum(distances.values()), cell))
            
        centrality.sort()
        first = centrality[0][1]
        second = None
        max_distance = -1
        
        central_candidates = [cell for _, cell in centrality[:8] if cell != first]
        for cell in central_candidates:
            dist = self._grid_distance(first, cell)
            if dist > max_distance:
                max_distance = dist
                second = cell
        return [first] + ([second] if second is not None else [])

    def _learn_radius(self, observation):
        for r in range(observation.shape[0]):
            for c in range(observation.shape[1]):
                if observation[r, c] != 0:
                    continue
                for dr, dc in DELTAS:
                    dist = 0
                    while True:
                        nr, nc = r + dr * (dist + 1), c + dc * (dist + 1)
                        if not self._inside((nr, nc)) or observation[nr, nc] != 0:
                            break
                        dist += 1
                    self.radius = max(self.radius, dist)

    def _visible_chase(self, pacman, ghost):
        best_action = (Move.STAY, 1)
        best_key = (-float('inf'), -float('inf'), -float('inf'), 0, 0, 0)
        ghost_moves = self._ghost_moves(ghost)
        
        for action in self._pacman_actions(pacman):
            p_next = self._apply_action(pacman, action)
            
            guaranteed_catch = 1
            worst_ghost_dist = -float('inf')
            los_retained_count = 0
            
            p_next_vision = self._visible_cells(p_next)
            
            for g_move in ghost_moves:
                g_next = self._apply_move(ghost, g_move)
                
                if not self._caught(p_next, g_next):
                    guaranteed_catch = 0
                    
                dist = self._grid_distance(p_next, g_next)
                worst_ghost_dist = max(worst_ghost_dist, dist)
                
                if g_next in p_next_vision:
                    los_retained_count += 1
            
            if guaranteed_catch:
                return action
                
            key = (
                guaranteed_catch,
                -worst_ghost_dist,
                los_retained_count,
                action[1],
                -action[0].value[0],
                -action[0].value[1]
            )
            
            if key > best_key:
                best_key = key
                best_action = action
                
        return best_action

    def _best_scan_post(self, start):
        best_cell, best_score = None, -float("inf")
        distances = self._distance_map(start)
        
        for cell in self._all_floor():
            visible = self._visible_cells(cell)
            # the number of visible turns since last seen cell that has not discovered yet
            max_staleness = max((self.turn - int(self.last_seen[p]) for p in visible), default=0)
            has_novel = 1 if any(self.seen_count[p] == 0 for p in visible) else 0 # not explored yet
            belief_score = 0

            if visible & self.belief:
                if getattr(self, 'last_known_ghost_pos', None):
                    ghost_dists = self._distance_map(self.last_known_ghost_pos)
                    dist_to_ghost = ghost_dists.get(cell, 999)
                    belief_score = max(0, 30 - dist_to_ghost)
                else:
                    belief_score = 15
                    
            exits = len(self._neighbors(cell))
            distance = distances.get(cell, 999)
            
            score = (
                (15 * belief_score) + (500 * has_novel) + (5 * max_staleness) + (10 * exits) - (4 * distance) - (80 * self.post_visits.get(cell, 0))
            )
            
            if cell == start:
                score -= 2
                
            if score > best_score:
                best_cell, best_score = cell, score
                
        return best_cell

    # fallback if best_scan_post can't return result/pacman meets deadend
    def _best_local_scan_move(self, pos):
        best_action, best_score = (Move.STAY, 1), -float("inf")
        future_belief = self._ghost_successors(self.belief)
        for action in self._pacman_actions(pos):
            end = self._apply_action(pos, action)
            visible = self._visible_cells(end)
            score = 80 * len(visible & future_belief) - sum(self.seen_count[p] for p in visible)
            if score > best_score:
                best_action, best_score = action, score
        return best_action

    def _action_toward(self, start, target):
        path = self._path(start, target)
        if len(path) < 2:
            return self._best_local_scan_move(start)
            
        move = self._move_between(path[0], path[1])
        steps = 1
        
        while steps < self.pacman_speed and steps + 1 < len(path):
            if self._move_between(path[steps], path[steps + 1]) != move:
                break
            steps += 1
            
        return (move, steps)

    # coordinate list for walkable path
    def _all_floor(self):
        return [(int(r), int(c)) for r, c in np.argwhere(self.floor)]

    # get visible cells based on current position
    def _visible_cells(self, pos):
        visible = {pos}
        for dr, dc in DELTAS:
            for distance in range(1, self.radius + 1):
                cell = (pos[0] + dr * distance, pos[1] + dc * distance)
                if not self._inside(cell) or not self.floor[cell]:
                    break
                visible.add(cell)
        return visible

    # predict potential ghost's position in the next step based on current position and visible local cells
    def _ghost_successors(self, positions):
        result = set()
        for pos in positions:
            result.add(pos)
            result.update(self._neighbors(pos))
        return result

    def _pacman_actions(self, pos):
        actions = [(Move.STAY, 1)]
        for move in MOVES:
            current = pos
            for step in range(1, self.pacman_speed + 1):
                nxt = self._apply_move(current, move)
                if nxt == current:
                    break
                current = nxt
                actions.append((move, step))
        return actions

    def _ghost_moves(self, pos):
        return [Move.STAY] + [move for move in MOVES if self._apply_move(pos, move) != pos]

    def _apply_action(self, pos, action):
        move, steps = action
        current = pos
        for _ in range(steps):
            nxt = self._apply_move(current, move)
            if nxt == current:
                break
            current = nxt
        return current

    def _apply_move(self, pos, move):
        cell = (pos[0] + move.value[0], pos[1] + move.value[1])
        return cell if self._inside(cell) and self.floor[cell] else pos

    def _path(self, start, target):
        queue, parent = deque([start]), {start: None}
        while queue:
            current = queue.popleft()
            if current == target:
                path = [current]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            for nxt in self._neighbors(current):
                if nxt not in parent:
                    parent[nxt] = current
                    queue.append(nxt)
        return []

    def _grid_distance(self, start, target):
        path = self._path(start, target)
        return len(path) - 1 if path else 999

    def _distance_map(self, start):
        queue, distance = deque([start]), {start: 0}
        while queue:
            current = queue.popleft()
            for nxt in self._neighbors(current):
                if nxt not in distance:
                    distance[nxt] = distance[current] + 1
                    queue.append(nxt)
        return distance

    def _neighbors(self, pos):
        return [self._apply_move(pos, move) for move in MOVES if self._apply_move(pos, move) != pos]

    def _move_between(self, a, b):
        delta = (b[0] - a[0], b[1] - a[1])
        return next((move for move in MOVES if move.value == delta), Move.STAY)

    def _caught(self, pacman, ghost):
        return abs(pacman[0] - ghost[0]) + abs(pacman[1] - ghost[1]) < CAPTURE_DISTANCE

    def _inside(self, pos):
        return 0 <= pos[0] < self.floor.shape[0] and 0 <= pos[1] < self.floor.shape[1]


class GhostAgent(BaseGhostAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Greedy Distance Maximization + Patrol (Least vistied junction)"
        self.floor = None
        self.visits = None
        self.patrol_target = None
        self.patrol_path = []
        self.previous_position = None

    def step(self, map_state, my_position, enemy_position, step_number):
        me = (int(my_position[0]), int(my_position[1]))
        if self.floor is None:
            self.floor = map_state != 1
            self.visits = np.zeros(map_state.shape, dtype=np.int32)
            
        self.visits[me] += 1
        
        if enemy_position is not None:
            threat = (int(enemy_position[0]), int(enemy_position[1]))
            self.patrol_path = []
            move = self._evade(me, threat)
        else:
            move = self._patrol(me)
            
        self.previous_position = me
        return move

    def _evade(self, me, threat):
        best_move, best_score = Move.STAY, -float("inf")
        for move in [Move.STAY] + list(MOVES):
            nxt = self._apply_move(me, move)
            if nxt == me and move != Move.STAY:
                continue
                
            distance = self._distance(threat, nxt)
            exits = len(self._neighbors(nxt))
            score = 20 * distance + 5 * exits - self.visits[nxt]
            
            if self.previous_position is not None and nxt == self.previous_position:
                score -= 80
                
            if score > best_score:
                best_move, best_score = move, score
                
        return best_move

    def _patrol(self, me):
        while self.patrol_path and self.patrol_path[0] == me:
            self.patrol_path.pop(0)
            
        if self.patrol_path:
            return self._move_between(me, self.patrol_path[0])

        posts = [cell for cell in self._all_floor() if len(self._neighbors(cell)) >= 3]
        if not posts:
            posts = self._all_floor()
            
        candidates = [cell for cell in posts if cell != me]
        if not candidates:
            return Move.STAY
            
        target = min(
            candidates,
            key=lambda cell: (self.visits[cell], -min(self._distance(me, cell), 8)),
        )
        path = self._path(me, target)
        
        if len(path) >= 2:
            self.patrol_target = target
            self.patrol_path = path[1:]
            return self._move_between(me, self.patrol_path[0])
            
        options = [move for move in MOVES if self._apply_move(me, move) != me]
        
        if self.previous_position is not None:
            non_reverse = [move for move in options if self._apply_move(me, move) != self.previous_position]
            if non_reverse:
                options = non_reverse
                
        return options[0] if options else Move.STAY

    def _all_floor(self):
        return [(int(r), int(c)) for r, c in np.argwhere(self.floor)]

    def _apply_move(self, pos, move):
        nxt = (pos[0] + move.value[0], pos[1] + move.value[1])
        return nxt if self._inside(nxt) and self.floor[nxt] else pos

    def _neighbors(self, pos):
        return [self._apply_move(pos, move) for move in MOVES if self._apply_move(pos, move) != pos]

    def _path(self, start, target):
        queue, parent = deque([start]), {start: None}
        while queue:
            current = queue.popleft()
            if current == target:
                path = [current]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            for nxt in self._neighbors(current):
                if nxt not in parent:
                    parent[nxt] = current
                    queue.append(nxt)
        return []

    def _distance(self, start, target):
        path = self._path(start, target)
        return len(path) - 1 if path else 999

    def _move_between(self, a, b):
        delta = (b[0] - a[0], b[1] - a[1])
        return next((move for move in MOVES if move.value == delta), Move.STAY)

    def _inside(self, pos):
        return 0 <= pos[0] < self.floor.shape[0] and 0 <= pos[1] < self.floor.shape[1]