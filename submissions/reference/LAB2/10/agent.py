import sys
import time
import math
import random
from pathlib import Path
from collections import deque
from heapq import heappush, heappop


src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


ALL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
MAP_SIZE = 21
HISTORY_LEN = 6
TIME_LIMIT = 0.9


def apply_move(pos, move):
    dr, dc = move.value
    return (pos[0] + dr, pos[1] + dc)





def is_walkable(pos, known_map):
    row, col = pos
    if row < 0 or row >= MAP_SIZE or col < 0 or col >= MAP_SIZE:
        return False
    return known_map[row, col] == 0




def is_in_bounds(pos):
    r, c = pos
    return 0 <= r < MAP_SIZE and 0 <= c < MAP_SIZE




def get_valid_moves(pos, known_map):
    return [m for m in ALL_MOVES if is_walkable(apply_move(pos, m), known_map)]


def get_valid_nexts(pos, known_map):
    result = []
    for move in ALL_MOVES:
        nxt = apply_move(pos, move)
        if is_walkable(nxt, known_map):
            result.append((nxt, move))
    return result


def manhattan(a, b):
    return abs(a[0] -  b[0]) + abs(a[1] - b[1])


def bfs_distance(start, goal, known_map):
    if start == goal:
        return 0
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        pos, dist = queue.popleft()
        for move in ALL_MOVES:
            nxt = apply_move(pos, move)
            if nxt == goal:
                return dist + 1
            if nxt not in visited and is_walkable(nxt, known_map):
                visited.add(nxt)
                queue.append((nxt, dist + 1))
    return 9999


def astar(start, goal, known_map):
    if start == goal:
        return []

    counter = 0
    frontier = [(manhattan(start, goal), 0, start, [])]
    visited = {}

    while frontier:
        f, _, pos, path = heappop(frontier)
        g = len(path)

        if pos in visited and visited[pos] <= g:
            continue
        visited[pos] = g

        for nxt, move in get_valid_nexts(pos, known_map):
            new_path = path + [move]
            if nxt == goal:
                return new_path
            new_g = len(new_path)
            if nxt not in visited or visited[nxt] > new_g:
                counter +=  1
                heappush(frontier, (
                    new_g + manhattan(nxt, goal),
                    counter, nxt, new_path
                ))
    return []


def bfs_to_unknown(start, known_map):
    visited = {start}
    queue = deque([(start, None)])
    while queue:
        pos, first_move = queue.popleft()
        for move in ALL_MOVES:
            nxt = apply_move(pos, move)
            if not is_in_bounds(nxt):
                continue
            val = known_map[nxt[0], nxt[1]]
            if val == -1:
                return first_move if first_move is not None else move
            if val == 0 and nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, move if first_move is None else first_move))
    return Move.STAY


def max_steps_straight(pos, move, known_map,  max_steps):
    steps, current = 0, pos
    for _ in range(max_steps):
        nxt = apply_move(current, move)
        if not is_walkable(nxt, known_map):
            break
        steps += 1
        current = nxt
    return steps


def update_known_map(known_map, map_state):
    mask = known_map == -1
    known_map[mask] = map_state[mask]


def count_mobility(pos, known_map):
    return len(get_valid_moves(pos, known_map))


def check_line_of_sight(pos1, pos2, known_map):
    r1, c1 = pos1
    r2, c2 = pos2

    if r1 != r2 and c1 != c2:
        return False

    if r1 == r2:
        for c in range(min(c1, c2) + 1, max(c1, c2)):
            if known_map[r1, c] == 1:
                return False
        return True

    for r in range(min(r1, r2) +  1, max(r1, r2)):
        if known_map[r, c1] == 1:
            return False
    return True


def get_pacman_reachable(pacman_pos, speed, known_map):
    reachable = set()
    queue = deque([(pacman_pos, 0)])
    while queue:
        curr, steps = queue.popleft()
        if steps == speed:
            reachable.add(curr)
            continue
        nexts = get_valid_nexts(curr, known_map)
        if not nexts:
            reachable.add(curr)
        for nxt, _ in nexts:
            queue.append((nxt, steps + 1))
    return reachable


def evaluate_ghost(ghost_pos, pacman_pos, known_map):
    dist = bfs_distance(ghost_pos, pacman_pos, known_map)
    if dist == 0:
        return -10000

    score = dist * 10

    mobility = count_mobility(ghost_pos, known_map)
    score += mobility * 10
    if mobility <= 1:
        score -= 300

    if not check_line_of_sight(ghost_pos,  pacman_pos, known_map):
        score += 500
    else:
        score -= 200

    return score


class PacmanAgent(BasePacmanAgent):
    PATROL_POINTS = [
        (9, 15),
        (5,  15),
        (2, 15),
        (2, 10),
        (2,  5),
        (6, 7),
        (5,  5),
        (9, 5),
        (13, 5),
        (19, 5),
        (19,10),
        (19, 15),
        (13, 15),
        (9, 10),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.known_map = np.full((MAP_SIZE, MAP_SIZE), -1, dtype=int)
        self.last_known_enemy = None
        self.position_history = deque(maxlen=HISTORY_LEN)

        self.patrol_index = 0
        self.pursuit_steps_left =  0
        self.PURSUIT_PERSISTENCE = 8

    def step(self, map_state, my_position, enemy_position, step_number):
        update_known_map(self.known_map, map_state)

        if enemy_position is not None:
            self.last_known_enemy = enemy_position
            self.pursuit_steps_left = self.PURSUIT_PERSISTENCE
        elif self.pursuit_steps_left > 0:
            self.pursuit_steps_left -= 1

        if my_position in self.position_history:
            valid = get_valid_moves(my_position, self.known_map)
            if valid:
                self.position_history.clear()
                return (random.choice(valid), 1)

        self.position_history.append(my_position)

        if enemy_position is not None:
            path = astar(my_position, enemy_position, self.known_map)
            if path:
                move = path[0]
                if (self.pacman_speed >= 2
                        and len(path) >= 2
                        and path[1] == move):
                    steps = max_steps_straight(
                        my_position, move, self.known_map, self.pacman_speed
                    )
                else:
                    steps = 1
                return (move, max(1, steps))
            return self._greedy_toward(my_position, enemy_position)

        if self.pursuit_steps_left > 0 and self.last_known_enemy is not None:
            path = astar(my_position, self.last_known_enemy, self.known_map)
            if path:
                move = path[0]
                steps = max_steps_straight(
                    my_position, move, self.known_map,  self.pacman_speed
                )
                return (move, max(1, steps))

        patrol_target = self.PATROL_POINTS[self.patrol_index]

        if manhattan(my_position, patrol_target) <= 1:
            self.patrol_index = (self.patrol_index + 1) % len(self.PATROL_POINTS)
            patrol_target = self.PATROL_POINTS[self.patrol_index]

        path = astar(my_position, patrol_target, self.known_map)
        if path:
            move = path[0]
            steps = max_steps_straight(
                my_position, move, self.known_map, self.pacman_speed
            )
            return (move, max(1, steps))

        move = bfs_to_unknown(my_position, self.known_map)
        if move != Move.STAY:
            steps = max_steps_straight(
                my_position, move, self.known_map, self.pacman_speed
            )
            return (move, max(1, steps))

        valid = get_valid_moves(my_position, self.known_map)
        return (random.choice(valid), 1) if valid else (Move.STAY, 1)

    def _greedy_toward(self, my_pos, target):
        best_move = Move.STAY
        best_dist = manhattan(my_pos, target)
        for move in ALL_MOVES:
            nxt = apply_move(my_pos, move)
            if is_walkable(nxt, self.known_map):
                d = manhattan(nxt, target)
                if d < best_dist:
                    best_dist = d
                    best_move = move
        return (best_move, 1)


class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.known_map = np.full((MAP_SIZE, MAP_SIZE),  -1, dtype=int)
        self.last_known_enemy = None
        self.steps_since_seen = 0
        self.survival_target = None
        self.position_history = deque(maxlen=HISTORY_LEN)
        self._t0 = 0

    def step(self, map_state, my_position, enemy_position, step_number):
        self._t0 = time.time()
        update_known_map(self.known_map, map_state)

        if enemy_position is not None:
            self.last_known_enemy = enemy_position
            self.steps_since_seen = 0
            self.survival_target = None
        else:
            self.steps_since_seen += 1

        valid_moves = get_valid_moves(my_position, self.known_map)
        if not valid_moves:
            return Move.STAY

        if my_position in self.position_history:
            self.position_history.clear()
            return random.choice(valid_moves)

        self.position_history.append( my_position)

        target = enemy_position or self.last_known_enemy

        if target is None or self.steps_since_seen > 5:
            if (not self.survival_target
                    or my_position == self.survival_target
                    or not is_walkable(self.survival_target, self.known_map)):
                self.survival_target = self._nearest_intersection(my_position)
            if self.survival_target:
                path = self._bfs_path(my_position, self.survival_target)
                if path:
                    return path[0]
            return random.choice(valid_moves)

        best_move = valid_moves[0]
        for depth in range(1, 20):
            if time.time() - self._t0 > TIME_LIMIT:
                break
            try:
                _, move = self._minimax(
                    my_position, target,
                    depth=depth,
                    is_ghost_turn=True
                )
                if time.time() - self._t0 < TIME_LIMIT and move is not None:
                    best_move = move
                else:
                    break
            except _TimeUp:
                break

        return best_move

    def _minimax(self, ghost_pos, pacman_pos, depth, is_ghost_turn):
        if time.time() - self._t0 > TIME_LIMIT:
            raise _TimeUp()

        if depth == 0 or ghost_pos == pacman_pos:
            return evaluate_ghost(ghost_pos, pacman_pos, self.known_map), Move.STAY

        if is_ghost_turn:
            nexts = get_valid_nexts(ghost_pos, self.known_map)
            if not nexts:
                return evaluate_ghost(ghost_pos, pacman_pos, self.known_map), Move.STAY

            best_val = -math.inf
            best_move = nexts[0][1]
            for nxt_pos, move in nexts:
                val, _ = self._minimax(
                    nxt_pos, pacman_pos, depth - 1, is_ghost_turn=False
                )
                if val > best_val:
                    best_val = val
                    best_move = move
            return best_val, best_move

        else:
            pac_reach = get_pacman_reachable(pacman_pos, 2, self.known_map)
            if ghost_pos in pac_reach:
                return -10000, None

            best_val = math.inf
            for nxt_pac in pac_reach:
                val, _ = self._minimax(
                    ghost_pos, nxt_pac, depth - 1, is_ghost_turn=True
                )
                if val < best_val:
                    best_val = val
            return best_val, None

    def _nearest_intersection(self, start):
        queue = deque([start])
        visited = {start}
        while queue:
            curr = queue.popleft()
            nexts = get_valid_nexts(curr, self.known_map)
            if len(nexts) >= 3 and curr != start:
                return curr
            for nxt, _ in nexts:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return start

    def _bfs_path(self, start, goal):
        if start == goal:
            return []
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            curr, path = queue.popleft()
            for nxt, move in get_valid_nexts(curr, self.known_map):
                if nxt == goal:
                    return path + [move]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [move]))
        return []


class _TimeUp(Exception):
    pass



















































# class PacmanAgent(BasePacmanAgent):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
#         self.known_map = np.full((MAP_SIZE, MAP_SIZE), -1, dtype=int)
#         self.last_known_enemy = None
#         self.position_history = deque(maxlen=HISTORY_LEN)

#     def step(self, map_state, my_position, enemy_position, step_number):
#         update_known_map(self.known_map, map_state)

#         if enemy_position is not None:
#             self.last_known_enemy = enemy_position

#         if my_position in self.position_history:
#             valid = get_valid_moves(my_position, self.known_map)
#             if valid:
#                 self.position_history.clear()
#                 return (random.choice(valid), 1)

#         self.position_history.append(my_position)

#         if enemy_position is not None:
#             path = astar(my_position, enemy_position, self.known_map)
#             if path:
#                 move = path[0]
#                 if (self.pacman_speed >= 2
#                         and len(path) >= 2
#                         and path[1] == move):
#                     steps = max_steps_straight(
#                         my_position, move, self.known_map, self.pacman_speed
#                     )
#                 else:
#                     steps = 1
#                 return (move, max(1, steps))
#             return self._greedy_toward(my_position, enemy_position)

#         if self.last_known_enemy is not None:
#             path = astar(my_position, self.last_known_enemy, self.known_map)
#             if path:
#                 move = path[0]
#                 steps = max_steps_straight(
#                     my_position, move, self.known_map, self.pacman_speed
#                 )
#                 return (move, max(1, steps))

#         move = bfs_to_unknown(my_position, self.known_map)
#         if move != Move.STAY:
#             steps = max_steps_straight(
#                 my_position, move, self.known_map, self.pacman_speed
#             )
#             return (move, max(1, steps))

#         valid = get_valid_moves(my_position, self.known_map)
#         return (random.choice(valid), 1) if valid else (Move.STAY, 1)

#     def _greedy_toward(self, my_pos, target):
#         best_move = Move.STAY
#         best_dist = manhattan(my_pos, target)
#         for move in ALL_MOVES:
#             nxt = apply_move(my_pos, move)
#             if is_walkable(nxt, self.known_map):
#                 d = manhattan(nxt, target)
#                 if d < best_dist:
#                     best_dist = d
#                     best_move = move
#         return (best_move, 1)


# class GhostAgent(BaseGhostAgent):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.known_map = np.full((MAP_SIZE, MAP_SIZE), -1, dtype=int)
#         self.last_known_enemy = None
#         self.steps_since_seen = 0
#         self.survival_target = None
#         self.position_history = deque(maxlen=HISTORY_LEN)
#         self._t0 = 0

#     def step(self, map_state, my_position, enemy_position, step_number):
#         self._t0 = time.time()
#         update_known_map(self.known_map, map_state)

#         if enemy_position is not None:
#             self.last_known_enemy = enemy_position
#             self.steps_since_seen = 0
#             self.survival_target = None
#         else:
#             self.steps_since_seen += 1

#         valid_moves = get_valid_moves(my_position, self.known_map)
#         if not valid_moves:
#             return Move.STAY

#         if my_position in self.position_history:
#             self.position_history.clear()
#             return random.choice(valid_moves)

#         self.position_history.append(my_position)

#         target = enemy_position or self.last_known_enemy

#         if target is None or self.steps_since_seen > 5:
#             if (not self.survival_target
#                     or my_position == self.survival_target
#                     or not is_walkable(self.survival_target, self.known_map)):
#                 self.survival_target = self._nearest_intersection(my_position)
#             if self.survival_target:
#                 path = self._bfs_path(my_position, self.survival_target)
#                 if path:
#                     return path[0]
#             return random.choice(valid_moves)

#         best_move = valid_moves[0]
#         for depth in range(1, 20):
#             if time.time() - self._t0 > TIME_LIMIT:
#                 break
#             try:
#                 _, move = self._minimax(
#                     my_position, target,
#                     depth=depth,
#                     is_ghost_turn=True
#                 )
#                 if time.time() - self._t0 < TIME_LIMIT and move is not None:
#                     best_move = move
#                 else:
#                     break
#             except _TimeUp:
#                 break

#         return best_move

#     def _minimax(self, ghost_pos, pacman_pos, depth, is_ghost_turn):
#         if time.time() - self._t0 > TIME_LIMIT:
#             raise _TimeUp()

#         if depth == 0 or ghost_pos == pacman_pos:
#             return evaluate_ghost(ghost_pos, pacman_pos, self.known_map), Move.STAY

#         if is_ghost_turn:
#             nexts = get_valid_nexts(ghost_pos, self.known_map)
#             if not nexts:
#                 return evaluate_ghost(ghost_pos, pacman_pos, self.known_map), Move.STAY

#             best_val = -math.inf
#             best_move = nexts[0][1]
#             for nxt_pos, move in nexts:
#                 val, _ = self._minimax(
#                     nxt_pos, pacman_pos, depth - 1, is_ghost_turn=False
#                 )
#                 if val > best_val:
#                     best_val = val
#                     best_move = move
#             return best_val, best_move

#         else:
#             pac_reach = get_pacman_reachable(pacman_pos, 2, self.known_map)
#             if ghost_pos in pac_reach:
#                 return -10000, None

#             best_val = math.inf
#             for nxt_pac in pac_reach:
#                 val, _ = self._minimax(
#                     ghost_pos, nxt_pac, depth - 1, is_ghost_turn=True
#                 )
#                 if val < best_val:
#                     best_val = val
#             return best_val, None

#     def _nearest_intersection(self, start):
#         queue = deque([start])
#         visited = {start}
#         while queue:
#             curr = queue.popleft()
#             nexts = get_valid_nexts(curr, self.known_map)
#             if len(nexts) >= 3 and curr != start:
#                 return curr
#             for nxt, _ in nexts:
#                 if nxt not in visited:
#                     visited.add(nxt)
#                     queue.append(nxt)
#         return start

#     def _bfs_path(self, start, goal):
#         if start == goal:
#             return []
#         queue = deque([(start, [])])
#         visited = {start}
#         while queue:
#             curr, path = queue.popleft()
#             for nxt, move in get_valid_nexts(curr, self.known_map):
#                 if nxt == goal:
#                     return path + [move]
#                 if nxt not in visited:
#                     visited.add(nxt)
#                     queue.append((nxt, path + [move]))
#         return []




















































#LAB )01

# import sys
# import random
# from pathlib import Path
# from collections import deque

# src_path = Path(__file__).parent.parent.parent / "src"
# sys.path.insert(0, str(src_path))

# from agent_interface import PacmanAgent as BasePacmanAgent
# from agent_interface import GhostAgent as BaseGhostAgent
# from environment import Move
# import numpy as np

# ALL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
# HISTORY_LEN = 6
# MAP_SIZE = 21

# def apply_move(pos, move):
#     dr, dc = move.value
#     return (pos[0] + dr, pos[1] + dc)

# def is_walkable(pos, map_state):
#     row, col = pos
#     if row < 0 or row >= MAP_SIZE or col < 0 or col >= MAP_SIZE:
#         return False
#     return map_state[row, col] == 0

# def get_valid_moves(pos, map_state):
#     return [m for m in ALL_MOVES if is_walkable(apply_move(pos, m), map_state)]

# def bfs_distance(start, goal, map_state):
#     if start == goal:
#         return 0
#     visited = {start}
#     queue = deque([(start, 0)])
#     while queue:
#         pos, dist = queue.popleft()
#         for move in ALL_MOVES:
#             nxt = apply_move(pos, move)
#             if nxt == goal:
#                 return dist + 1
#             if nxt not in visited and is_walkable(nxt, map_state):
#                 visited.add(nxt)
#                 queue.append((nxt, dist + 1))
#     return 9999

# def bfs_first_move(start, goal, map_state):
#     if start == goal:
#         return Move.STAY
#     visited = {start}
#     queue = deque([(start, None)])
#     while queue:
#         pos, first_move =  queue.popleft()
#         for move in ALL_MOVES:
#             nxt = apply_move(pos, move)
#             if nxt == goal:
#                 return first_move if first_move is not None else move
#             if nxt not in visited and is_walkable(nxt, map_state):
#                 visited.add(nxt)
#                 queue.append((nxt, move if first_move is None else first_move))
#     return Move.STAY

# def count_mobility(pos, map_state):
#     return len(get_valid_moves(pos, map_state))

# def evaluate(ghost_pos, pacman_pos,  map_state):
#     dist = bfs_distance(ghost_pos, pacman_pos, map_state)
#     mobility = count_mobility(ghost_pos, map_state)
#     return dist + MOBILITY_WEIGHT * mobility

# MOBILITY_WEIGHT = 0.5

# def max_steps_straight(pos, move, map_state, max_steps):
#     steps, current = 0, pos
#     for _ in range(max_steps):
#         nxt = apply_move(current, move)
#         if not is_walkable(nxt, map_state):
#             break
#         steps += 1
#         current = nxt
#     return steps

# class PacmanAgent(BasePacmanAgent):

#     def __init__(self,  **kwargs):
#         super().__init__(**kwargs)
#         self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
#         self.position_history = deque(maxlen=HISTORY_LEN)

#     def step(self, map_state, my_position, enemy_position, step_number):
#         if my_position in self.position_history:
#             valid = get_valid_moves(my_position, map_state)
#             if valid:
#                 self.position_history.clear()
#                 return (random.choice(valid), 1)

#         self.position_history.append(my_position)

#         move = bfs_first_move(my_position,  enemy_position, map_state)

#         if move == Move.STAY:
#             return (Move.STAY, 1)

#         steps = max_steps_straight(my_position, move, map_state, self.pacman_speed)
#         return (move, max(1, steps))

# class GhostAgent(BaseGhostAgent):



#     DEPTH = 10

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.position_history = deque(maxlen=HISTORY_LEN)
#         self._map_state = None

#     def step(self, map_state, my_position, enemy_position, step_number):
#         valid_moves = get_valid_moves(my_position, map_state)
#         if not valid_moves:
#             return Move.STAY

#         if my_position in self.position_history:
#             self.position_history.clear()
#             return random.choice(valid_moves)

#         self.position_history.append(my_position)

#         self._map_state = map_state

#         best_move = valid_moves[0]
#         best_value = float("-inf")
#         alpha = float("-inf")

#         for move in valid_moves:
#             ghost_next = apply_move(my_position, move)
#             value = self._alphabeta(
#                 ghost_next, enemy_position,
#                 depth=self.DEPTH - 1,
#                 alpha=alpha,
#                 beta=float("inf"),
#                 is_ghost_turn=False
#             )
#             if value > best_value:
#                 best_value = value
#                 best_move = move
#             alpha = max(alpha, best_value)

#         return best_move

#     def _alphabeta(self, ghost_pos, pacman_pos, depth,
#                    alpha, beta, is_ghost_turn):
#         if depth == 0 or ghost_pos == pacman_pos:
#             return evaluate(ghost_pos, pacman_pos, self._map_state)

#         if is_ghost_turn:
#             ghost_moves = get_valid_moves(ghost_pos, self._map_state)
#             if not ghost_moves:
#                 return evaluate(ghost_pos, pacman_pos, self._map_state)

#             best = float("-inf")
#             for move in ghost_moves:
#                 val = self._alphabeta(
#                     apply_move(ghost_pos, move), pacman_pos,
#                     depth - 1, alpha, beta,
#                     is_ghost_turn=False
#                 )
#                 if val > best:
#                     best = val
#                 if best > alpha:
#                     alpha = best
#                 if best >= beta:
#                     break
#             return best

#         else:
#             pacman_moves = get_valid_moves(pacman_pos, self._map_state)
#             if not pacman_moves:
#                 return evaluate(ghost_pos, pacman_pos, self._map_state)

#             best = float("inf")
#             for move in pacman_moves:
#                 val = self._alphabeta(
#                     ghost_pos, apply_move(pacman_pos, move),
#                     depth - 1, alpha, beta,
#                     is_ghost_turn=True
#                 )
#                 if val < best:
#                     best = val
#                 if best < beta:
#                     beta = best
#                 if best <= alpha:
#                     break
#             return  best