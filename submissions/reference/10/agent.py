import sys
import random
# from abc import ABC, abstractmethod
from pathlib import Path
from collections import deque


src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))



from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np

# nhom 10
# Phong 24127102
# Vinh 24127591

ALL_MOVES = [Move.UP, Move.DOWN, Move.LEFT,  Move.RIGHT]
HISTORY_LEN = 6


# def is_open(map_state, r, c):
#     rows = len(map_state)
#     cols = len(map_state[0])
#     if r < 0 or r >= rows or c < 0 or c >= cols:
#         return False
#     return map_state[r][c] == 0
 
# def neighbors(map_state, r, c):
#     for move in DIRECTIONS:
#         dr, dc = move.value
#         nr, nc = r + dr, c + dc
#         if is_open(map_state, nr, nc):
#             yield move, (nr, nc


def is_valid(pos, map_state):
    row, col = pos
    height, width = map_state.shape
    if row < 0 or row >= height or col < 0 or col >= width:
        return False
    return map_state[row, col] == 0

def apply_move(pos, move):
    dr, dc = move.value
    return (pos[0] + dr, pos[1] + dc)

def manhattan_distance(pos1, pos2):
    return abs(pos1[0]  - pos2[0]) + abs(pos1[1] - pos2[1])


def bfs_first_move(start, goal, map_state):
    if start == goal:
        return Move.STAY

    visited = {start}
    queue = deque()
    queue.append((start, None))

    while queue:
        pos, first_move = queue.popleft()

        for move  in ALL_MOVES:
            next_pos = apply_move(pos, move)

            if next_pos == goal:
                return first_move if first_move is not None else move

            if next_pos in visited or not is_valid(next_pos, map_state):
                continue

            visited.add(next_pos)
            new_first_move = move if first_move is None else first_move
            queue.append((next_pos, new_first_move))

    return Move.STAY


def get_valid_moves(pos, map_state):
    moves = []
    for move in ALL_MOVES:
        if is_valid(apply_move(pos,  move), map_state):
            moves.append(move)
    return moves


def max_steps_in_direction(pos, move, map_state, max_steps):
    steps = 0
    current = pos
    for _ in range(max_steps):
        next_pos = apply_move(current, move)
        if not is_valid(next_pos, map_state):
            break
        steps += 1
        current = next_pos
    return steps





# def bfs(map_state, start, goal):

#     if start == goal:
#         return 0, Move.STAY
 
#     visited = {start}
#     queue = deque([(start, None, 0)])   
 

#     while queue:
#         (r, c), first_move, dist = queue.popleft()
#         for move, nxt in neighbors(map_state, r, c):
#             if nxt in visited:
#                 continue
            
#             fm = move if first_move is None else first_move
#             if nxt == goal:
#                 return dist + 1, fm
#             visited.add(nxt)
#             queue.append((nxt, fm, dist + 1))
 
#     return float("inf"), Move.STAY 
 
 






class PacmanAgent(BasePacmanAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.last_known_enemy_pos = None
        self.position_history = deque(maxlen=HISTORY_LEN)

    def step(self, map_state, my_position, enemy_position, step_number):
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        target = enemy_position if enemy_position is not None else self.last_known_enemy_pos
  

        if target is None:
            for move in ALL_MOVES:
                if is_valid(apply_move(my_position, move), map_state):
                    return (move, 1)
            return (Move.STAY, 1)

        if my_position in self.position_history:
            valid_moves = get_valid_moves(my_position, map_state)
            if valid_moves:
                move = random.choice(valid_moves)
                self.position_history.clear()
                return (move, 1)

        self.position_history.append(my_position)

        move = bfs_first_move(my_position, target, map_state)

        if move == Move.STAY:
            return (Move.STAY, 1)

        steps = max_steps_in_direction(my_position, move, map_state, self.pacman_speed)
        steps = max(1, steps)

        return (move, steps)


class GhostAgent(BaseGhostAgent):

    DEPTH = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_known_enemy_pos = None
        self.position_history = deque(maxlen=HISTORY_LEN)
        

  


    def step(self, map_state, my_position, enemy_position, step_number):
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        threat = enemy_position if enemy_position is not None else self.last_known_enemy_pos

        if threat is None:
            for move in ALL_MOVES:
                if is_valid(apply_move(my_position, move), map_state):
                    return move
            return Move.STAY

        ghost_moves = get_valid_moves(my_position, map_state)
        if not ghost_moves:
            return Move.STAY

        if my_position in self.position_history:
            self.position_history.clear()
            return random.choice(ghost_moves)

        self.position_history.append(my_position)

        best_move = Move.STAY
        best_value = float("-inf")

        for move in ghost_moves:
            ghost_next = apply_move(my_position, move)
            value = self._minimax(ghost_next, threat, self.DEPTH, map_state)
            if value > best_value:
                best_value = value
                best_move = move

        return best_move

    def _minimax(self, ghost_pos, pacman_pos, depth, map_state):
        if depth == 0 or ghost_pos == pacman_pos:
            return manhattan_distance(ghost_pos, pacman_pos)

        pacman_moves = get_valid_moves(pacman_pos, map_state)
        if not pacman_moves:
            return manhattan_distance(ghost_pos, pacman_pos)


        worst_case = float("inf")
        for p_move in pacman_moves:

            pacman_next = apply_move(pacman_pos, p_move)
            value = manhattan_distance(ghost_pos, pacman_next)
            if value < worst_case:
                worst_case = value

        return   worst_case