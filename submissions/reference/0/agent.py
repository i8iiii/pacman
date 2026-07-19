# - GROUP INFORMATION
#   + NO.: 0
#   + NAME: A-Star
#   + MEMBER: 1
#       - STUDENT ID: 19127616

import sys
from pathlib import Path
from collections import deque
import heapq

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


# Kiểm tra ô có đi được không (không phải tường, không ra ngoài map)
def is_valid(pos, map_state):
    r, c = pos
    h, w = map_state.shape
    if r < 0 or r >= h or c < 0 or c >= w:
        return False
    return map_state[r, c] == 0


# Lấy danh sách các ô kề có thể đi
def get_neighbors(pos, map_state):
    neighbors = []
    for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
        dr, dc = move.value
        npos = (pos[0] + dr, pos[1] + dc)
        if is_valid(npos, map_state):
            neighbors.append((npos, move))
    return neighbors


# Khoảng cách Manhattan giữa 2 ô
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# A* tìm đường ngắn nhất từ start đến goal, trả về danh sách các bước đi
def astar(map_state, start, goal):
    if start == goal:
        return []

    counter = 0
    heap = [(manhattan(start, goal), 0, counter, start, [])]
    visited = set()

    while heap:
        _, g, _, pos, path = heapq.heappop(heap)

        if pos in visited:
            continue
        visited.add(pos)

        if pos == goal:
            return path

        for npos, move in get_neighbors(pos, map_state):
            if npos not in visited:
                ng = g + 1
                heapq.heappush(
                    heap, (ng + manhattan(npos, goal), ng, counter, npos, path + [move])
                )
                counter += 1

    return []  # Không tìm được đường


# BFS từ start, trả về khoảng cách thực tế đến mọi ô trong map
def bfs_distances(map_state, start):
    dist = {start: 0}
    queue = deque([start])
    while queue:
        pos = queue.popleft()
        for npos, _ in get_neighbors(pos, map_state):
            if npos not in dist:
                dist[npos] = dist[pos] + 1
                queue.append(npos)
    return dist


# Pacman: vai trò Seeker, dùng A* để đuổi Ghost
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))

    def step(self, map_state, my_position, enemy_position, step_number):
        if enemy_position is None:
            return (Move.STAY, 1)

        my_pos = tuple(my_position)
        target = tuple(enemy_position)

        # Tìm đường ngắn nhất đến Ghost
        path = astar(map_state, my_pos, target)
        if not path:
            return (Move.STAY, 1)

        first_move = path[0]

        # Đi thẳng tối đa pacman_speed bước nếu path tiếp theo cùng hướng
        steps = 1
        for i in range(1, min(self.pacman_speed, len(path))):
            if path[i] == first_move:
                steps += 1
            else:
                break

        return (first_move, steps)


# Ghost: vai trò Hider, dùng BFS để chạy trốn xa Pacman nhất có thể
class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, map_state, my_position, enemy_position, step_number):
        if enemy_position is None:
            return Move.STAY

        my_pos = tuple(my_position)
        pac_pos = tuple(enemy_position)

        # Tính khoảng cách thực tế từ Pacman đến mọi ô
        dist_map = bfs_distances(map_state, pac_pos)

        best_move = Move.STAY
        best_score = -1

        # Chọn bước đi xa Pacman nhất, ưu tiên ô có nhiều ngả rẽ (tránh ngõ cụt)
        for npos, move in get_neighbors(my_pos, map_state) + [(my_pos, Move.STAY)]:
            dist = dist_map.get(npos, -1)
            if dist == -1:
                continue
            mobility = len(get_neighbors(npos, map_state))
            score = dist * 10 + mobility
            if score > best_score:
                best_score = score
                best_move = move

        return best_move
