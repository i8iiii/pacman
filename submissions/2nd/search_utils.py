from collections import deque
import numpy as np

from environment import Move

# 4 hướng đi cơ bản (không tính STAY). Thứ tự cố định để kết quả lặp lại được.
ORTHO_MOVES = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)


def manhattan(a: tuple, b: tuple) -> int:
    """Khoảng cách Manhattan giữa hai ô (row, col)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class MazeGraph:
    def __init__(self, map_state: np.ndarray):
        self.map = map_state
        self.h, self.w = map_state.shape
        # source (row, col) -> mảng khoảng cách (h, w), giá trị -1 = không tới được.
        self._dist_cache = {}

    #truy vấn ô 
    def passable(self, row, col) -> bool:
        if row < 0 or row >= self.h or col < 0 or col >= self.w:
            return False
        return self.map[row, col] == 0

    def neighbors(self, pos):
        r, c = pos
        out = []
        for m in ORTHO_MOVES:
            dr, dc = m.value
            nr, nc = r + dr, c + dc
            if self.passable(nr, nc):
                out.append(((nr, nc), m))
        return out

    def open_degree(self, pos) -> int:
        return len(self.neighbors(pos))

    #BFS khoảng cách thực
    def dist_map(self, source):
        cached = self._dist_cache.get(source)
        if cached is not None:
            return cached

        dist = np.full((self.h, self.w), -1, dtype=np.int16)
        sr, sc = source
        if not self.passable(sr, sc):
            # nguồn nằm trên tường (không nên xảy ra) -> trả về mảng rỗng.
            self._dist_cache[source] = dist
            return dist

        dist[sr, sc] = 0
        q = deque([source])
        while q:
            r, c = q.popleft()
            d0 = dist[r, c]
            for m in ORTHO_MOVES:
                dr, dc = m.value
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.h and 0 <= nc < self.w \
                        and self.map[nr, nc] == 0 and dist[nr, nc] == -1:
                    dist[nr, nc] = d0 + 1
                    q.append((nr, nc))

        self._dist_cache[source] = dist
        return dist

    def dist(self, a, b) -> int:
        return int(self.dist_map(a)[b[0], b[1]])

    def next_move_towards(self, start, goal):
        if start == goal:
            return Move.STAY
        dmap = self.dist_map(goal)  # khoảng cách MỌI ô tới goal
        if dmap[start[0], start[1]] <= 0:
            return Move.STAY
        # chọn hàng xóm có khoảng cách-tới-goal nhỏ nhất.
        best_move, best_d = Move.STAY, dmap[start[0], start[1]]
        for (npos, mv) in self.neighbors(start):
            d = dmap[npos[0], npos[1]]
            if d != -1 and d < best_d:
                best_d, best_move = d, mv
        return best_move

    def farthest_move_from(self, start, threat):
        dmap = self.dist_map(threat)
        best_move, best_score = Move.STAY, dmap[start[0], start[1]] + 0.1 * self.open_degree(start)
        for (npos, mv) in self.neighbors(start):
            d = dmap[npos[0], npos[1]]
            if d == -1:
                continue
            score = d + 0.1 * self.open_degree(npos)
            if score > best_score:
                best_score, best_move = score, mv
        return best_move


#  Sinh nước đi
def ghost_moves(graph: MazeGraph, pos):
    moves = [(Move.STAY, pos)]
    for (npos, mv) in graph.neighbors(pos):
        moves.append((mv, npos))
    return moves


def pacman_moves(graph: MazeGraph, pos, speed: int):
    moves = [(Move.STAY, pos)]
    for m in ORTHO_MOVES:
        dr, dc = m.value
        cur = pos
        for s in range(1, speed + 1):
            nxt = (cur[0] + dr, cur[1] + dc)
            if not graph.passable(nxt[0], nxt[1]):
                break
            cur = nxt
            action = m if s == 1 else (m, s)
            moves.append((action, cur))
    return moves
