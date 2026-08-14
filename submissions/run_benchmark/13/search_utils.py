"""
search_utils.py
================
Đồ thị và tiện ích cho Hide-and-Seek Arena.

Mục tiêu của bản V4:
- Sinh đúng action Pacman 1..speed ô theo MỘT hướng thẳng.
- Cache BFS và khoảng cách theo số lượt Pacman.
- Hỗ trợ vùng bắt, điểm chặn, hướng chạy, giao lộ và rollout ngắn.
"""

from collections import deque
import numpy as np

from environment import Move

ORTHO_MOVES = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def action_move(action):
    """Lấy Move từ Move hoặc tuple (Move, steps)."""
    return action[0] if isinstance(action, tuple) else action


def action_steps(action):
    """Số ô của action; STAY được xem là 0 ô."""
    move = action_move(action)
    if move == Move.STAY:
        return 0
    return int(action[1]) if isinstance(action, tuple) else 1


def direction_between(a, b):
    """Vector đơn vị từ a tới b nếu cùng hàng/cột và khác nhau; ngược lại None."""
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if dr == 0 and dc != 0:
        return (0, 1 if dc > 0 else -1)
    if dc == 0 and dr != 0:
        return (1 if dr > 0 else -1, 0)
    return None


def opposite_move(move):
    move = action_move(move) if move is not None else None
    return {
        Move.UP: Move.DOWN,
        Move.DOWN: Move.UP,
        Move.LEFT: Move.RIGHT,
        Move.RIGHT: Move.LEFT,
    }.get(move)


class MazeGraph:
    """
    Đồ thị mê cung với cache cho các truy vấn lặp lại.

    Lab 2 (Blind Adversary): map_state mỗi bước chỉ là QUAN SÁT CỤC BỘ (ô
    ngoài tầm nhìn = -1). Vì bản đồ thật KHÔNG đổi giữa các bước, ta duy trì
    MỘT graph xuyên suốt cả trận và gọi update() mỗi bước để hợp nhất (merge)
    quan sát mới vào bản đồ đã biết — không bao giờ "quên" ô đã từng thấy chỉ
    vì nó hiện đang nằm ngoài tầm nhìn (-1 chỉ có nghĩa 'chưa biết ở BƯỚC
    NÀY', không có nghĩa 'ô này không tồn tại').
    """

    def __init__(self, map_state: np.ndarray):
        self.map = np.array(map_state, copy=True)
        self.h, self.w = self.map.shape

        self._dist_cache = {}
        self._reach_cache = {}
        self._dead_end_cache = {}
        self._cycle_cache = {}
        self._pac_turn_cache = {}
        self._capture_turn_cache = {}
        self._max_reach_cache = {}
        self._shortest_path_cache = {}

        self._recompute_static()

    def _recompute_static(self):
        """
        Tính lại degree_map / corridor_map / corner_map / junction_map từ
        self.map hiện tại. Gọi lại mỗi khi self.map thay đổi (update()).
        Bản đồ tối đa 21x21 nên chi phí này luôn rất nhỏ (~vài trăm ô).
        """
        self.degree_map = np.full((self.h, self.w), -1, dtype=np.int8)
        for r in range(self.h):
            for c in range(self.w):
                if self.map[r, c] != 0:
                    continue
                degree = 0
                for move in ORTHO_MOVES:
                    dr, dc = move.value
                    if self.passable(r + dr, c + dc):
                        degree += 1
                self.degree_map[r, c] = degree

        self.corridor_map = np.zeros((self.h, self.w), dtype=bool)
        self.corner_map = np.zeros((self.h, self.w), dtype=bool)
        self.junction_map = self.degree_map >= 3

        for r in range(self.h):
            for c in range(self.w):
                if self.degree_map[r, c] != 2:
                    continue
                dirs = []
                for move in ORTHO_MOVES:
                    dr, dc = move.value
                    if self.passable(r + dr, c + dc):
                        dirs.append((dr, dc))
                if len(dirs) != 2:
                    continue
                d1, d2 = dirs
                if d1 == (-d2[0], -d2[1]):
                    self.corridor_map[r, c] = True
                else:
                    self.corner_map[r, c] = True

    def update(self, map_state: np.ndarray) -> bool:
        """
        Hợp nhất quan sát cục bộ mới nhất vào bản đồ đã biết (Lab 2).

        - Chỉ ghi đè các ô mà quan sát mới KHÔNG phải -1 (đã thấy rõ). Ô nằm
          ngoài tầm nhìn hiện tại (-1) giữ nguyên giá trị đã biết trước đó.
        - Vì mê cung tĩnh, một khi ô đã được xác định (0=trống hoặc 1=tường)
          giá trị đó không bao giờ đổi — nên merge kiểu "ưu tiên dữ liệu đã
          biết" là an toàn tuyệt đối, không có rủi ro xung đột.
        - Trả về True nếu có ô mới được khám phá (để bên gọi biết cache cũ
          không còn hợp lệ); tự động dọn cache khi cần.
        """
        new_map = np.asarray(map_state)
        if new_map.shape != self.map.shape:
            # Kích thước bản đồ lệch (không nên xảy ra vì mê cung tĩnh) ->
            # coi như trận mới, thay hẳn bản đồ.
            self.map = np.array(new_map, copy=True)
            self.h, self.w = self.map.shape
            self._clear_caches()
            self._recompute_static()
            return True

        known_mask = new_map != -1
        if not known_mask.any():
            return False

        newly_learned = known_mask & (self.map == -1)
        changed = newly_learned.any()
        if changed:
            self.map[known_mask] = new_map[known_mask]
            self._clear_caches()
            self._recompute_static()
        return bool(changed)

    def _clear_caches(self):
        self._dist_cache.clear()
        self._reach_cache.clear()
        self._dead_end_cache.clear()
        self._cycle_cache.clear()
        self._pac_turn_cache.clear()
        self._capture_turn_cache.clear()
        self._max_reach_cache.clear()
        self._shortest_path_cache.clear()

    # ------------------------------------------------------------------
    # Khám phá (Lab 2 — partial observability)
    # ------------------------------------------------------------------
    def frontier_cells(self):
        """
        Các ô TRỐNG đã biết mà có ít nhất một ô lân cận trực giao còn CHƯA
        biết (-1). Đây là 'biên khám phá' — di chuyển tới các ô này sẽ mở
        rộng tầm nhìn sang khu vực mới. Dùng làm mục tiêu khi chưa/không còn
        thấy đối thủ, để chủ động dò bản đồ thay vì đứng yên.
        """
        out = []
        for r in range(self.h):
            for c in range(self.w):
                if self.map[r, c] != 0:
                    continue
                for move in ORTHO_MOVES:
                    dr, dc = move.value
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.h and 0 <= nc < self.w and self.map[nr, nc] == -1:
                        out.append((r, c))
                        break
        return out

    def nearest_frontier(self, start):
        """Ô biên khám phá gần `start` nhất theo khoảng cách BFS thật (None nếu hết biên)."""
        start = tuple(start)
        best_pos, best_d = None, None
        for pos in self.frontier_cells():
            d = self.dist(start, pos)
            if d < 0:
                continue
            if best_d is None or d < best_d:
                best_d, best_pos = d, pos
        return best_pos

    def farthest_known_cell(self, start):
        """Ô trống đã biết XA start nhất theo BFS thật — dùng khi hết biên khám phá để tuần tra."""
        dmap = self.dist_map(start)
        idx = np.unravel_index(np.argmax(dmap), dmap.shape)
        if dmap[idx] <= 0:
            return None
        return (int(idx[0]), int(idx[1]))

    def signature(self):
        return (self.map.shape, hash(self.map.tobytes()))

    # ------------------------------------------------------------------
    # Ô và láng giềng
    # ------------------------------------------------------------------
    def passable(self, row, col) -> bool:
        return (
            0 <= row < self.h
            and 0 <= col < self.w
            and self.map[row, col] == 0
        )

    def neighbors(self, pos):
        r, c = pos
        out = []
        for move in ORTHO_MOVES:
            dr, dc = move.value
            nxt = (r + dr, c + dc)
            if self.passable(*nxt):
                out.append((nxt, move))
        return out

    def open_degree(self, pos) -> int:
        value = self.degree_map[pos[0], pos[1]]
        return int(value) if value >= 0 else 0

    def average_neighbor_degree(self, pos) -> float:
        neigh = self.neighbors(pos)
        if not neigh:
            return 0.0
        return sum(self.open_degree(p) for p, _ in neigh) / len(neigh)

    def is_corridor(self, pos) -> bool:
        return bool(self.corridor_map[pos[0], pos[1]])

    def is_corner(self, pos) -> bool:
        return bool(self.corner_map[pos[0], pos[1]])

    def is_junction(self, pos) -> bool:
        return bool(self.junction_map[pos[0], pos[1]])

    # ------------------------------------------------------------------
    # BFS distance và đường ngắn nhất
    # ------------------------------------------------------------------
    def dist_map(self, source):
        source = tuple(source)
        cached = self._dist_cache.get(source)
        if cached is not None:
            return cached

        dist = np.full((self.h, self.w), -1, dtype=np.int16)
        if not self.passable(*source):
            self._dist_cache[source] = dist
            return dist

        dist[source[0], source[1]] = 0
        q = deque([source])
        while q:
            r, c = q.popleft()
            nd = int(dist[r, c]) + 1
            for move in ORTHO_MOVES:
                dr, dc = move.value
                nr, nc = r + dr, c + dc
                if self.passable(nr, nc) and dist[nr, nc] == -1:
                    dist[nr, nc] = nd
                    q.append((nr, nc))

        self._dist_cache[source] = dist
        return dist

    def dist(self, a, b) -> int:
        a, b = tuple(a), tuple(b)
        return int(self.dist_map(a)[b[0], b[1]])

    def shortest_path_positions(self, start, goal):
        """Danh sách vị trí sau từng bước trên một đường BFS ngắn nhất."""
        start, goal = tuple(start), tuple(goal)
        key = (start, goal)
        cached = self._shortest_path_cache.get(key)
        if cached is not None:
            return list(cached)
        if start == goal:
            self._shortest_path_cache[key] = tuple()
            return []

        dmap = self.dist_map(goal)
        current_d = int(dmap[start[0], start[1]])
        if current_d < 0:
            return []

        current = start
        path = []
        while current != goal:
            candidates = []
            for npos, move in self.neighbors(current):
                d = int(dmap[npos[0], npos[1]])
                if d >= 0 and d < current_d:
                    candidates.append((d, npos, move))
            if not candidates:
                return []
            candidates.sort(key=lambda x: x[0])
            current_d, current, _ = candidates[0]
            path.append(current)

        self._shortest_path_cache[key] = tuple(path)
        return path

    def next_move_towards(self, start, goal):
        if start == goal:
            return Move.STAY
        dmap = self.dist_map(goal)
        current = int(dmap[start[0], start[1]])
        if current <= 0:
            return Move.STAY
        best = (current, Move.STAY)
        for npos, move in self.neighbors(start):
            d = int(dmap[npos[0], npos[1]])
            if 0 <= d < best[0]:
                best = (d, move)
        return best[1]

    # ------------------------------------------------------------------
    # Khoảng cách theo số lượt Pacman
    # ------------------------------------------------------------------
    def pacman_turn_map(self, source, speed: int):
        source = tuple(source)
        speed = max(1, int(speed))
        key = (source, speed)
        cached = self._pac_turn_cache.get(key)
        if cached is not None:
            return cached

        turns = np.full((self.h, self.w), -1, dtype=np.int16)
        if not self.passable(*source):
            self._pac_turn_cache[key] = turns
            return turns

        turns[source[0], source[1]] = 0
        q = deque([source])
        while q:
            pos = q.popleft()
            next_turn = int(turns[pos[0], pos[1]]) + 1
            for _, end in pacman_moves(self, pos, speed):
                if end == pos:
                    continue
                if turns[end[0], end[1]] == -1:
                    turns[end[0], end[1]] = next_turn
                    q.append(end)

        self._pac_turn_cache[key] = turns
        return turns

    def pacman_turn_distance(self, source, target, speed: int) -> int:
        target = tuple(target)
        return int(self.pacman_turn_map(source, speed)[target[0], target[1]])

    def capture_zone(self, target, capture_dist: int = 2):
        """Các ô Pacman có thể đứng để bắt target theo Manhattan < capture_dist."""
        target = tuple(target)
        out = []
        radius = max(0, int(capture_dist) - 1)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                pos = (target[0] + dr, target[1] + dc)
                if manhattan(pos, target) < capture_dist and self.passable(*pos):
                    out.append(pos)
        return out

    def pacman_capture_turn_distance(self, source, target, speed: int, capture_dist: int = 2):
        source, target = tuple(source), tuple(target)
        key = (source, target, int(speed), int(capture_dist))
        cached = self._capture_turn_cache.get(key)
        if cached is not None:
            return cached
        tmap = self.pacman_turn_map(source, speed)
        best = 10 ** 9
        for pos in self.capture_zone(target, capture_dist):
            value = int(tmap[pos[0], pos[1]])
            if 0 <= value < best:
                best = value
        result = -1 if best == 10 ** 9 else best
        self._capture_turn_cache[key] = result
        return result

    def pacman_action_towards(self, source, target, speed: int):
        """Action giảm số lượt tới target; ưu tiên đi đủ speed khi hòa."""
        best_action = Move.STAY
        best_key = (10 ** 9, 10 ** 9, 0)
        for action, end in pacman_moves(self, source, speed):
            turns = self.pacman_turn_distance(end, target, speed)
            if turns < 0:
                continue
            key = (turns, self.dist(end, target), -action_steps(action))
            if key < best_key:
                best_key = key
                best_action = action
        return best_action

    def pacman_action_to_capture_zone(self, source, target, speed: int, capture_dist: int = 2):
        """Action tiến nhanh nhất tới một ô có thể bắt target."""
        best_action = Move.STAY
        best_key = (10 ** 9, 10 ** 9, 0)
        zones = self.capture_zone(target, capture_dist)
        for action, end in pacman_moves(self, source, speed):
            tmap = self.pacman_turn_map(end, speed)
            turns = min(
                (int(tmap[p[0], p[1]]) for p in zones if int(tmap[p[0], p[1]]) >= 0),
                default=10 ** 9,
            )
            maze = min((self.dist(end, p) for p in zones), default=10 ** 9)
            key = (turns, maze, -action_steps(action))
            if key < best_key:
                best_key = key
                best_action = action
        return best_action

    # ------------------------------------------------------------------
    # Địa hình
    # ------------------------------------------------------------------
    def reachable_area(self, pos, radius: int) -> int:
        pos = tuple(pos)
        key = (pos, int(radius))
        cached = self._reach_cache.get(key)
        if cached is not None:
            return cached
        if not self.passable(*pos):
            self._reach_cache[key] = 0
            return 0

        seen = {pos}
        frontier = [pos]
        for _ in range(max(0, int(radius))):
            nxt = []
            for p in frontier:
                for npos, _ in self.neighbors(p):
                    if npos not in seen:
                        seen.add(npos)
                        nxt.append(npos)
            if not nxt:
                break
            frontier = nxt
        result = len(seen)
        self._reach_cache[key] = result
        return result

    def local_space(self, pos, radius: int = 3) -> int:
        return self.reachable_area(pos, radius)

    def free_space(self, pos, radius: int = 3) -> int:
        return self.reachable_area(pos, radius)

    def max_reachable_area(self, radius: int = 3) -> int:
        radius = int(radius)
        cached = self._max_reach_cache.get(radius)
        if cached is not None:
            return cached
        best = 1
        for r in range(self.h):
            for c in range(self.w):
                if self.map[r, c] == 0:
                    best = max(best, self.reachable_area((r, c), radius))
        self._max_reach_cache[radius] = best
        return best

    def dead_end_trend(self, pos, steps: int = 3) -> float:
        pos = tuple(pos)
        key = (pos, int(steps))
        cached = self._dead_end_cache.get(key)
        if cached is not None:
            return cached
        if not self.passable(*pos):
            return 0.0

        seen = {pos}
        frontier = [pos]
        layers = []
        for _ in range(max(1, int(steps))):
            nxt = []
            for p in frontier:
                for npos, _ in self.neighbors(p):
                    if npos not in seen:
                        seen.add(npos)
                        nxt.append(npos)
            if not nxt:
                layers.append(0.0)
                break
            layers.append(sum(self.open_degree(p) for p in nxt) / len(nxt))
            frontier = nxt
        trend = 0.0 if len(layers) < 2 else layers[-1] - layers[0]
        self._dead_end_cache[key] = trend
        return trend

    def cycle_potential(self, pos, radius: int = 5) -> float:
        pos = tuple(pos)
        key = (pos, int(radius))
        cached = self._cycle_cache.get(key)
        if cached is not None:
            return cached
        if not self.passable(*pos):
            return 0.0

        seen = {pos}
        frontier = [pos]
        for _ in range(max(0, int(radius))):
            nxt = []
            for p in frontier:
                for npos, _ in self.neighbors(p):
                    if npos not in seen:
                        seen.add(npos)
                        nxt.append(npos)
            if not nxt:
                break
            frontier = nxt
        edges_twice = sum(
            sum(1 for npos, _ in self.neighbors(p) if npos in seen)
            for p in seen
        )
        cycles = max(0.0, edges_twice / 2.0 - len(seen) + 1.0)
        self._cycle_cache[key] = cycles
        return cycles

    def clear_straight_distance(self, a, b) -> int:
        ar, ac = a
        br, bc = b
        if ar == br:
            step = 1 if bc > ac else -1
            for c in range(ac + step, bc, step):
                if not self.passable(ar, c):
                    return -1
            return abs(bc - ac)
        if ac == bc:
            step = 1 if br > ar else -1
            for r in range(ar + step, br, step):
                if not self.passable(r, ac):
                    return -1
            return abs(br - ar)
        return -1

    def speed_denial_score(self, ghost_pos, pac_pos) -> float:
        degree = self.open_degree(ghost_pos)
        score = 0.0
        if self.is_corner(ghost_pos):
            score += 2.0
        if degree >= 3:
            score += 2.5
        if self.is_corridor(ghost_pos):
            score -= 1.5
        if degree <= 1:
            score -= 4.0
        straight = self.clear_straight_distance(pac_pos, ghost_pos)
        if straight >= 0:
            score -= 2.5
            if straight <= 6:
                score -= (7 - straight) * 0.35
        else:
            score += 0.75
        score += min(2.0, 0.5 * self.cycle_potential(ghost_pos, 4))
        return score

    def forward_choke_points(self, start, direction, max_steps: int = 7):
        """Quét theo hướng G: trả các ngã rẽ/góc và điểm cuối hành lang."""
        if direction is None:
            return []
        if isinstance(direction, Move):
            direction = direction.value
        dr, dc = direction
        if abs(dr) + abs(dc) != 1:
            return []

        current = tuple(start)
        out = []
        for _ in range(max(1, int(max_steps))):
            nxt = (current[0] + dr, current[1] + dc)
            if not self.passable(*nxt):
                if current != start and current not in out:
                    out.append(current)
                break
            current = nxt
            if self.is_junction(current) or self.is_corner(current):
                out.append(current)
                if self.is_junction(current):
                    break
        if current != start and current not in out:
            out.append(current)
        return out

    def nearby_choke_points(self, start, radius: int = 6):
        """Ngã rẽ/góc gần start, xếp theo khoảng cách BFS."""
        dmap = self.dist_map(start)
        out = []
        for r in range(self.h):
            for c in range(self.w):
                d = int(dmap[r, c])
                if 0 < d <= radius and (self.is_junction((r, c)) or self.is_corner((r, c))):
                    out.append(((r, c), d))
        out.sort(key=lambda item: item[1])
        return [p for p, _ in out]

    def farthest_move_from(self, start, threat):
        candidates = ghost_moves(self, start)
        return max(
            candidates,
            key=lambda item: (
                self.dist(threat, item[1]),
                self.local_space(item[1], 3),
                self.cycle_potential(item[1], 4),
                self.open_degree(item[1]),
                int(item[0] != Move.STAY),
            ),
        )[0]


def ghost_moves(graph: MazeGraph, pos):
    moves = [(Move.STAY, tuple(pos))]
    for npos, move in graph.neighbors(pos):
        moves.append((move, npos))
    return moves


def pacman_moves(graph: MazeGraph, pos, speed: int):
    speed = max(1, int(speed))
    pos = tuple(pos)
    moves = [(Move.STAY, pos)]
    for move in ORTHO_MOVES:
        dr, dc = move.value
        current = pos
        for steps in range(1, speed + 1):
            nxt = (current[0] + dr, current[1] + dc)
            if not graph.passable(*nxt):
                break
            current = nxt
            action = move if steps == 1 else (move, steps)
            moves.append((action, current))
    return moves
