import sys
import time
from pathlib import Path
from collections import deque
import heapq
import math
import numpy as np


src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


# Pacman chỉ dùng 4 hướng chính; số bước đi thẳng được xử lý ở legal_pacman_actions().
PACMAN_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

# Ghost đi 1 ô mỗi lượt hoặc đứng yên.
GHOST_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]

# Seek thắng khi Manhattan distance < 2, tức là 0 hoặc 1.
CAPTURE_DISTANCE = 2

# INF/BAD_SCORE là các mốc điểm lớn để biểu diễn "rất xa" hoặc "rất xấu".
INF = 10 ** 6
BAD_SCORE = -10 ** 6


def manhattan(a, b):
    # Khoảng cách Manhattan: đi theo hàng/cột, không đi chéo.
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def next_cell(pos, move):
    # Move.value là vector (dr, dc). Ví dụ Move.UP = (-1, 0).
    dr, dc = move.value
    return (pos[0] + dr, pos[1] + dc)


class MapHelper:
    """
    Gom các hàm xử lý map vào một chỗ để Pacman và Ghost dùng chung.

    Class này dùng để:
    - Kiểm tra ô đi được/tường.
    - Sinh action hợp lệ cho Pacman/Ghost.
    - Tính khoảng cách BFS/A*.
    - Cache lại kết quả vì map 21x21 nhỏ nhưng mỗi step gọi rất nhiều lần.
    """

    def __init__(self, map_state, pacman_speed):
        self.map_state = np.asarray(map_state)
        self.rows, self.cols = self.map_state.shape
        self.pacman_speed = max(1, int(pacman_speed))

        # Danh sách các ô trống, dùng để phân tích map nếu cần.
        self.open_cells = []
        self.cell_id = {}
        self.prepare_cells()

        # Cache để tránh tính lại nhiều lần trong cùng một map.
        # Map chỉ 21x21 nên các cache này nhỏ, đổi lại tốc độ nhanh hơn.
        self.ghost_move_cache = {}
        self.ghost_pos_cache = {}
        self.pac_action_cache = {}
        self.pac_result_cache = {}
        self.maze_dist_cache = {}
        self.turn_dist_cache = {}
        self.capture_dist_cache = {}
        self.exit_cache = {}
        self.junction_cache = {}
        self.safe_area_cache = {}

    def prepare_cells(self):
        idx = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.map_state[r, c] == 0:
                    pos = (r, c)
                    self.open_cells.append(pos)
                    self.cell_id[pos] = idx
                    idx += 1

    def valid_pos(self, pos):
        # Theo đề: 0 là ô đi được, 1 là tường.
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.map_state[r, c] == 0

    def legal_ghost_moves(self, pos):
        # Ghost chỉ đi 1 ô hoặc STAY, nên chỉ cần kiểm tra ô kế bên có hợp lệ không.
        if pos in self.ghost_move_cache:
            return self.ghost_move_cache[pos]

        moves = []
        for move in GHOST_MOVES:
            if self.valid_pos(next_cell(pos, move)):
                moves.append(move)

        if not moves:
            moves = [Move.STAY]

        self.ghost_move_cache[pos] = moves
        return moves

    def legal_ghost_positions(self, pos):
        if pos not in self.ghost_pos_cache:
            self.ghost_pos_cache[pos] = [next_cell(pos, move) for move in self.legal_ghost_moves(pos)]
        return self.ghost_pos_cache[pos]

    def legal_pacman_actions(self, pos, allow_stay=False):
        # Pacman có thể đi 1..pacman_speed ô theo cùng một hướng.
        # Nếu gặp tường giữa đường thì dừng.
        if not allow_stay and pos in self.pac_action_cache:
            return self.pac_action_cache[pos]

        actions = []
        for move in PACMAN_MOVES:
            current = pos
            for steps in range(1, self.pacman_speed + 1):
                current = next_cell(current, move)
                if not self.valid_pos(current):
                    break
                actions.append((move, steps))

        if allow_stay:
            actions.append((Move.STAY, 1))
        if not actions:
            actions = [(Move.STAY, 1)]

        if not allow_stay:
            self.pac_action_cache[pos] = actions
        return actions

    def apply_pacman_action(self, pos, action):
        # Mô phỏng Pacman sau khi thực hiện (Move, steps).
        # Hàm này chỉ trả vị trí cuối, còn việc thắng/thua do Arena kiểm tra sau khi cả hai agent cùng đi.
        key = (pos, action)
        if key in self.pac_result_cache:
            return self.pac_result_cache[key]

        move, steps = action
        current = pos
        if move != Move.STAY:
            for _ in range(max(1, int(steps))):
                nxt = next_cell(current, move)
                if not self.valid_pos(nxt):
                    break
                current = nxt

        self.pac_result_cache[key] = current
        return current

    def bfs_from(self, start):
        # BFS theo từng ô thường, dùng để biết khoảng cách thật có xét tường.
        # Không dùng Manhattan ở đây vì Manhattan bị sai khi có tường chắn.
        if start in self.maze_dist_cache:
            return self.maze_dist_cache[start]

        dist = {start: 0}
        queue = deque([start])

        while queue:
            cur = queue.popleft()
            for move in PACMAN_MOVES:
                nxt = next_cell(cur, move)
                if nxt not in dist and self.valid_pos(nxt):
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)

        self.maze_dist_cache[start] = dist
        return dist

    def maze_distance(self, start, goal):
        if not self.valid_pos(start) or not self.valid_pos(goal):
            return INF
        return self.bfs_from(start).get(goal, INF)

    def pacman_turns_from(self, start):
        """
        BFS theo "lượt" của Pacman.

        Khác BFS thường:
        - BFS thường: mỗi cạnh = đi 1 ô.
        - Ở đây: mỗi cạnh = 1 action của Pacman, có thể đi 1 hoặc 2 ô thẳng.
        """
        if start in self.turn_dist_cache:
            return self.turn_dist_cache[start]

        dist = {start: 0}
        queue = deque([start])

        while queue:
            cur = queue.popleft()
            for action in self.legal_pacman_actions(cur):
                nxt = self.apply_pacman_action(cur, action)
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)

        self.turn_dist_cache[start] = dist
        return dist

    def pacman_turn_distance(self, start, goal):
        if not self.valid_pos(start) or not self.valid_pos(goal):
            return INF
        return self.pacman_turns_from(start).get(goal, INF)

    def capture_zone(self, ghost_pos):
        # Pacman không cần đứng đúng ô của Ghost; chỉ cần Manhattan distance < 2.
        # Vì vậy vùng bắt gồm ô Ghost đang đứng và các ô kề cạnh hợp lệ.
        cells = [ghost_pos]
        for move in PACMAN_MOVES:
            pos = next_cell(ghost_pos, move)
            if self.valid_pos(pos):
                cells.append(pos)
        return cells

    def capture_turn_distance(self, pac_pos, ghost_pos):
        """
        Số lượt ít nhất để Pacman đứng vào vùng có thể bắt Ghost,
        giả sử Ghost đứng yên tại ghost_pos.

        - Pacman muốn số này nhỏ.
        - Ghost muốn số này lớn.
        """
        key = (pac_pos, ghost_pos)
        if key in self.capture_dist_cache:
            return self.capture_dist_cache[key]

        if not self.valid_pos(pac_pos) or not self.valid_pos(ghost_pos):
            return INF

        best = INF
        for cell in self.capture_zone(ghost_pos):
            best = min(best, self.pacman_turn_distance(pac_pos, cell))

        self.capture_dist_cache[key] = best
        return best

    def pacman_can_capture_now(self, pac_pos, ghost_pos):
        return manhattan(pac_pos, ghost_pos) < CAPTURE_DISTANCE

    def astar_by_turns(self, start, goal):
        """
        A* trên action-space của Pacman.

        Mỗi node là một vị trí.
        Mỗi cạnh là một action hợp lệ (Move, steps), chi phí = 1 lượt.
        Heuristic dùng Manhattan / pacman_speed để ước lượng số lượt còn lại.
        """
        if not self.valid_pos(start) or not self.valid_pos(goal):
            return [start]

        heap = []
        count = 0
        heapq.heappush(heap, (manhattan(start, goal) / self.pacman_speed, count, start))
        cost = {start: 0}
        parent = {start: None}

        while heap:
            _, _, cur = heapq.heappop(heap)
            if cur == goal:
                break

            for action in self.legal_pacman_actions(cur):
                nxt = self.apply_pacman_action(cur, action)
                new_cost = cost[cur] + 1
                if new_cost < cost.get(nxt, INF):
                    cost[nxt] = new_cost
                    parent[nxt] = cur
                    count += 1
                    priority = new_cost + manhattan(nxt, goal) / self.pacman_speed
                    heapq.heappush(heap, (priority, count, nxt))

        if goal not in parent:
            return [start]

        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    def action_to_reach(self, start, target):
        best_action = (Move.STAY, 1)
        best_dist = INF

        for action in self.legal_pacman_actions(start, allow_stay=True):
            pos_after = self.apply_pacman_action(start, action)
            if pos_after == target:
                return action

            d = self.maze_distance(pos_after, target)
            if d < best_dist:
                best_dist = d
                best_action = action

        return best_action

    def exit_count(self, pos):
        # Đếm số hướng thoát ở một ô. >=3 thường được xem là ngã ba/ngã tư.
        if pos in self.exit_cache:
            return self.exit_cache[pos]

        count = 0
        for move in PACMAN_MOVES:
            if self.valid_pos(next_cell(pos, move)):
                count += 1

        self.exit_cache[pos] = count
        return count

    def is_junction(self, pos):
        return self.exit_count(pos) >= 3

    def safe_area_size(self, start, pacman_pos, limit=5, danger_turns=1):
        """
        Flood fill giới hạn quanh Ghost.

        Không chỉ xét ô hiện tại, mà nhìn quanh vài bước để xem
        vùng đó có nhiều ô còn tương đối an toàn không.
        Nếu vùng an toàn nhỏ quá thì dễ là hẻm cụt/túi chết.
        """
        key = (start, pacman_pos, limit, danger_turns)
        if key in self.safe_area_cache:
            return self.safe_area_cache[key]

        if not self.valid_pos(start):
            return 0

        total = 0
        queue = deque([(start, 0)])
        seen = {start}

        while queue:
            cur, depth = queue.popleft()
            if self.capture_turn_distance(pacman_pos, cur) > danger_turns:
                total += 1

            if depth >= limit:
                continue

            for nxt in self.legal_ghost_positions(cur):
                if nxt not in seen and self.valid_pos(nxt):
                    seen.add(nxt)
                    queue.append((nxt, depth + 1))

        self.safe_area_cache[key] = total
        return total

    def junction_distance(self, start, max_depth=6):
        # Tìm khoảng cách tới ngã ba/ngã tư gần nhất.
        # Ghost thường muốn ở gần junction để có nhiều lựa chọn né.
        key = (start, max_depth)
        if key in self.junction_cache:
            return self.junction_cache[key]

        queue = deque([(start, 0)])
        seen = {start}

        while queue:
            cur, depth = queue.popleft()
            if self.is_junction(cur):
                self.junction_cache[key] = depth
                return depth
            if depth >= max_depth:
                continue
            for nxt in self.legal_ghost_positions(cur):
                if nxt not in seen and self.valid_pos(nxt):
                    seen.add(nxt)
                    queue.append((nxt, depth + 1))

        self.junction_cache[key] = max_depth + 1
        return max_depth + 1

    def dead_end_depth(self, start, max_depth=8):
        """
        Đo xem từ ô hiện tại đi bao lâu mới tới được junction.

        Giá trị lớn nghĩa là đang ở sâu trong hành lang/corridor,
        dễ bị Pacman cắt đầu hơn. Hàm này dùng để tránh các đường cụt sâu.
        """
        queue = deque([(start, 0)])
        seen = {start}

        while queue:
            cur, depth = queue.popleft()
            if self.exit_count(cur) >= 3:
                return depth
            if depth >= max_depth:
                return max_depth + 1
            for nxt in self.legal_ghost_positions(cur):
                if nxt != cur and nxt not in seen and self.valid_pos(nxt):
                    seen.add(nxt)
                    queue.append((nxt, depth + 1))

        return 0

    def ordered_pacman_actions(self, pos, target):
        actions = self.legal_pacman_actions(pos)
        return sorted(
            actions,
            key=lambda action: (
                self.capture_turn_distance(self.apply_pacman_action(pos, action), target),
                self.maze_distance(self.apply_pacman_action(pos, action), target),
                -action[1]
            )
        )

    def ordered_ghost_positions_for_pacman(self, pac_pos, ghost_pos):
        positions = self.legal_ghost_positions(ghost_pos)
        return sorted(
            positions,
            key=lambda pos: (
                -self.capture_turn_distance(pac_pos, pos),
                -self.safe_area_size(pos, pac_pos, limit=4, danger_turns=1),
                -self.exit_count(pos)
            )
        )


class PacmanAgent(BasePacmanAgent):
    """
    Seek agent.

    1. Nếu có nước đi chắc chắn bắt được sau khi Ghost cũng phản ứng -> đi ngay.
    2. Nếu còn xa hoặc Ghost bị hạn chế đường thoát -> dùng A*.
    3. Nếu gần/vùng mở -> dùng lookahead để xét các hướng Ghost có thể né.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "PacmanAgent"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.last_seen_enemy = None
        self.map_helper = None
        self.map_key = None
        self.deadline = 0.0

        # Lưu vài vị trí/khoảng cách gần đây để phát hiện Pacman bị lặp vòng
        # hoặc đuổi mãi mà không tiến gần Ghost.
        self.recent_pac_positions = deque(maxlen=8)
        self.recent_capture_dists = deque(maxlen=6)

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        self.deadline = time.perf_counter() + 0.82
        my_position = tuple(my_position)

        try:
            if enemy_position is not None:
                self.last_seen_enemy = tuple(enemy_position)

            target = tuple(enemy_position) if enemy_position is not None else self.last_seen_enemy
            board = self.get_map_helper(map_state)

            if target is None or not board.valid_pos(target):
                return self.finish_action(board, my_position, self.move_to_open_area(board, my_position), None)

            actions = board.legal_pacman_actions(my_position)
            if not actions:
                return self.finish_action(board, my_position, (Move.STAY, 1), target)

            # Hai agent đi đồng thời, nên không được chỉ kiểm tra vị trí Ghost hiện tại.
            # Chỉ return sớm nếu action này vẫn bắt được với mọi nước Ghost có thể đi.
            ghost_replies = board.legal_ghost_positions(target)
            best_forced_action = None
            best_cover = -1
            best_forced_dist = INF
            for action in actions:
                pac_next = board.apply_pacman_action(my_position, action)
                cover = 0
                worst_after = 0
                for g_next in ghost_replies:
                    d = manhattan(pac_next, g_next)
                    worst_after = max(worst_after, d)
                    if d < CAPTURE_DISTANCE:
                        cover += 1
                if cover > best_cover or (cover == best_cover and worst_after < best_forced_dist):
                    best_cover = cover
                    best_forced_dist = worst_after
                    best_forced_action = action

            if best_cover == len(ghost_replies) and best_forced_action is not None:
                return self.finish_action(board, my_position, best_forced_action, target)

            fast_action = self.chase_with_astar(board, my_position, target)
            capture_dist = board.capture_turn_distance(my_position, target)
            ghost_mobility = len(board.legal_ghost_positions(target))
            ghost_safe_area = board.safe_area_size(target, my_position, limit=4, danger_turns=1)
            stuck = self.is_stuck(capture_dist)

            # Quy tắc chọn mode:
            # - Ghost đang ít đường thoát hoặc còn xa: đuổi nhanh bằng A* thường hiệu quả hơn.
            # - Ghost gần/vùng mở: cần lookahead để tránh bị Ghost rẽ hướng hoặc dụ vào hành lang.
            if not stuck and ((capture_dist > 5 and ghost_mobility <= 3) or capture_dist > 8):
                return self.finish_action(board, my_position, fast_action, target)

            best_action = fast_action
            best_score = self.evaluate_pacman_action(
                board,
                board.apply_pacman_action(my_position, fast_action),
                target,
                depth=2,
                action=fast_action
            )

            search_depth = 3 if capture_dist <= 5 or ghost_mobility >= 4 or ghost_safe_area >= 10 else 2

            for action in board.ordered_pacman_actions(my_position, target):
                pac_next = board.apply_pacman_action(my_position, action)
                score = self.evaluate_pacman_action(board, pac_next, target, search_depth, action)

                if score < best_score:
                    best_score = score
                    best_action = action

                if self.is_time_up():
                    break

            return self.finish_action(board, my_position, best_action, target)
        except Exception:
            return (Move.STAY, 1)

    def finish_action(self, board, my_position, action, target):
        pac_next = board.apply_pacman_action(my_position, action)
        self.recent_pac_positions.append(pac_next)
        if target is not None and board.valid_pos(target):
            self.recent_capture_dists.append(board.capture_turn_distance(pac_next, target))
        return action

    def is_stuck(self, current_capture_dist):
        if len(self.recent_capture_dists) < 4:
            return False
        # If the distance has not improved for several steps, avoid the cheap A* fallback
        # and let the bounded adversarial search pick a less repetitive pressure move.
        best_recent = min(self.recent_capture_dists)
        return current_capture_dist >= best_recent and len(set(self.recent_pac_positions)) <= 4

    def evaluate_pacman_action(self, board, pac_pos, ghost_pos, depth, action):
        # Chấm một action của Pacman bằng cách xét các phản ứng có thể của Ghost.
        # Score càng nhỏ càng tốt cho Pacman.
        ghost_options = board.ordered_ghost_positions_for_pacman(pac_pos, ghost_pos)
        worst_score = BAD_SCORE
        total_score = 0.0
        can_capture_count = 0
        checked = 0

        for g_next in ghost_options:
            checked += 1
            if board.pacman_can_capture_now(pac_pos, g_next):
                score = 0.0
                can_capture_count += 1
            else:
                score = 1.0 + self.search_value(board, pac_pos, g_next, depth - 1)

            worst_score = max(worst_score, score)
            total_score += score

            if self.is_time_up():
                break

        avg_score = total_score / max(1, checked)
        current_dist = board.capture_turn_distance(pac_pos, ghost_pos)
        ghost_mobility = len(board.legal_ghost_positions(ghost_pos))
        pac_mobility = len(board.legal_pacman_actions(pac_pos))
        corridor_penalty = 0.55 if board.exit_count(pac_pos) <= 2 and action[1] >= 2 else 0.0
        speed_bonus = 0.08 * action[1] if board.exit_count(pac_pos) >= 2 else 0.0
        loop_penalty = 0.65 if pac_pos in self.recent_pac_positions else 0.0

        # Công thức score:
        # - worst_score: trường hợp Ghost né tốt nhất, nên đặt trọng số lớn để chống worst-case.
        # - avg_score: trung bình các khả năng Ghost đi.
        # - current_dist/ghost_mobility: Ghost càng xa và càng nhiều đường thoát thì càng bất lợi.
        # - speed_bonus: thưởng nhẹ nếu Pacman tận dụng được speed.
        # - loop_penalty: phạt nếu Pacman quay lại vị trí gần đây.
        return (
            2.35 * worst_score
            + 0.55 * avg_score
            + 0.18 * current_dist
            + 0.22 * ghost_mobility
            + corridor_penalty
            + loop_penalty
            - 0.05 * pac_mobility
            - speed_bonus
            - 3.2 * can_capture_count
        )

    def search_value(self, board, pac_pos, ghost_pos, depth):
        # Lookahead dạng minimax nông:
        # Pacman chọn action tốt nhất, Ghost được giả định sẽ chọn phản ứng làm Pacman khó bắt nhất.
        if board.pacman_can_capture_now(pac_pos, ghost_pos):
            return 0.0
        if depth <= 0 or self.is_time_up():
            return self.static_score(board, pac_pos, ghost_pos)

        best_value = INF
        for action in board.ordered_pacman_actions(pac_pos, ghost_pos):
            p_next = board.apply_pacman_action(pac_pos, action)
            worst_reply = BAD_SCORE

            for g_next in board.ordered_ghost_positions_for_pacman(p_next, ghost_pos):
                if board.pacman_can_capture_now(p_next, g_next):
                    value = 1.0
                else:
                    value = 1.0 + self.search_value(board, p_next, g_next, depth - 1)

                worst_reply = max(worst_reply, value)
                if worst_reply >= best_value or self.is_time_up():
                    break

            best_value = min(best_value, worst_reply)
            if self.is_time_up():
                break

        return best_value

    def static_score(self, board, pac_pos, ghost_pos):
        capture_turns = board.capture_turn_distance(pac_pos, ghost_pos)
        if capture_turns >= INF:
            capture_turns = board.maze_distance(pac_pos, ghost_pos)

        ghost_escape_options = len(board.legal_ghost_positions(ghost_pos))
        ghost_safe_area = board.safe_area_size(ghost_pos, pac_pos, limit=4, danger_turns=1)
        return float(capture_turns) + 0.08 * manhattan(pac_pos, ghost_pos) + 0.28 * ghost_escape_options + 0.035 * ghost_safe_area

    def chase_with_astar(self, board, my_position, target):
        # Chase the closest cell that is enough to capture, not necessarily the ghost cell itself.
        best_goal = target
        best_turns = INF
        for cell in board.capture_zone(target):
            turns = board.pacman_turn_distance(my_position, cell)
            if turns < best_turns:
                best_turns = turns
                best_goal = cell

        path = board.astar_by_turns(my_position, best_goal)
        if len(path) >= 2:
            return board.action_to_reach(my_position, path[1])
        return self.move_to_open_area(board, my_position)

    def move_to_open_area(self, board, my_position):
        actions = board.legal_pacman_actions(my_position)
        if not actions:
            return (Move.STAY, 1)

        center = (board.rows // 2, board.cols // 2)
        return min(actions, key=lambda action: board.maze_distance(board.apply_pacman_action(my_position, action), center))

    def get_map_helper(self, map_state):
        arr = np.asarray(map_state)
        key = (arr.shape, arr.tobytes(), self.pacman_speed)
        if key != self.map_key:
            self.map_helper = MapHelper(arr, self.pacman_speed)
            self.map_key = key
        return self.map_helper

    def is_time_up(self):
        return time.perf_counter() >= self.deadline

class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    
    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Initialize any data structures you need

        self.name = "But then i have a very great idea. I use F5."

        # Memory for limited observation mode
        self.last_known_enemy_pos = None

        # multiplier to makes to agent less likely to move
        # when both agents are on the axis of a (horizontally) mirrored map on the first turn
        self.first_turn = 1
    
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
        
        # map to check the fastest time the enemy can reach a node
        enemy_bfs = self.make_distance_map(enemy_position, map_state, 2)

        # best_node as in best position to go from current position
        best_node = None
        best_val = None

        # compute our own distance map to min-max moves
        my_bfs = self.make_distance_map(my_position, map_state, 1, enemy_position)

        for p_node, p_val in my_bfs.items():
            if not self.is_reachable(p_node, my_bfs, enemy_bfs):
                continue

            neighbor_list = self.get_neighbor_values(p_node, enemy_bfs)

            # no neighbor means the node is unreachable for the seeker --> priority
            if len(neighbor_list) < 1:
                best_node = p_node
                break

            # difference in arrival time between the hider and seeker
            # a.k.a the time the hider has to think when it made its way here
            # however, if this value is negative, that means the hider cannot afford to arrive on time
            est_spare = min(neighbor_list) - p_val[0]

            # if this value is positive, we reduce this by a 0.5 modifier as the extra planning turns cannot be as good as plain evading turns
            if est_spare > 0:
                turn_in = self.time_til_turn(p_node, my_bfs)
                # the extra thinking turns only benefit the hider when there are more nodes to traverse
                # this is shallow as it assumes every turn is connected to another path and not a dead-end
                if turn_in < 0 or est_spare < turn_in:
                    est_spare = 0
                else:
                    est_spare = math.floor(est_spare * (0.5 + self.first_turn * 0.4))

            # benefit of arriving to considered node
            est_bonus = min(neighbor_list) - min(self.get_neighbor_values(my_position, enemy_bfs))

            # final score
            est_value = est_bonus + est_spare

            # replace best node if it meets condition
            if best_node is None or est_value > best_val[0] or (est_value == best_val[0] and self.estimate_distance(p_node, enemy_position) > self.estimate_distance(best_node, enemy_position)):
                best_node = p_node
                best_val = (est_value, est_spare)


        self.first_turn = 0

        if best_node == my_position:
            return (Move.STAY)

        # trace to get direction
        while my_bfs[best_node][1] != my_position:
            best_node = my_bfs[best_node][1]

        for m in [Move.UP, Move.LEFT, Move.RIGHT, Move.DOWN]:
            if (my_position[0] + m.value[0], my_position[1] + m.value[1]) == best_node:
                return m

        return (Move.STAY)

    # Helper methods (you can add more)
    
    def is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0

    def time_til_turn(self, pos: tuple, my_bfs: dict) -> int:
        """
        How many steps it will take to get to a potential junction.
        NOTE: This does not take the length of the derived paths into consideration.
        """

        counter = 0

        move_v = (pos[0] - my_bfs[pos][1][0], pos[1] - my_bfs[pos][1][1])
        if move_v == (0, 0):
            return 0
        
        cur_node = pos
        while my_bfs.get(cur_node):
            flip = (move_v[1], move_v[0])
            
            if my_bfs.get((cur_node[0] + flip[0], cur_node[1] + flip[1])) or my_bfs.get((cur_node[0] - flip[0], cur_node[1] - flip[1])):
                return counter
            
            cur_node = (cur_node[0] + move_v[0], cur_node[1] + move_v[1])
            counter += 1

        return -1 

    # multiplier is used for the enemy where they can reach further nodes
    def make_distance_map(self, pos: tuple, map_state: np.ndarray, multiplier: int = 1, enemy_pos: tuple = None) -> dict:
        """
        Use BFS to build a distance map to every other node.
        With customized max step for seeker and hider and enemy position acts as a wall (for hider only).
        """   

        frontier = deque([(pos)])
        dist_map = {pos: (0, pos)}

        multiplier = max(1, multiplier + 1)
        
        while len(frontier) > 0:
            node = frontier.popleft()
            
            for m in [(1, 0), (0, -1), (-1, 0), (0, 1)]:
                for i in range(1, multiplier):
                    next_node = (node[0] + i * m[0], node[1] + i * m[1])
                    if not self.is_valid_position(next_node, map_state) or next_node in dist_map or pos == enemy_pos:
                        break
                        
                    dist_map[next_node] = (dist_map[node][0] + 1, node)
                    frontier.append(next_node)

        return dist_map

    def estimate_distance(self, start: tuple, end: tuple) -> int:
        """
        Basic manhattan distance calculation with additional property taken into consideration.
        Is generally more useful when both seeker is closer to hider.
        """
        hor = abs(start[1] - end[1])
        ver = abs(start[0] - end[0])

        if ver == 0:
            return math.ceil(hor / 2)
        if hor == 0:
            return math.ceil(ver / 2)

        return hor + ver
    
    def get_neighbor_values(self, pos: tuple, bfs_map: dict) -> list:
        """
        Get (up to) four values from nodes that are adjacent to current tile.
        """
        neighbors = []
        for m in ([(1, 0), (0, 1), (-1, 0), (0, -1)]):
            next_pos = (pos[0] + m[0], pos[1] + m[1])
            if bfs_map.get(next_pos):
                neighbors.append(bfs_map[next_pos][0])

        return neighbors

    def is_reachable(self, node: tuple, my_bfs: dict, enemy_bfs: dict) -> bool:
        """
        Check if each node on the path is reachable by comparing arrival time of both agents.
        """
        trace = node
        while (my_bfs[trace][0] != 0):
            if my_bfs[trace][0] >= enemy_bfs[trace][0]:
                return False
            trace = my_bfs[trace][1]
        return True