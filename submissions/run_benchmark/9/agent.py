import sys
from pathlib import Path
import numpy as np
import random
import heapq

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
from collections import deque

class PacmanAgent(BasePacmanAgent):
    """Smart Sweeper Pacman: Tuần tra tuần tự toàn bản đồ với cơ chế Timeout Waypoint chống kẹt góc"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Smart Sweeper Pacman"
        
        self.enemy_history = deque(maxlen=5)
        self.my_history = deque(maxlen=8)
        self.last_move = None
        
        self.waypoints = []
        self.current_wp_index = 0
        self.steps_at_current_wp = 0 # Đếm số bước kẹt tại 1 waypoint

    def _generate_strategic_waypoints(self, map_state):
        """
        Hàm sinh tọa độ tuần tra động dựa trên tỷ lệ hình học của bản đồ.
        Đảm bảo tính tổng quát hóa trên mọi kích thước không gian.
        """
        h, w = map_state.shape
        cy, cx = h // 2, w // 2  # Trọng tâm bản đồ (Ví dụ map 21x21 -> cy=10, cx=10)
        
        # Tính toán các vùng phân phối dựa trên khoảng cách tương đối
        raw_points = [
            (cy - 1, cx),                             # (9, 10) - Hub
            (cy // 2, cx + cx // 4 + 1),              # (5, 12) - Sào huyệt phải
            (cy + cy // 3, cx + 1),                   # (13, 11) - Trung tâm dưới
            (cy - 1, cx // 2),                        # (9, 5) - Sào huyệt trái
            (2, w - cx // 2 - 1),                     # (2, 15) - Góc trên phải
            (2, cx // 2),                             # (2, 5) - Góc trên trái
            (h - 1, cx // 2),                         # (20, 5) - Góc dưới trái
            (h - 1, w - cx // 2 - 1)                  # (20, 15) - Góc dưới phải
        ]
        
        valid_waypoints = []
        for r, c in raw_points:
            # Ép kiểu và đảm bảo không vượt quá biên ma trận
            r = max(0, min(h - 1, int(r)))
            c = max(0, min(w - 1, int(c)))
            
            # Nếu toán học rơi trúng ô tường (1), tịnh tiến ra ô trống gần nhất
            if map_state[r, c] == 1:
                r, c = self._find_nearest_empty(r, c, map_state)
                
            valid_waypoints.append((r, c))
            
        return valid_waypoints

    def _find_nearest_empty(self, r, c, map_state):
        """Tìm ô trống gần nhất nếu điểm chiến lược rơi vào tường"""
        h, w = map_state.shape
        queue = deque([(r, c)])
        visited = {(r, c)}
        
        while queue:
            curr_r, curr_c = queue.popleft()
            if map_state[curr_r, curr_c] != 1:
                return curr_r, curr_c
                
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return r, c

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        height, width = map_state.shape
        target = None

        if not self.waypoints:
            self.waypoints = self._generate_strategic_waypoints(map_state)
        
        # 1. Thấy Ghost trực tiếp -> Hủy bỏ đi tuần, lập tức truy đuổi và đón đầu
        if enemy_position is not None:
            self.enemy_history.append(enemy_position)
            target = self._get_interception_target(my_position, enemy_position, map_state)
            self.steps_at_current_wp = 0
        else:
            # 2. Đang đi tuần tra theo danh sách Waypoint
            current_target = self.waypoints[self.current_wp_index]
            self.steps_at_current_wp += 1
            
            # Điều kiện chuyển Waypoint: Đã tới nơi HOẶC bị kẹt quá 18 bước ở waypoint đó
            if (abs(my_position[0] - current_target[0]) + abs(my_position[1] - current_target[1]) <= 1) or (self.steps_at_current_wp > 18):
                self.current_wp_index = (self.current_wp_index + 1) % len(self.waypoints)
                current_target = self.waypoints[self.current_wp_index]
                self.steps_at_current_wp = 0  # Reset bộ đếm cho waypoint mới
                
            target = current_target

        # 3. Chống kẹt vòng lặp cục bộ (Anti-loop)
        self.my_history.append(my_position)
        is_looping = len(self.my_history) == 8 and len(set(self.my_history)) <= 3
        if is_looping:
            if enemy_position is not None:
                target = enemy_position
            else:
                # Nếu đi vòng lặp vô ích, lập tức nhảy cóc sang waypoint tiếp theo
                self.current_wp_index = (self.current_wp_index + 1) % len(self.waypoints)
                target = self.waypoints[self.current_wp_index]
                self.steps_at_current_wp = 0

        # 4. Tìm đường A* tối ưu tốc độ và phạt bẻ lái
        path = self._speed_aware_astar(my_position, target, map_state, self.last_move)
        
        if not path:
            fallback = [m for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT] if self._is_valid_move(my_position, m, map_state)]
            if fallback:
                m = random.choice(fallback)
                self.last_move = m
                return (m, 1)
            return (Move.STAY, 1)

        next_move = path[0]
        steps = 1
        
        # 5. Kích hoạt bứt tốc Speed = 2 an toàn
        if self.pacman_speed > 1:
            nr1 = my_position[0] + next_move.value[0]
            nc1 = my_position[1] + next_move.value[1]
            if len(path) > 1 and path[0] == path[1]:
                if self._is_valid_move((nr1, nc1), next_move, map_state):
                    if enemy_position is None or (nr1, nc1) != enemy_position:
                        steps = min(self.pacman_speed, 2)

        if is_looping and enemy_position is None:
            valid_moves = [m for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT] if self._is_valid_move(my_position, m, map_state)]
            if valid_moves:
                next_move = random.choice(valid_moves)
                steps = 1

        self.last_move = next_move
        return (next_move, steps)

    def _get_interception_target(self, my_pos, ghost_pos, map_state):
        if len(self.enemy_history) >= 2:
            prev = self.enemy_history[-2]
            curr = self.enemy_history[-1]
            dr, dc = curr[0] - prev[0], curr[1] - prev[1]
            dist = abs(my_pos[0] - ghost_pos[0]) + abs(my_pos[1] - ghost_pos[1])
            if dist > 3:
                pred = (ghost_pos[0] + dr * 2, ghost_pos[1] + dc * 2)
                pred = (max(0, min(map_state.shape[0]-1, int(pred[0]))), max(0, min(map_state.shape[1]-1, int(pred[1]))))
                if map_state[pred] != 1:
                    return pred
        return ghost_pos

    def _speed_aware_astar(self, start, goal, map_state, initial_last_move):
        frontier = []
        counter = 0
        heapq.heappush(frontier, (0, counter, start[0], start[1], initial_last_move))
        
        came_from = {}
        cost_so_far = {(start, initial_last_move): 0}
        best_goal_state = None
        min_dist = float('inf')
        closest_state = None

        while frontier:
            _, _, r, c, last_move = heapq.heappop(frontier)
            curr_pos = (r, c)
            curr_state = (curr_pos, last_move)
            
            if curr_pos == goal:
                best_goal_state = curr_state
                break
                
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if not self._is_valid_move(curr_pos, move, map_state):
                    continue
                next_pos = (r + move.value[0], c + move.value[1])
                next_state = (next_pos, move)
                
                step_cost = 1.0
                if last_move is not None and move != last_move:
                    step_cost = 1.2
                    
                new_cost = cost_so_far[curr_state] + step_cost
                if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                    cost_so_far[next_state] = new_cost
                    dist = abs(goal[0] - next_pos[0]) + abs(goal[1] - next_pos[1])
                    priority = new_cost + dist
                    counter += 1
                    heapq.heappush(frontier, (priority, counter, next_pos[0], next_pos[1], move))
                    came_from[next_state] = (curr_state, move)
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_state = next_state
                        
        if best_goal_state is None:
            best_goal_state = closest_state if closest_state else (start, initial_last_move)
            
        path = []
        curr = best_goal_state
        while curr in came_from:
            prev_state, move = came_from[curr]
            path.append(move)
            curr = prev_state
            
        path.reverse()
        return path

    def _is_valid_move(self, pos, move, map_state):
        nr, nc = pos[0] + move.value[0], pos[1] + move.value[1]
        if 0 <= nr < map_state.shape[0] and 0 <= nc < map_state.shape[1]:
            return map_state[nr, nc] != 1
        return False
class GhostAgent(BaseGhostAgent):
    """Surgical Ghost: Chạy nước rút và né tránh linh hoạt trên mọi kích thước bản đồ, không dùng Hardcode"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Surgical Ghost (Dynamic Map)"
        self.last_known_pacman = None
        self.turns_since_seen = 999
        self.mode = "SPRINT"
        self.sprint_target = None
        self.camp_target = None

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        # Tự động đo kích thước bản đồ thời gian thực
        height, width = map_state.shape
        # Tự động tính toán tâm bản đồ làm mối đe dọa mặc định
        center_pos = (height // 2, width // 2)
        
        # 1. CẬP NHẬT TRÍ NHỚ
        if enemy_position is not None:
            self.last_known_pacman = enemy_position
            self.turns_since_seen = 0
        else:
            self.turns_since_seen += 1

        # 2. CHUYỂN ĐỔI TRẠNG THÁI TỐI ƯU
        if enemy_position is not None or self.turns_since_seen <= 5:
            self.mode = "KITE"
            self.sprint_target = None
            self.camp_target = None
        elif step_number <= 8:
            self.mode = "SPRINT"
        else:
            self.mode = "CAMP"

        # 3. THỰC THI CHIẾN THUẬT
        if self.mode == "SPRINT":
            if not self.sprint_target:
                # lấy điểm xa trung tâm nhất làm mục tiêu nước rút (các góc)
                self.sprint_target = self._find_farthest_point(center_pos, map_state)
                
            if my_position != self.sprint_target:
                path = self._bfs_path(my_position, self.sprint_target, map_state, prefer_horizontal=True)
                if path: return path[0]
            return Move.STAY

        elif self.mode == "CAMP":
            if not self.camp_target:
                # Nếu chưa từng thấy Pacman, coi trung tâm là mối đe dọa để tìm hầm ở viền ngoài
                threat = self.last_known_pacman if self.last_known_pacman else center_pos
                self.camp_target = self._find_ultimate_sanctuary(threat, map_state)

            if my_position != self.camp_target:
                path = self._bfs_path(my_position, self.camp_target, map_state)
                if path: return path[0]
            return Move.STAY

        elif self.mode == "KITE":
            # Xử lý thả diều linh hoạt trên mọi map
            threat = self.last_known_pacman if self.last_known_pacman else center_pos
            pacman_distances = self._bfs_distances(threat, map_state)

            valid_moves = []
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nr, nc = my_position[0] + move.value[0], my_position[1] + move.value[1]
                if 0 <= nr < height and 0 <= nc < width and map_state[nr, nc] != 1:
                    valid_moves.append((move, (nr, nc)))

            if not valid_moves:
                return Move.STAY

            best_move = valid_moves[0][0]
            best_score = -float('inf')

            for move, next_pos in valid_moves:
                score = 0
                dist = pacman_distances.get(next_pos, 0)
                
                score += dist * 1000
                
                if self._has_line_of_sight(next_pos, threat, map_state):
                    score -= 50000 
                
                exits = len(self._get_valid_neighbors(next_pos, map_state))
                if exits <= 1:
                    score -= 80000 
                elif exits >= 3:
                    score += 500   
                    
                edge_distance = min(next_pos[0], height - 1 - next_pos[0], next_pos[1], width - 1 - next_pos[1])
                score -= edge_distance * 10 

                if score > best_score:
                    best_score = score
                    best_move = move
                    
            return best_move

    # ================= HÀM BỔ TRỢ =================

    def _find_farthest_point(self, start_pos, map_state):
        distances = self._bfs_distances(start_pos, map_state)
        height, width = map_state.shape
        best_spot = (1, 1)
        max_dist = -1
        
        for r in range(height):
            for c in range(width):
                if map_state[r, c] == 1: continue
                d = distances.get((r, c), 0)
                if d >= max_dist:
                    max_dist = d
                    best_spot = (r, c)
        return best_spot

    def _find_ultimate_sanctuary(self, threat_pos, map_state):
        distances = self._bfs_distances(threat_pos, map_state)
        height, width = map_state.shape
        best_spot = (1, 1)
        max_score = -float('inf')

        for r in range(height):
            for c in range(width):
                if map_state[r, c] == 1: continue
                exits = len(self._get_valid_neighbors((r, c), map_state))
                if exits >= 2:
                    d = distances.get((r, c), 0)
                    edge_distance = min(r, height - 1 - r, c, width - 1 - c)
                    score = d - (edge_distance * 2)
                    
                    if score > max_score:
                        max_score = score
                        best_spot = (r, c)
        return best_spot

    def _bfs_path(self, start, target, map_state, prefer_horizontal=False):
        if start == target: return []
        queue = deque([(start, [])])
        visited = {start}
        
        directions = [Move.RIGHT, Move.LEFT, Move.UP, Move.DOWN] if prefer_horizontal else [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        
        while queue:
            curr, path = queue.popleft()
            if curr == target: return path
            for m in directions:
                nr, nc = curr[0] + m.value[0], curr[1] + m.value[1]
                if 0 <= nr < map_state.shape[0] and 0 <= nc < map_state.shape[1]:
                    if map_state[nr, nc] != 1 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [m]))
        return []

    def _bfs_distances(self, start, map_state):
        distances = {start: 0}
        queue = deque([start])
        while queue:
            curr = queue.popleft()
            d = distances[curr]
            for nr, nc in self._get_valid_neighbors(curr, map_state):
                if (nr, nc) not in distances:
                    distances[(nr, nc)] = d + 1
                    queue.append((nr, nc))
        return distances

    def _get_valid_neighbors(self, pos, map_state):
        neighbors = []
        for m in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pos[0] + m[0], pos[1] + m[1]
            if 0 <= nr < map_state.shape[0] and 0 <= nc < map_state.shape[1]:
                if map_state[nr, nc] != 1:
                    neighbors.append((nr, nc))
        return neighbors

    def _has_line_of_sight(self, p1, p2, map_state):
        r1, c1 = p1
        r2, c2 = p2
        if r1 == r2:
            step = 1 if c2 > c1 else -1
            for c in range(c1 + step, c2, step):
                if map_state[r1, c] == 1: return False
            return True
        if c1 == c2:
            step = 1 if r2 > r1 else -1
            for r in range(r1 + step, r2, step):
                if map_state[r, c1] == 1: return False
            return True
        return False