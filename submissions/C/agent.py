"""
Nhóm 08 - Improved Agent:
- PacmanAgent: DQN + A* Fallback + BELIEF TRACKING + INTERCEPTION
- GhostAgent: Minimax Iterative Deepening (giữ nguyên, đã tốt)
"""

import sys
from pathlib import Path
from collections import deque
from heapq import heappush, heappop
import random
import time
import math

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np

try:
    import torch
    from model import PacmanNet
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    PacmanNet = None

# ============================================================
# PACMAN AGENT — Improved với Belief Tracking + Interception
# ============================================================

class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) v2.0:
    1. DQN model (khi thấy địch rõ ràng)
    2. Belief Tracking: xác suất vị trí Ghost khi mất tầm nhìn
    3. Interception: chặn đầu Ghost thay vì chỉ đuổi thẳng
    4. A* Fallback: khi DQN fail hoặc không chắc
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Pacman v2.0 - Belief+Intercept"

        # --- Memory ---
        self.internal_map = None
        self.map_initialized = False
        self.last_move = None
        self.step_number = 0

        # --- Belief Tracking ---
        # belief[r][c] = xác suất Ghost đang ở (r, c)
        self.belief = None
        self.belief_initialized = False
        self.last_known_enemy_pos = None
        self.steps_since_seen = 0
        # Lịch sử vị trí Ghost (để tính velocity)
        self.enemy_history = deque(maxlen=5)
        
        # Lịch sử vị trí Pacman (để detect loop)
        self.my_pos_history = deque(maxlen=8)

        # --- Load DQN Model ---
        self.device = torch.device("cpu") if TORCH_AVAILABLE else None
        self.model = None
        if TORCH_AVAILABLE and PacmanNet:
            try:
                self.model = PacmanNet()
                current_dir = Path(__file__).parent
                model_path = current_dir / "pacman_smart_ghost.pt"
                if not model_path.exists():
                    model_path = current_dir / "pacman_dqn.pt"
                if model_path.exists():
                    self.model.load_state_dict(
                        torch.load(model_path, map_location=self.device)
                    )
                    self.model.eval()
                else:
                    self.model = None
            except Exception:
                self.model = None

    # ================================================================
    # MAIN STEP
    # ================================================================

    def step(self, map_state: np.ndarray, my_position: tuple,
             enemy_position: tuple, step_number: int):
        self.step_number = step_number

        # 1. Cập nhật bản đồ memory
        self._update_map_memory(map_state)

        # 2. Cập nhật Belief
        self._update_belief(map_state, my_position, enemy_position)

        # 3. Xác định target tốt nhất
        if enemy_position is not None:
            # Thấy Ghost trực tiếp
            target = enemy_position
            mode = "visible"
        elif self.last_known_enemy_pos is not None and self.steps_since_seen <= 8:
            # Mới mất dấu → dùng belief
            target = self._get_belief_target(my_position)
            mode = "belief"
        else:
            # Mất dấu lâu → explore
            target = None
            mode = "explore"

        # 4. Chọn nước đi
        chosen_move = Move.STAY
        path = None

        # Phát hiện bị kẹt trong vòng lặp (di chuyển qua lại giữa 2-3 ô)
        self.my_pos_history.append(my_position)
        is_looping = False
        if len(self.my_pos_history) == 8 and len(set(self.my_pos_history)) <= 3:
            is_looping = True

        if mode == "visible":
            ml_move = None
            # Nếu đang bị kẹt loop → BỎ QUA ML, dùng A* trực diện để thoát kẹt
            if not is_looping:
                ml_move = self._get_ml_action(map_state, my_position, enemy_position)
                
            if ml_move not in (None, Move.STAY):
                chosen_move = ml_move
                # Lấy A* path để verify speed=2 không overshoot ngã rẽ
                path = self.astar(my_position, enemy_position, self.internal_map)
            else:
                # DQN fail hoặc đang kẹt loop → dùng A*
                # Trực tiếp đuổi theo để phá loop, nếu không kẹt thì chặn đầu
                target_pos = enemy_position if is_looping else self._get_interception_target(my_position, enemy_position)
                path = self.astar(my_position, target_pos, self.internal_map)
                if path:
                    chosen_move = path[0]
                else:
                    chosen_move = self._greedy_toward(my_position, enemy_position)

        elif mode == "belief":
            # Dùng A* đến belief target
            path = self.astar(my_position, target, self.internal_map)
            if path:
                chosen_move = path[0]
            else:
                chosen_move = self._greedy_toward(my_position, target)

        else:
            # Explore: tìm frontier
            frontier = self._find_best_frontier(my_position)
            if frontier:
                path = self.astar(my_position, frontier, self.internal_map)
                if path:
                    chosen_move = path[0]
            if chosen_move == Move.STAY:
                chosen_move = self._random_valid_move(my_position)

        # 5. Tính steps (tận dụng speed=2 trên đường thẳng)
        steps = 1
        if chosen_move != Move.STAY and self.pacman_speed >= 2:
            # Nếu có thể đi 2 bước và hướng đi đó không đâm xuyên qua Ghost (overshoot)
            dist_to_ghost = self._manhattan(my_position, enemy_position) if enemy_position else 999
            
            # Chỉ đi 2 bước nếu khoảng cách đến ghost > 1, hoặc nếu hướng đi 2 bước đó 
            # ĐÚNG LÀ vị trí của Ghost (dist == 2 thẳng hàng).
            # Tránh trường hợp Ghost cách 1 ô, Pacman đi 2 ô thành ra đi xuyên qua Ghost.
            target_pos = (
                my_position[0] + chosen_move.value[0] * 2,
                my_position[1] + chosen_move.value[1] * 2
            )
            
            can_move_2 = self._can_move_n(my_position, chosen_move, self.internal_map, 2)
            
            # KIỂM TRA OVERSHOOT NGÃ RẼ:
            # Nếu path chỉ định rẽ ở ô tiếp theo (bước 1 = chosen, bước 2 != chosen),
            # ta chỉ được đi 1 bước. Tránh việc nhảy ngang qua ngã rẽ và lặp lại.
            if path and len(path) >= 2 and path[0] == chosen_move and path[1] != chosen_move:
                can_move_2 = False
                
            if can_move_2:
                # Nếu đường thẳng không vướng tường/không nảy ngã rẽ, ta HẾT SỨC tận dụng speed 2
                # Dù Ghost cách 1 ô, ta vẫn đi 2 ô (nếu được) để bắt được Ghost nếu nó lùi lại
                steps = 2
                        
        self.last_move = chosen_move
        return (chosen_move, steps)

    # ================================================================
    # BELIEF TRACKING
    # ================================================================

    def _init_belief(self, map_state):
        """Khởi tạo belief đều trên các ô đi được."""
        h, w = map_state.shape
        self.belief = np.zeros((h, w), dtype=np.float32)
        walkable = (map_state == 0)
        n = walkable.sum()
        if n > 0:
            self.belief[walkable] = 1.0 / n
        self.belief_initialized = True

    def _update_belief(self, map_state, my_pos, enemy_pos):
        """Cập nhật belief theo chu trình Predict → Update."""
        h, w = map_state.shape

        if not self.belief_initialized:
            self._init_belief(map_state)

        # --- PREDICT: Ghost có thể đi 1 bước bất kỳ ---
        new_belief = np.zeros_like(self.belief)
        dirs = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        for r in range(h):
            for c in range(w):
                if self.belief[r, c] < 1e-9:
                    continue
                p = self.belief[r, c]
                neighbors = []
                for mv in dirs:
                    dr, dc = mv.value
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] != 1:
                        neighbors.append((nr, nc))
                # Phân phối đều sang các ô lân cận + giữ nguyên
                total = len(neighbors) + 1
                new_belief[r, c] += p / total
                for nr, nc in neighbors:
                    new_belief[nr, nc] += p / total

        # Đảm bảo không ở ô tường
        walls = (map_state == 1)
        new_belief[walls] = 0.0

        # --- UPDATE: Xử lý quan sát ---
        if enemy_pos is not None:
            # Thấy Ghost: đặt belief tại vị trí đó = 1
            new_belief[:] = 0.0
            new_belief[enemy_pos] = 1.0
            self.last_known_enemy_pos = enemy_pos
            self.steps_since_seen = 0
            self.enemy_history.append(enemy_pos)
        else:
            self.steps_since_seen += 1
            # Project 2: Partial Observability with cross-shaped vision (max 5 cells)
            # map_state contains: 0 (empty), 1 (wall), 2 (pacman), 3 (ghost), -1 (unknown)
            # Only cells that we CAN SEE (map_state != -1) and are NOT the ghost mean the ghost is definitely NOT there.
            # We must zero out the probability of the ghost being in any clearly visible empty cell.
            known_empty = (map_state == 0) | (map_state == 2)
            new_belief[known_empty] = 0.0

        # Normalize
        total = new_belief.sum()
        if total > 1e-9:
            new_belief /= total
        else:
            self._init_belief(map_state)
            return

        self.belief = new_belief

    def _get_belief_target(self, my_pos):
        """Trả về vị trí có xác suất Ghost cao nhất làm target."""
        if self.belief is None:
            return self.last_known_enemy_pos
        idx = np.unravel_index(np.argmax(self.belief), self.belief.shape)
        return (int(idx[0]), int(idx[1]))

    # ================================================================
    # INTERCEPTION LOGIC
    # ================================================================

    def _predict_ghost_next(self, ghost_pos, my_pos):
        """
        Dự đoán vị trí Ghost sẽ đến dựa trên velocity.
        Ghost chạy ra xa Pacman nhất → predict 2-3 bước.
        """
        if len(self.enemy_history) >= 2:
            prev = self.enemy_history[-2]
            curr = self.enemy_history[-1]
            dr = curr[0] - prev[0]
            dc = curr[1] - prev[1]
            # Dự đoán ghost tiếp tục hướng này
            predicted = (curr[0] + dr * 2, curr[1] + dc * 2)
            # Clamp vào bản đồ
            h, w = self.internal_map.shape
            predicted = (
                max(0, min(h - 1, predicted[0])),
                max(0, min(w - 1, predicted[1]))
            )
            if self.internal_map[predicted] != 1:
                return predicted
        return ghost_pos

    def _get_interception_target(self, my_pos, ghost_pos):
        """
        Tìm điểm chặn đầu Ghost:
        - Dự đoán Ghost sẽ đi đến đâu
        - Tìm điểm trên đường thoát của Ghost mà Pacman có thể đến trước
        """
        if ghost_pos not in list(self.enemy_history)[-1:]:
            self.enemy_history.append(ghost_pos)

        # 0. Nếu Ghost ở quá gần (<= 5 ô), đuổi thẳng luôn để tránh cắt nhầm đường, HOẶC nếu Ghost ở ngay tầm mắt (tự tin cắt)
        if self._manhattan(my_pos, ghost_pos) <= 5:
            return ghost_pos

        # 1. Dự đoán vị trí Ghost sau 3-4 bước (Aggressive prediction)
        if len(self.enemy_history) >= 2:
            prev = self.enemy_history[-2]
            curr = self.enemy_history[-1]
            dr = curr[0] - prev[0]
            dc = curr[1] - prev[1]
            
            # Predict further ahead (3 steps) to cut corners
            predicted = (curr[0] + dr * 3, curr[1] + dc * 3)
            h, w = self.internal_map.shape
            predicted = (max(0, min(h - 1, predicted[0])), max(0, min(w - 1, predicted[1])))
            
            # If the predicted linear path hits a wall, find the nearest walkable cell
            if self.internal_map[predicted] == 1:
                predicted = self._find_nearest_walkable(predicted)
                
            dist_pac = self._bfs_distance(my_pos, predicted)
            dist_ghost = self._bfs_distance(ghost_pos, predicted)
            
            # If Pacman can reach the interception point before or at the same time as Ghost
            # Factoring in Pacman's speed=2 advantage (dist_pac / 1.5 roughly)
            if dist_pac is not None and dist_ghost is not None:
                effective_pac_dist = dist_pac / 1.5 if self.pacman_speed >= 2 else dist_pac
                if effective_pac_dist <= dist_ghost + 1:
                    return predicted

        # 2. Nếu không intercept được xa, tìm chokepoint gần
        chokepoint = self._find_chokepoint(my_pos, ghost_pos)
        if chokepoint:
            return chokepoint

        # 3. Fallback: đuổi thẳng
        return ghost_pos

    def _find_nearest_walkable(self, pos):
        """BFS search for nearest non-wall cell."""
        if self.internal_map is None: return pos
        h, w = self.internal_map.shape
        queue = deque([pos])
        visited = {pos}
        while queue:
            curr = queue.popleft()
            if self.internal_map[curr] != 1:
                return curr
            for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = mv.value
                nr, nc = curr[0] + dr, curr[1] + dc
                nxt = (nr, nc)
                if 0 <= nr < h and 0 <= nc < w and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return pos

    def _find_chokepoint(self, my_pos, ghost_pos):
        """
        Tìm ô hẹp (ít lối ra) trên đường giữa Ghost và Pacman.
        Ghost thường phải đi qua các ô hẹp này.
        """
        if self.internal_map is None:
            return None

        # BFS từ Ghost, ưu tiên ô ít lối ra (chokepoint)
        queue = deque([(ghost_pos, 0)])
        visited = {ghost_pos}
        best = None
        best_score = -1

        while queue:
            curr, depth = queue.popleft()
            if depth > 6:
                break

            # Tính số lối ra của ô này
            exits = self._count_exits(curr)
            # Ưu tiên ô hẹp (1-2 exits) gần Pacman hơn gần Ghost
            dist_to_pac = self._manhattan(curr, my_pos)
            dist_to_ghost = self._manhattan(curr, ghost_pos)

            if exits <= 2 and dist_to_pac < dist_to_ghost and curr != my_pos:
                score = dist_to_ghost - dist_to_pac
                if score > best_score:
                    best_score = score
                    best = curr

            for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = mv.value
                nr, nc = curr[0] + dr, curr[1] + dc
                nxt = (nr, nc)
                h, w = self.internal_map.shape
                if (0 <= nr < h and 0 <= nc < w
                        and self.internal_map[nr, nc] != 1
                        and nxt not in visited):
                    visited.add(nxt)
                    queue.append((nxt, depth + 1))

        return best

    def _count_exits(self, pos):
        """Đếm số lối ra từ một ô."""
        count = 0
        if self.internal_map is None:
            return 4
        h, w = self.internal_map.shape
        for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = mv.value
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < h and 0 <= nc < w and self.internal_map[nr, nc] != 1:
                count += 1
        return count

    # ================================================================
    # DQN
    # ================================================================

    def _get_ml_action(self, map_state, my_pos, enemy_pos):
        """Chạy DQN để lấy nước đi."""
        if self.model is None or not TORCH_AVAILABLE:
            return None
        try:
            input_map = self.internal_map.copy().astype(np.float32)
            input_map[input_map == -1] = 0
            if my_pos:
                input_map[my_pos] = 2
            if enemy_pos:
                input_map[enemy_pos] = 3

            state_tensor = torch.FloatTensor(input_map).unsqueeze(0).unsqueeze(0).to(self.device)

            all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            move_idx = all_moves.index(self.last_move) if self.last_move in all_moves else -1
            last_move_vec = torch.zeros(1, 4).to(self.device)
            if move_idx >= 0:
                last_move_vec[0, move_idx] = 1.0

            with torch.no_grad():
                q_values = self.model(state_tensor, last_move_vec)
                action_idx = torch.argmax(q_values).item()

            predicted_move = all_moves[action_idx]

            # Không quay đầu lại
            reverse_map = {Move.UP: Move.DOWN, Move.DOWN: Move.UP,
                           Move.LEFT: Move.RIGHT, Move.RIGHT: Move.LEFT}
            if predicted_move == reverse_map.get(self.last_move):
                return None

            if self._can_move_n(my_pos, predicted_move, self.internal_map, 1):
                return predicted_move
        except Exception:
            pass
        return None

    # ================================================================
    # HELPERS
    # ================================================================

    def _update_map_memory(self, map_state):
        if not self.map_initialized:
            self.internal_map = np.full_like(map_state, -1)
            # Initialize border walls if they exist in the first observation
            self.internal_map[map_state == 1] = 1
            self.map_initialized = True
        
        # Merge known cells (0, 1, 2, 3) into memory, overriding previous state
        visible_mask = map_state != -1
        self.internal_map[visible_mask] = map_state[visible_mask]
        
        # Restore Pacman/Ghost marks back to empty in history since they move
        self.internal_map[(self.internal_map == 2) | (self.internal_map == 3)] = 0

    def _greedy_toward(self, my_pos, target):
        """Di chuyển greedy theo hướng giảm Manhattan distance."""
        best_move = Move.STAY
        best_dist = self._manhattan(my_pos, target)
        for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = mv.value
            nxt = (my_pos[0] + dr, my_pos[1] + dc)
            if self._is_valid(nxt):
                d = self._manhattan(nxt, target)
                if d < best_dist:
                    best_dist = d
                    best_move = mv
        return best_move

    def _find_best_frontier(self, my_pos):
        """Tìm frontier (ô biên giữa known và unknown) tốt nhất."""
        if self.internal_map is None:
            return None
        h, w = self.internal_map.shape
        best = None
        best_score = -1
        for r in range(h):
            for c in range(w):
                if self.internal_map[r, c] != 0:
                    continue
                # Kiểm tra có ô -1 kề không
                has_unknown = False
                for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    dr, dc = mv.value
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and self.internal_map[nr, nc] == -1:
                        has_unknown = True
                        break
                if has_unknown:
                    dist = self._manhattan(my_pos, (r, c))
                    if dist == 0:
                        continue
                    score = 1.0 / dist
                    if score > best_score:
                        best_score = score
                        best = (r, c)
        return best

    def _random_valid_move(self, pos):
        moves = [mv for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
                 if self._can_move_n(pos, mv, self.internal_map, 1)]
        return random.choice(moves) if moves else Move.STAY

    def _can_move_n(self, pos, move, map_data, n):
        r, c = pos
        dr, dc = move.value
        for i in range(1, n + 1):
            nr, nc = r + dr * i, c + dc * i
            if map_data is None:
                return False
            if not (0 <= nr < map_data.shape[0] and 0 <= nc < map_data.shape[1]):
                return False
            if map_data[nr, nc] == 1:
                return False
        return True

    def _is_valid(self, pos):
        if self.internal_map is None:
            return False
        r, c = pos
        h, w = self.internal_map.shape
        return 0 <= r < h and 0 <= c < w and self.internal_map[r, c] != 1

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _bfs_distance(self, start, goal):
        """BFS để tính khoảng cách thực tế (vượt tường)."""
        if self.internal_map is None:
            return None
        if start == goal:
            return 0
        queue = deque([(start, 0)])
        visited = {start}
        h, w = self.internal_map.shape
        while queue:
            curr, dist = queue.popleft()
            for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = mv.value
                nr, nc = curr[0] + dr, curr[1] + dc
                nxt = (nr, nc)
                if (0 <= nr < h and 0 <= nc < w
                        and self.internal_map[nr, nc] != 1
                        and nxt not in visited):
                    if nxt == goal:
                        return dist + 1
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))
        return None

    def _get_neighbors(self, pos, map_state):
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            next_pos = (pos[0] + dr, pos[1] + dc)
            if (0 <= next_pos[0] < map_state.shape[0] and
                    0 <= next_pos[1] < map_state.shape[1] and
                    map_state[next_pos] != 1):
                neighbors.append((next_pos, move))
        return neighbors

    def astar(self, start, goal, map_state):
        """A* tìm đường ngắn nhất, ưu tiên đường thẳng để tận dụng speed=2."""
        if map_state is None or start == goal:
            return []
        
        # (f_score, counter, current_pos, path, last_move)
        frontier = [(0, 0, start, [], None)]
        # visited dictionary to store the best g_score for (position, last_move)
        visited = {}
        counter = 0
        
        while frontier:
            _, _, current, path, last_move = heappop(frontier)
            if current == goal:
                return path
                
            state_key = (current, last_move)
            g_score = len(path)
            
            if state_key in visited and visited[state_key] <= g_score:
                continue
            visited[state_key] = g_score
            
            for next_pos, move in self._get_neighbors(current, map_state):
                new_path = path + [move]
                # Tính chi phí: phạt RẤT NHẸ nếu rẽ hướng, không phạt nặng để tránh loop
                turn_penalty = 0
                if last_move is not None and move != last_move:
                    turn_penalty = 0.1
                
                g = len(new_path) + turn_penalty
                h = self._manhattan(next_pos, goal)
                counter += 1
                heappush(frontier, (g + h, counter, next_pos, new_path, move))
        return []


# ============================================================
# GHOST AGENT — Giữ nguyên (Minimax Iterative Deepening)
# ============================================================

class GhostAgent(BaseGhostAgent):
    """
    Ghost Ninja Pro:
    - Iterative Deepening Minimax với Pacman Speed 2
    - Evaluate: tầm nhìn, số lối thoát, khoảng cách
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_known_enemy_pos = None
        self.steps_since_seen = 0
        self.survival_target = None
        self.time_start = 0
        self.time_limit = 0.9
        
        # Memory map cho Ghost
        self.internal_map = None
        self.map_initialized = False

    def _update_map_memory(self, map_state):
        if not self.map_initialized:
            self.internal_map = np.full_like(map_state, -1)
            self.internal_map[map_state == 1] = 1
            self.map_initialized = True
        visible_mask = map_state != -1
        self.internal_map[visible_mask] = map_state[visible_mask]
        self.internal_map[(self.internal_map == 2) | (self.internal_map == 3)] = 0

    def step(self, map_state: np.ndarray, my_position: tuple,
             enemy_position: tuple, step_number: int) -> Move:
        self.time_start = time.time()
        self._update_map_memory(map_state)

        if enemy_position:
            self.last_known_enemy_pos = enemy_position
            self.steps_since_seen = 0
            self.survival_target = None
        else:
            self.steps_since_seen += 1

        target_enemy = enemy_position or self.last_known_enemy_pos

        # Tuần tra khi không thấy địch hoặc đã mất dấu lâu
        if target_enemy is None or self.steps_since_seen > 5:
            if not self.survival_target or my_position == self.survival_target:
                self.survival_target = self.find_nearest_intersection(my_position, self.internal_map)
            path = self.bfs_find_path(my_position, self.survival_target, self.internal_map)
            if path:
                return path[0]
            return self.get_random_valid_move(my_position, self.internal_map)

        # Iterative Deepening Minimax using internal memory map
        best_move = Move.STAY
        for depth in range(1, 20):
            try:
                if time.time() - self.time_start > self.time_limit:
                    break
                _, move = self.minimax(my_position, target_enemy, depth, True, self.internal_map)
                if time.time() - self.time_start < self.time_limit:
                    best_move = move
                else:
                    break
            except TimeoutError:
                break
        return best_move

    def minimax(self, my_pos, enemy_pos, depth, is_maximizing, map_state):
        if time.time() - self.time_start > self.time_limit:
            raise TimeoutError()
        if depth == 0 or my_pos == enemy_pos:
            return self.evaluate_ninja_state(my_pos, enemy_pos, map_state), Move.STAY

        valid_moves = self.get_valid_moves_with_pos(my_pos, map_state)
        if not valid_moves:
            return -1000, Move.STAY
        best_move = valid_moves[0][1]

        if is_maximizing:  # Ghost turn
            max_eval = -math.inf
            for next_pos, move in valid_moves:
                eval_score, _ = self.minimax(next_pos, enemy_pos, depth - 1, False, map_state)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
            return max_eval, best_move
        else:  # Pacman turn (speed 2)
            min_eval = math.inf
            pacman_reachable = self.get_pacman_reachable_positions(enemy_pos, 2, map_state)
            if my_pos in pacman_reachable:
                return -10000, None
            for next_enemy_pos in pacman_reachable:
                eval_score, _ = self.minimax(my_pos, next_enemy_pos, depth - 1, True, map_state)
                if eval_score < min_eval:
                    min_eval = eval_score
            return min_eval, None

    def get_pacman_reachable_positions(self, start_pos, speed, map_state):
        reachable = set()
        queue = deque([(start_pos, 0)])
        while queue:
            curr, steps = queue.popleft()
            if steps == speed:
                reachable.add(curr)
                continue
            moves = self.get_valid_moves_with_pos(curr, map_state)
            if not moves:
                reachable.add(curr)
            for next_pos, _ in moves:
                queue.append((next_pos, steps + 1))
        return reachable

    def evaluate_ninja_state(self, my_pos, enemy_pos, map_state):
        dist = abs(my_pos[0] - enemy_pos[0]) + abs(my_pos[1] - enemy_pos[1])
        if dist == 0:
            return -10000
        score = dist * 10
        is_visible = self.check_line_of_sight(my_pos, enemy_pos, map_state)
        if not is_visible:
            score += 500
            if my_pos[0] != enemy_pos[0] and my_pos[1] != enemy_pos[1]:
                score += 50
        else:
            score -= 200
        escapes = len(self.get_valid_moves_with_pos(my_pos, map_state))
        if escapes <= 1:
            score -= 300
        return score

    def check_line_of_sight(self, pos1, pos2, map_state):
        r1, c1 = pos1
        r2, c2 = pos2
        if r1 != r2 and c1 != c2:
            return False
        if r1 == r2:
            for c in range(min(c1, c2) + 1, max(c1, c2)):
                if map_state[r1, c] == 1:
                    return False
            return True
        for r in range(min(r1, r2) + 1, max(r1, r2)):
            if map_state[r, c1] == 1:
                return False
        return True

    def find_nearest_intersection(self, start_pos, map_state):
        queue = deque([start_pos])
        visited = {start_pos}
        while queue:
            curr = queue.popleft()
            moves = self.get_valid_moves_with_pos(curr, map_state)
            if len(moves) >= 3 and curr != start_pos:
                return curr
            for next_pos, _ in moves:
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append(next_pos)
        return start_pos

    def bfs_find_path(self, start, end, map_state):
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            curr, path = queue.popleft()
            if curr == end:
                return path
            for next_pos, move in self.get_valid_moves_with_pos(curr, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))
        return []

    def get_valid_moves_with_pos(self, pos, map_state):
        valid = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            nr, nc = pos[0] + dr, pos[1] + dc
            if (0 <= nr < map_state.shape[0] and
                    0 <= nc < map_state.shape[1] and
                    map_state[nr, nc] != 1):
                valid.append(((nr, nc), move))
        return valid

    def get_random_valid_move(self, pos, map_state):
        moves = self.get_valid_moves_with_pos(pos, map_state)
        if moves:
            return random.choice(moves)[1]
        return Move.STAY