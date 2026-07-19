import sys
from pathlib import Path
from collections import deque
import random
import numpy as np

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


class GameState:
    # Vị trí của pacman/ghost dùng cho minimax
    def __init__(self, pacman_pos, ghost_pos):
        self.pacman_pos = pacman_pos
        self.ghost_pos = ghost_pos


class PacmanAgent(BasePacmanAgent):
    # Pacman đổi algorithm dựa vào khoảng cách (tính bằng BFS)
    # - > 7 dist: BFS
    # - <= 7 dist: minimax + alpha-beta pruning
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Hybrid Minimax Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        
        
        self.max_depth = 6 # Độ sâu minimax
        self.alg_change = 7 # Tính khoảng cách để chuyển đổi alg từ BFS thành minimax
        
        self.dist_cache = {} # Lưu trữ khoảng cách
        self.last_known_enemy_pos = None

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        
        if step_number == 1:
            self.dist_cache.clear()

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        target = enemy_position or self.last_known_enemy_pos
        
        if target is None:
            moves = self._get_pacman_moves(my_position, map_state)
            return random.choice(moves) if moves else (Move.STAY, 1)

        # Khoảng cách giữa Pacman và ghost
        maze_dist = self._get_maze_distance(my_position, target, map_state)
        
        if maze_dist > self.alg_change:
            # dist > 7 => BFS
            return self._BFS(my_position, target, map_state)
        else:
            # dist <= 7 => minimax
            best_score = float('-inf')
            best_action = (Move.STAY, 1)
            alpha = float('-inf')
            beta = float('inf')

            legal_actions = self._get_pacman_moves(my_position, map_state)

            for action in legal_actions:
                next_pos = self._simulate_pacman_move(my_position, action)
                next_state = GameState(next_pos, target)
                
                score = self._Minimax(next_state, 1, alpha, beta, False, map_state)
                
                if score > best_score:
                    best_score = score
                    best_action = action
                    
                alpha = max(alpha, best_score)

            return best_action

    def _BFS(self, start: tuple, goal: tuple, map_state: np.ndarray):
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            if current == goal:
                if not path:
                    return (Move.STAY, 1)
                
                first_step = path[0]
                dr = first_step[0] - start[0]
                dc = first_step[1] - start[1]
                
                move = Move.STAY
                for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    if m.value == (dr, dc):
                        move = m
                        break
                        
                steps = 1
                curr = first_step
                for i in range(1, self.pacman_speed):
                    if i < len(path):
                        next_path_pos = path[i]
                        
                        # Check có thể đi được 2 bước trong 1 lượt hay không 
                        if next_path_pos[0] - curr[0] == dr and next_path_pos[1] - curr[1] == dc:
                            steps += 1
                            curr = next_path_pos
                        else:
                            break # không được
                    else:
                        break 
                        
                return (move, steps)
                
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nxt = (current[0] + dr, current[1] + dc)
                if self._is_valid_position(nxt, map_state) and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
                    
        moves = self._get_pacman_moves(start, map_state)
        return moves[0] if moves else (Move.STAY, 1)

    def _Minimax(self, state: GameState, depth: int, alpha: float, beta: float, is_pacman: bool, map_state: np.ndarray) -> float:
        # TH: pacman bắt được ghost, tính số điểm bằng cách lấy số cực kỳ lớn - chiều sâu (số bước đi để đến)
        if is_pacman:
            manhattan_dist = abs(state.pacman_pos[0] - state.ghost_pos[0]) + abs(state.pacman_pos[1] - state.ghost_pos[1])
            if manhattan_dist < 2:
                return 99999 - depth 

        # TH: đến chiều sâu của cây mà chưa bắt được ghost
        if depth == self.max_depth:
            maze_distance = self._get_maze_distance(state.pacman_pos, state.ghost_pos, map_state)
            ghost_mobility = len(self._get_ghost_moves(state.ghost_pos, map_state))
            # Cách tính điểm : - khoảng cách - (lối thoát của ma * 2), hình thức phạt nặng hơn nếu có nhiều lối thoát hơn 
            return -maze_distance - (ghost_mobility * 2)

        # Lượt pacman (Max)
        if is_pacman:
            max_eval = float('-inf')
            for action in self._get_pacman_moves(state.pacman_pos, map_state):
                next_pos = self._simulate_pacman_move(state.pacman_pos, action)
                next_state = GameState(next_pos, state.ghost_pos)
                
                eval_score = self._Minimax(next_state, depth + 1, alpha, beta, False, map_state)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    break
            return max_eval
            
        # Lượt ghost (Min)
        else:
            min_eval = float('inf')
            for move in self._get_ghost_moves(state.ghost_pos, map_state):
                next_pos = self._simulate_ghost_move(state.ghost_pos, move)
                next_state = GameState(state.pacman_pos, next_pos)
                
                eval_score = self._Minimax(next_state, depth + 1, alpha, beta, True, map_state)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    break 
            return min_eval

    # Ultilities (distance, legal moves)
    def _get_maze_distance(self, start: tuple, goal: tuple, map_state: np.ndarray) -> int:
        if start == goal: return 0
            
        cache_key = (start, goal)
        if cache_key in self.dist_cache:
            return self.dist_cache[cache_key]
            
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            current_pos, dist = queue.popleft()
            if current_pos == goal:
                self.dist_cache[cache_key] = dist
                self.dist_cache[(goal, start)] = dist
                return dist
                
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nr, nc = current_pos[0] + dr, current_pos[1] + dc
                if self._is_valid_position((nr, nc), map_state) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))
                    
        return 999

    def _get_pacman_moves(self, pos: tuple, map_state: np.ndarray) -> list:
        moves = [(Move.STAY, 1)]
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            current = pos
            for step in range(1, self.pacman_speed + 1):
                nr, nc = current[0] + dr, current[1] + dc
                if self._is_valid_position((nr, nc), map_state):
                    moves.append((move, step))
                    current = (nr, nc)
                else:
                    break
        return moves

    def _get_ghost_moves(self, pos: tuple, map_state: np.ndarray) -> list:
        moves = [Move.STAY]
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_position(self._simulate_ghost_move(pos, move), map_state):
                moves.append(move)
        return moves

    def _simulate_pacman_move(self, pos: tuple, action: tuple) -> tuple:
        move, steps = action
        dr, dc = move.value
        return (pos[0] + (dr * steps), pos[1] + (dc * steps))

    def _simulate_ghost_move(self, pos: tuple, move: Move) -> tuple:
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width: return False
        return map_state[row, col] == 0


class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "BFS Ghost"
        self.last_known_enemy_pos = None
        self.dead_end_cells = set()
        self.map_initialized = False
        self.current_target = None
        self.current_path = []

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        if not self.map_initialized:
            self._init_map_data(map_state)
            self.map_initialized = True

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        threat = enemy_position or self.last_known_enemy_pos
        if threat is None:
            return self._random_safe_move(my_position, map_state)

        pacman_distances = self._get_pacman_bfs_distance(threat, map_state)
        ghost_distances = self._get_bfs_distance(my_position, map_state)
        dist_to_pacman = pacman_distances.get(my_position, 0)

        # TH1: Nếu khoảng cách gần với pacman và đang cùng hàng (urgent)
        # -> Sử dụng hàm break_line để tìm hướng rẽ khác để cắt đuôi (hủy target hiện tại)
        if dist_to_pacman <= 4 and self._check_pacman_inline(my_position, threat, map_state):
            break_move = self._break_line(my_position, threat, pacman_distances, map_state)
            if break_move:
                self.current_target = None
                self.current_path = []
                return break_move

        # TH2: Kiểm tra độ an toàn path hiện tại đang đi
        # Hủy path nếu tới đích hoặc bị pacman đón trước/pacman tới ô đó trước)
        if self.current_target and self.current_path:
            next_pos = self.current_path[0]
            target_pacman_distance = pacman_distances.get(self.current_target, -1)
            target_ghost_distance = ghost_distances.get(self.current_target, float('inf'))
            next_pacman_distance = pacman_distances.get(next_pos, 0)
        
            if (my_position == self.current_target or target_pacman_distance <= target_ghost_distance + 1 or next_pacman_distance <= 2):
                self.current_target = None
                self.current_path = []

        # TH3: Áp dụng BFS tìm đường khi chưa có mục tiêu
        if not self.current_target:
            self.current_target = self._get_global_target(my_position, pacman_distances, ghost_distances, map_state)
            if self.current_target:
                full_path = self._ghost_bfs_path(my_position, self.current_target, map_state)
                if full_path and len(full_path) > 1:
                    self.current_path = full_path[1:] # Bỏ qua my_position

        if self.current_path:
            next_pos = self.current_path.pop(0)
            return self._translate_move(my_position, next_pos)

        # Nếu trên đều k trả kết quả thì cố đi đường có điểm cao nhất
        return self._best_evasion_move(my_position, pacman_distances, map_state)
    
    # Tìm trước các ngõ cụt trên map và loại từ đầu
    def _init_map_data(self, map_state: np.ndarray):
        rows, cols = map_state.shape
        for r in range(rows):
            for c in range(cols):
                start = (r, c)
                if not self._is_valid_position(start, map_state):
                    continue
                
                neighbors = self._get_neighbors(start, map_state)
                if len(neighbors) != 1:
                    continue

                prev, cur = None, start
                while True:
                    self.dead_end_cells.add(cur)
                    valid_next = [n for n in self._get_neighbors(cur, map_state) if n != prev]
                    
                    if not valid_next:
                        break
                        
                    nxt = valid_next[0]
                    if len(self._get_neighbors(nxt, map_state)) >= 3:
                        break
                        
                    prev, cur = cur, nxt

    # Tính toán điểm an toàn dựa trên 
    def _get_global_target(self, my_pos: tuple, pacman_distances: dict, ghost_distances: dict, map_state: np.ndarray):
        best_targets = []
        max_score = -float('inf')
        
        rows, cols = map_state.shape
        for r in range(rows):
            for c in range(cols):
                cell = (r, c)
                if cell in ghost_distances and cell in pacman_distances and cell not in self.dead_end_cells:
                    dist_p = pacman_distances[cell]
                    dist_g = ghost_distances[cell]
                    
                    if dist_p > dist_g + 2: # Kiểm tra có đến trước pacman trong 2 step không
                        exits = len(self._get_neighbors(cell, map_state))
                        
                        score = (dist_p * 10) - (dist_g * 2) + (exits * 20)
                        
                        if score > max_score:
                            max_score = score
                            best_targets = [cell]
                        elif score == max_score:
                            best_targets.append(cell)
                            
        if best_targets:
            return random.choice(best_targets)
        return None

    # Lấy danh sách khoảng cách của pacman tương ứng tọa độ của từng điểm trên bản đồ
    def _get_pacman_bfs_distance(self, start_pos: tuple, map_state: np.ndarray) -> dict:
        distances = {start_pos: 0}
        queue = deque([start_pos])

        while queue:
            current = queue.popleft()
            current_distance = distances[current]
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                n1 = (current[0] + dr, current[1] + dc)
                if self._is_valid_position(n1, map_state):
                    if n1 not in distances or distances[n1] > current_distance + 1:
                        distances[n1] = current_distance + 1
                        queue.append(n1)
                    
                    # Tính ghost di chuyển 2 cells per step
                    n2 = (current[0] + 2 * dr, current[1] + 2 * dc)
                    if self._is_valid_position(n2, map_state):
                        if n2 not in distances or distances[n2] > current_distance + 1:
                            distances[n2] = current_distance + 1
                            queue.append(n2)
        return distances

    # Lấy dánh sách khoảng cách của ghost tương ứng tọa độ từng điểm trên bản đồ
    def _get_bfs_distance(self, start_pos: tuple, map_state: np.ndarray) -> dict:
        distances = {start_pos: 0}
        queue = deque([start_pos])

        while queue:
            current = queue.popleft()
            current_distance = distances[current]
            
            for next_pos in self._get_neighbors(current, map_state):
                if next_pos not in distances:
                    queue.append(next_pos)
                    distances[next_pos] = current_distance + 1

        return distances

    # Tìm path với BFS thuần
    def _ghost_bfs_path(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            current, path = queue.popleft()
            if current == goal:
                return path

            for nxt in self._get_neighbors(current, map_state):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
        return []

    def _best_evasion_move(self, my_pos: tuple, pacman_distances: dict, map_state: np.ndarray) -> Move:
        best_moves = []
        best_score = -float('inf')

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            next_pos = (my_pos[0] + dr, my_pos[1] + dc)

            if not self._is_valid_position(next_pos, map_state):
                continue

            dist_from_pacman = pacman_distances.get(next_pos, -1)
            exits = len(self._get_neighbors(next_pos, map_state))
            
            score = dist_from_pacman * 10 + exits * 2
            if next_pos in self.dead_end_cells:
                score -= 1000

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        if best_moves:
            return random.choice(best_moves)
        return Move.STAY

    # Tìm ngã rẽ để cắt đuôi
    # Tính điểm dựa trên khoảng cách pacman đến các điểm rẽ sau đó
    def _break_line(self, my_pos: tuple, pacman: tuple, pacman_distances: dict, map_state: np.ndarray):
        best_moves = []
        best_score = -float('inf')
        r_pacman, c_pacman = pacman

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            next_pos = (my_pos[0] + dr, my_pos[1] + dc)

            if not self._is_valid_position(next_pos, map_state):
                continue

            if next_pos in self.dead_end_cells:
                continue

            if (r_pacman == my_pos[0] and dr != 0) or (c_pacman == my_pos[1] and dc != 0):
                score = pacman_distances.get(next_pos, 0)
                if score > best_score:
                    best_score = score
                    best_moves = [move]
                elif score == best_score:
                    best_moves.append(move)

        if best_moves:
            return random.choice(best_moves)
        return None

    def _random_safe_move(self, my_pos: tuple, map_state: np.ndarray) -> Move:
        valid_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            next_pos = (my_pos[0] + dr, my_pos[1] + dc)
            
            if self._is_valid_position(next_pos, map_state):
                if next_pos not in self.dead_end_cells:
                    valid_moves.append(move)
                    
        if valid_moves:
            return random.choice(valid_moves)
            
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            next_pos = (my_pos[0] + dr, my_pos[1] + dc)
            if self._is_valid_position(next_pos, map_state):
                return move
                
        return Move.STAY

    # Cac ham helper
    def _check_pacman_inline(self, pos1: tuple, pos2: tuple, map_state: np.ndarray) -> bool:
        r1, c1 = pos1
        r2, c2 = pos2

        # Cùng hàng
        if r1 == r2:
            if c2 > c1:
                step_c = 1
            else:
                step_c = -1
            for c in range(c1 + step_c, c2, step_c):
                if map_state[r1, c] == 1:
                    return False
            return True

        # Cùng cột
        if c1 == c2:
            if r2 > r1:
                step_r = 1
            else:
                step_r = -1
            for r in range(r1 + step_r, r2, step_r):
                if map_state[r, c1] == 1:
                    return False
            return True

        return False
    
    def _translate_move(self, current: tuple, next_pos: tuple) -> Move:
        dr = next_pos[0] - current[0]
        dc = next_pos[1] - current[1]
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if move.value == (dr, dc):
                return move
        return Move.STAY

    def _get_neighbors(self, pos: tuple, map_state: np.ndarray) -> list:
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (pos[0] + dr, pos[1] + dc)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append(next_pos)
        return neighbors

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0