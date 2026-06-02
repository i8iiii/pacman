
"""
Example student submission showing the required interface.

Students should implement their own PacmanAgent and/or GhostAgent
following this template.
"""
from collections import deque
from inspect import currentframe
from re import X
import sys
from pathlib import Path
import time
import pickle 

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import random

import pacmanAlgorithm

class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Cornering Master Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        
        self.capture_threshold = 1 
        if '--capture-distance' in sys.argv:
            try:
                idx = sys.argv.index('--capture-distance')
                self.capture_threshold = int(sys.argv[idx + 1])
            except (ValueError, IndexError):
                pass
                
        self.algorithm = "ASTAR" 
        self.last_ghost_pos = None # Ghi nhớ vị trí cũ của Ghost để tính quán tính

        self.locked_target = None
        self.lock_counter = 0

    def _get_forward_choke_point(self, ghost_pos, ghost_dir, map_state):
        """Hàm quét thẳng về phía trước để tìm ngã tư hoặc góc chết để chặn đầu"""
        curr_pos = ghost_pos
        # Quét tối đa 7 ô để tránh nhìn quá xa (Ghost có thể rẽ trước khi tới)
        for _ in range(7): 
            next_pos = (curr_pos[0] + ghost_dir[0], curr_pos[1] + ghost_dir[1])
            
            # Nếu đụng tường hoặc ra khỏi bản đồ, chặn ngay tại vị trí hiện tại của Ghost
            if not self._is_valid_position(next_pos, map_state):
                return curr_pos 
            
            # Đếm số đường đi tại ô tiếp theo
            valid_neighbors = 0
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                n = (next_pos[0] + move.value[0], next_pos[1] + move.value[1])
                if self._is_valid_position(n, map_state):
                    valid_neighbors += 1
            
            # Nếu có > 2 đường đi (Ngã 3, ngã 4)
            if valid_neighbors > 2:
                return next_pos
                
            curr_pos = next_pos
            
        return curr_pos # Nếu đường thẳng quá dài, chặn ở ô thứ 7

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        
        try:
            path = []
            # Tính hướng di chuyển của Ghost
            ghost_dir = None
            if self.last_ghost_pos:
                ghost_dir = (enemy_position[0] - self.last_ghost_pos[0], 
                             enemy_position[1] - self.last_ghost_pos[1])
            self.last_ghost_pos = enemy_position

            target_pos = enemy_position 

            # Dồn góc nếu Ghost đã đi được một khoảng nhất định và có hướng di chuyển rõ ràng
            if step_number > 10 and ghost_dir is not None:
                # Khóa mục tiêu
                if self.lock_counter > 0 and self.locked_target is not None:
                    target_pos = self.locked_target
                    self.lock_counter -= 1
                    
                    # Nếu Pacman đã chạy tới ngay điểm chặn rồi thì mở khóa sớm
                    if my_position == self.locked_target:
                        self.lock_counter = 0
                        self.locked_target = None
                        target_pos = enemy_position
                
                # Quét tìm điểm chặn mới
                else:
                    choke_point = self._get_forward_choke_point(enemy_position, ghost_dir, map_state)
                    target_pos = choke_point
                    
                    # Nếu tìm ra được một ngã tư/góc chết hợp lý (khác chỗ Ghost đang đứng)
                    if choke_point != enemy_position:
                        self.locked_target = choke_point
                        # Khóa mục tiêu trong 3 lượt 
                        self.lock_counter = 3  
                
                # Tìm đường tới điểm dồn góc
                path = pacmanAlgorithm.astar(my_position, target_pos, map_state)
                
                # Nếu không tìm được đường tới điểm chặn (hoặc rủi ro), quay lại đuổi thẳng mặt và xóa khóa
                if not path and target_pos != enemy_position:
                    target_pos = enemy_position
                    self.lock_counter = 0
                    self.locked_target = None
            else:
                # Thuần A*
                self.lock_counter = 0
                self.locked_target = None

           
            if target_pos == enemy_position:
                best_moves = []
                min_path_len = float('inf')
                paths_dict = {}

                for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    dr, dc = move.value
                    next_pos = (my_position[0] + dr, my_position[1] + dc)
                    
                    if self._is_valid_position(next_pos, map_state):
                        if next_pos == enemy_position:
                            min_path_len = 0
                            best_moves = [move]
                            paths_dict[move] = []
                            break
                            
                        sub_path = pacmanAlgorithm.astar(next_pos, enemy_position, map_state)
                        if not sub_path: continue
                            
                        path_len = len(sub_path)
                        paths_dict[move] = sub_path
                        
                        if path_len < min_path_len:
                            min_path_len = path_len
                            best_moves = [move]
                        elif path_len == min_path_len:
                            best_moves.append(move) 
                
                # Tie-breaker: Chọn ngã rẽ trùng với hướng Ghost vừa đi
                chosen_move = None
                if len(best_moves) > 1 and ghost_dir is not None:
                    for m in best_moves:
                        if m.value == ghost_dir:
                            chosen_move = m
                            break
                            
                if chosen_move is None and best_moves:
                    chosen_move = best_moves[0]
                    
                if chosen_move:
                    path = [chosen_move] + paths_dict[chosen_move]

            # Di chuyển với tốc độ tối đa có thể
            if path:
                best_move = path[0]
                desired_steps = 1
                for i in range(1, len(path)):
                    if path[i] == best_move:
                        desired_steps += 1
                    else:
                        break
                        
                if desired_steps == len(path) and desired_steps < self.pacman_speed:
                    desired_steps = self.pacman_speed
                    
                steps = self._max_valid_steps(my_position, best_move, map_state, desired_steps)
                if steps > 0:
                    return (best_move, steps)
                    
            return (Move.STAY, 1)

        except Exception as e:
            # Fallback an toàn nếu lỗi
            fallback_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            for move in fallback_moves:
                delta_row, delta_col = move.value
                next_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
                if self._is_valid_position(next_pos, map_state):
                    return (move, 1)
            return (Move.STAY, 1)
        
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, desired_steps: int) -> int:
        steps = 0
        max_steps = min(self.pacman_speed, max(1, desired_steps))
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps

    def _desired_steps(self, move: Move, row_diff: int, col_diff: int) -> int:
        if move in (Move.UP, Move.DOWN):
            return abs(row_diff)
        if move in (Move.LEFT, Move.RIGHT):
            return abs(col_diff)
        return 1

class GhostAgent(BaseGhostAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Smart Minimax Ghost"
        self.last_move = None # lưu vị trí đã đi để phạt nếu quay đầu

    def _smart_pacman_move(self, pacman_pos, ghost_pos, map_state):
        """Pacman di chuyển thông minh: ưu tiên bước nào đến gần ghost nhất"""
        dist_map = self._compute_distance_map(ghost_pos, map_state)
    
        best_pos = pacman_pos
        best_dist = dist_map.get(pacman_pos, float("inf"))
    
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dx, dy = move.value
            new_pos = (pacman_pos[0] + dx, pacman_pos[1] + dy)
            if self._is_valid_position(new_pos, map_state):
                d = dist_map.get(new_pos, float("inf"))
                if d < best_dist:      # pacman chọn bước rút ngắn khoảng cách đến ghost
                    best_dist = d
                    best_pos = new_pos
    
        return best_pos

    def _smart_ghost_escape(self, ghost_pos, pacman_pos, map_state):
        dist_map = self._compute_distance_map(pacman_pos, map_state)

        best_pos = None
        best_dist = -1
        valid_moves = []

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dx, dy = move.value
            new_pos = (ghost_pos[0] + dx, ghost_pos[1] + dy)

            if not self._is_valid_position(new_pos, map_state):
                continue

            valid_moves.append(new_pos)

            if self._is_dead(new_pos, pacman_pos):
                continue

            d = dist_map.get(new_pos, -1)

            if d > best_dist:
                best_dist = d
                best_pos = new_pos

        if best_pos is not None:
            return best_pos

        if valid_moves:
            return random.choice(valid_moves)

        return ghost_pos

    def monte_carlo(self, ghost_pos, pacman_pos, map_state, simulations=10):

        total_score = 0

        for _ in range(simulations):

            sim_ghost = ghost_pos
            sim_pacman = pacman_pos

            survived = True

            for step in range(5):  # số bước nhìn trước

                #  Ghost chạy 
                sim_ghost = self._smart_ghost_escape(sim_ghost, sim_pacman, map_state)

                # chết nếu đứng cạnh
                if self._is_dead(sim_ghost, sim_pacman):
                    total_score += -1000 + step
                    survived = False
                    break

                #  Pacman đi 2 bước 
                for _ in range(2):

                    if random.random() < 0.8:
                        sim_pacman = self._smart_pacman_move(sim_pacman, sim_ghost, map_state)
                    else:
                        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
                        valid = []

                        for m in moves:
                            dx, dy = m.value
                            np = (sim_pacman[0] + dx, sim_pacman[1] + dy)
                            if self._is_valid_position(np, map_state):
                                valid.append(np)

                        if valid:
                            sim_pacman = random.choice(valid)

                    #  check chết sau mỗi bước
                    if self._is_dead(sim_ghost, sim_pacman):
                        total_score += -1000 + step
                        survived = False
                        break

                if not survived:
                    break

            # nếu sống sót 
            if survived:
                dist_map = self._compute_distance_map(sim_pacman, map_state)
                dist = dist_map.get(sim_ghost, 0)

                # BONUS NGÃ RẼ
                escape_routes = self._count_valid_moves(sim_ghost, map_state)

                total_score += dist + 0.3 * escape_routes

        return total_score / simulations

    #check bị bắt
    def _is_dead(self, ghost_pos, pacman_pos):
        return abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1]) <= 1


    def minimax(self, ghost_pos, pacman_pos, map_state, depth, ghost_turn):
        # Base case: Hết độ sâu dự đoán
        if depth == 0:
            # Thay vì dùng Manhattan, ta gọi BFS nhanh từ pacman_pos hiện tại
            # để biết khoảng cách THỰC TẾ đến ghost_pos
            dist_map = self._compute_distance_map(pacman_pos, map_state)
            return dist_map.get(ghost_pos, -1) # Trả về số bước đi thực tế

        if ghost_turn:
            best = -float("inf") # Lượt Ghost: Muốn khoảng cách max
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dx, dy = move.value
                new_pos = (ghost_pos[0] + dx, ghost_pos[1] + dy)
                if self._is_valid_position(new_pos, map_state):
                    score = self.minimax(new_pos, pacman_pos, map_state, depth-1, False)
                    best = max(best, score)
            # Nếu bị kẹt không có đường đi hợp lệ
            return best if best != -float("inf") else -1000

        else:
            best = float("inf") # Lượt Pacman: Muốn khoảng cách min
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dx, dy = move.value
                new_pos = (pacman_pos[0] + dx, pacman_pos[1] + dy)
                if self._is_valid_position(new_pos, map_state):
                    score = self.minimax(ghost_pos, new_pos, map_state, depth-1, True)
                    best = min(best, score)
            return best if best != float("inf") else 1000


    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:

        best_move = Move.STAY
        best_score = -float("inf")


        opposite = {
            Move.UP: Move.DOWN,
            Move.DOWN: Move.UP,
            Move.LEFT: Move.RIGHT,
            Move.RIGHT: Move.LEFT,
        }

        valid_move_count = self._count_valid_moves(my_position, map_state)

        # Duyệt 4 hướng di chuyển hiện tại của Ghost
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dx, dy = move.value
            new_pos = (my_position[0] + dx, my_position[1] + dy)

            # fix lại điều kiện
            if not self._is_valid_position(new_pos, map_state):
                continue

            # tránh chết ngay
            if self._is_dead(new_pos, enemy_position):
                continue

            if self._is_valid_position(new_pos, map_state):
                # Gọi Minimax với độ sâu 2 (1 lượt Ghost đi, 1 lượt Pacman đi)
                score = self.minimax(new_pos, enemy_position, map_state, 4, False)
                mc_score = self.monte_carlo(new_pos, enemy_position, map_state, simulations=5) 
                score = score + 0.2 * mc_score
                #  BONUS NGÃ RẼ
                next_moves = self._count_valid_moves(new_pos, map_state)

                # nếu là ngã cụt (1 đường) → phạt
                if next_moves <= 1:
                    score -= 7

                # nếu là ngã rẽ (>=3 hướng) → thưởng
                elif next_moves >= 3:
                    score += 5
                
                if score > best_score:
                    best_score = score
                    best_move = move

        # lưu lại hướng đi
        self.last_move = best_move
        return best_move

    def _count_valid_moves(self, pos, map_state):
        count = 0
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dx, dy = move.value
            new_pos = (pos[0] + dx, pos[1] + dy)
        if self._is_valid_position(new_pos, map_state):
            count += 1
        return count             

    def _compute_distance_map(self, start_pos: tuple, map_state: np.ndarray) -> dict:
        """Dùng BFS để tính khoảng cách từ một điểm đến toàn bộ các ô khác."""
        queue = deque([(start_pos, 0)]) 
        visited = {start_pos: 0}
        height, width = map_state.shape 
        
        while queue:
            (curr_row, curr_col), dist = queue.popleft()
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nr, nc = curr_row + dr, curr_col + dc
                
                if (0 <= nr < height and 0 <= nc < width and 
                    map_state[nr, nc] == 0 and (nr, nc) not in visited):
                    
                    visited[(nr, nc)] = dist + 1
                    queue.append(((nr, nc), dist + 1))
        return visited
        
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        h, w = map_state.shape
        return 0 <= row < h and 0 <= col < w and map_state[row, col] == 0
