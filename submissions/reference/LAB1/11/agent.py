import sys
from pathlib import Path
import random
import time 
import heapq #For priority queue
from collections import deque #For BFS 
import numpy as np

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move



class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "YoruNiKakeru Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.neighbor_cache = {}

    def _heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def astar_path(self, start, goal, map_state):
        if start == goal:
            return [start]

        open_heap = [(self._heuristic(start, goal), 0, start)]
        came_from = {}
        g_score = {start: 0}
        visited = set()

        while open_heap:
            _, g, current = heapq.heappop(open_heap)

            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor in self._get_neighbors(current, map_state):
                tentative_g = g + 1
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    heapq.heappush(
                        open_heap,
                        (tentative_g + self._heuristic(neighbor, goal), tentative_g, neighbor),
                    )

        return None  # không có đường đi

    def _delta_to_move(self, dr, dc):
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if move.value == (dr, dc):
                return move
        return Move.STAY

    def next_move(self, my_pos, goal, map_state):
        path = self.astar_path(my_pos, goal, map_state)

        if not path or len(path) < 2:
            return (Move.STAY, 1)

        dr = path[1][0] - my_pos[0]
        dc = path[1][1] - my_pos[1]
        move = self._delta_to_move(dr, dc)

        # đi tiếp cùng hướng nếu path vẫn thẳng, tối đa pacman_speed ô
        steps = 1
        idx = 1
        while steps < self.pacman_speed and idx + 1 < len(path):
            ndr = path[idx + 1][0] - path[idx][0]
            ndc = path[idx + 1][1] - path[idx][1]
            if (ndr, ndc) != (dr, dc):
                break
            steps += 1
            idx += 1

        return (move, steps)

    #Find all possible cells nearby
    def _get_neighbors(self, pos, map_state):
        if pos in self.neighbor_cache:
            return self.neighbor_cache[pos]

        r, c = pos
        h, w = map_state.shape
        out = []

        if r > 0 and map_state[r-1, c] == 0:
            out.append((r-1, c))
        if r + 1 < h and map_state[r+1, c] == 0:
            out.append((r+1, c))
        if c > 0 and map_state[r, c-1] == 0:
            out.append((r, c-1))
        if c + 1 < w and map_state[r, c+1] == 0:
            out.append((r, c+1))

        self.neighbor_cache[pos] = out
        return out

    def explore(self, my_pos, map_state):
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(moves)
        for move in moves:
            steps = self._max_valid_steps(my_pos, move, map_state, self.pacman_speed)
            if steps > 0:
                return (move, steps)
        return (Move.STAY, 1)

    def step(self, map_state, my_position, enemy_position, step_number):
        if enemy_position is None:
            return self.explore(my_position, map_state)
        return self.next_move(my_position, enemy_position, map_state)

    def _is_valid_position(self, pos, map_state):
        row, col = pos
        h, w = map_state.shape
        if row < 0 or row >= h or col < 0 or col >= w:
            return False
        return map_state[row, col] == 0

    def _max_valid_steps(self, pos, move, map_state, desired_steps):
        steps = 0
        max_steps = min(self.pacman_speed, max(1, desired_steps))
        cur = pos
        for _ in range(max_steps):
            dr, dc = move.value
            npos = (cur[0] + dr, cur[1] + dc)
            if not self._is_valid_position(npos, map_state):
                break
            steps += 1
            cur = npos
        return steps


class TimeOutException(Exception):
    pass

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Hybrid Phantom V_MAX"
        self._history = []
        self._history_ban = 4  
        self.last_known_enemy_pos = None 
        
        self.pacman_speed = max(2, int(kwargs.get("pacman_speed", 2))) 
        
        self.map_analyzed = False
        self.valid_tiles = {}
        self.dead_ends = set()
        self.junctions = set()
        self.map_BFS_cache = {}
        self.dist_to_junction = {} 

    def _analyze_map(self, map_state):
        if self.map_analyzed:
            return
        
        height, width = map_state.shape
        for r in range(height):
            for c in range(width):
                if map_state[r, c] in [0, -1]:
                    self.valid_tiles[(r, c)] = True
                    if self._cell_exits((r, c), map_state) >= 3:
                        self.junctions.add((r, c))
                        
        graph = {pos: [] for pos in self.valid_tiles}
        for pos in self.valid_tiles:
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                next_pos = (pos[0] + move.value[0], pos[1] + move.value[1])
                if next_pos in self.valid_tiles:
                    graph[pos].append(next_pos)
                    
        initial_dead_ends = [pos for pos, neighbors in graph.items() if len(neighbors) <= 1]
        queue = deque(initial_dead_ends)
        
        while queue:
            current = queue.popleft()
            self.dead_ends.add(current)
            for neighbor in graph[current]:
                if neighbor not in self.dead_ends:
                    exits = sum(1 for n in graph[neighbor] if n not in self.dead_ends)
                    if exits <= 1:
                        queue.append(neighbor)
                        
        if self.junctions:
            q_junc = deque(self.junctions)
            for j in self.junctions:
                self.dist_to_junction[j] = 0
            while q_junc:
                curr = q_junc.popleft()
                for m, n_pos in self._get_valid_moves(curr):
                    if n_pos not in self.dist_to_junction:
                        self.dist_to_junction[n_pos] = self.dist_to_junction[curr] + 1
                        q_junc.append(n_pos)
        else:
            for pos in self.valid_tiles:
                self.dist_to_junction[pos] = 0

        self.map_analyzed = True

    def _is_valid_position(self, pos, map_state=None):
        return pos in self.valid_tiles

    def _get_valid_moves(self, pos, map_state=None):
        moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = (pos[0] + move.value[0], pos[1] + move.value[1])
            if next_pos in self.valid_tiles:
                moves.append((move, next_pos))
        return moves

    def _cell_exits(self, pos, map_state=None):
        exits = 0
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            n_pos = (pos[0] + move.value[0], pos[1] + move.value[1])
            if n_pos in self.valid_tiles:
                exits += 1
        return exits

    def get_BFS_map(self, start: tuple):
        if start in self.map_BFS_cache:
            return self.map_BFS_cache[start]
        
        dist = {start: 0}
        parent = {start: None}
        q = deque([start])
        
        while q:
            curr = q.popleft()
            for move, next_pos in self._get_valid_moves(curr):
                if next_pos not in dist:
                    dist[next_pos] = dist[curr] + 1
                    parent[next_pos] = (curr, move)
                    q.append(next_pos)
                    
        self.map_BFS_cache[start] = (dist, parent)
        if len(self.map_BFS_cache) > 2000:
            self.map_BFS_cache.pop(next(iter(self.map_BFS_cache)))
        return dist, parent

    def is_in_los(self, p1, p2):
        r1, c1 = p1
        r2, c2 = p2
        if r1 == r2:
            for c in range(min(c1, c2) + 1, max(c1, c2)):
                if (r1, c) not in self.valid_tiles: return False
            return True
        if c1 == c2:
            for r in range(min(r1, r2) + 1, max(r1, r2)):
                if (r, c1) not in self.valid_tiles: return False
            return True
        return False

    def get_voronoi_area(self, g_pos, p_dist_map):
        g_dist, _ = self.get_BFS_map(g_pos)
        safe_area = 0
        for pos, g_d in g_dist.items():
            p_d = p_dist_map.get(pos, 0)
            if g_d < (p_d / 1.5):
                safe_area += 1
        return safe_area

    def explore_escape_routes(self, start_pos, p_dist_map, enemy_pos, start_time, max_time):
        # Thuật toán Lookahead DFS tối giản, tập trung vào việc né BFS của Pacman
        global_best_move = Move.STAY
        max_fallback_dist = -1
        fallback_move = Move.STAY

        for target_depth in range(2, 12): # Nhìn trước tới 12 bước rất nhẹ nhàng
            stack = [(start_pos, [], 0, {start_pos})]
            depth_best_move = Move.STAY
            depth_best_score = -float('inf')

            while stack:
                if time.time() - start_time > max_time:
                    return global_best_move if global_best_move != Move.STAY else fallback_move

                curr_pos, path_moves, score, visited = stack.pop()
                at_horizon = len(path_moves) >= target_depth
                
                if at_horizon:
                    area = self.get_voronoi_area(curr_pos, p_dist_map)
                    final_score = score + (area * 100)
                    
                    if curr_pos in self.junctions:
                        final_score += 500

                    if final_score > depth_best_score:
                        depth_best_score = final_score
                        depth_best_move = path_moves[0] if len(path_moves) > 0 else Move.STAY
                    continue

                valid_next = self._get_valid_moves(curr_pos)
                is_dead_end = True

                for move, next_pos in valid_next:
                    if next_pos not in visited:
                        dist_to_p = p_dist_map.get(next_pos, 0)
                        
                        if len(path_moves) == 0 and dist_to_p > max_fallback_dist:
                            max_fallback_dist = dist_to_p
                            fallback_move = move

                        # Pacman chỉ đuổi theo BFS đơn thuần, nên ta cũng ước lượng cực kỳ đơn giản
                        p_future_dist = dist_to_p - ((len(path_moves) + 1) * 2.0)
                        
                        if p_future_dist <= 2.0:
                            if score > depth_best_score:
                                depth_best_score = score
                                depth_best_move = path_moves[0] if len(path_moves) > 0 else move
                            continue 

                        is_dead_end = False
                        step_score = dist_to_p * 2
                        
                        if self.is_in_los(next_pos, enemy_pos):
                            step_score -= 1000 # Thấy là né liền!
                            
                        # Phạt đi lùi
                        if len(path_moves) == 0 and next_pos in self._history:
                            step_score -= 500

                        new_visited = set(visited)
                        new_visited.add(next_pos)
                        stack.append((next_pos, path_moves + [move], score + step_score, new_visited))

                if is_dead_end and len(path_moves) > 0:
                    if score - 5000 > depth_best_score:
                        depth_best_score = score - 5000
                        depth_best_move = path_moves[0]

            if depth_best_score != -float('inf'):
                global_best_move = depth_best_move

        return global_best_move if global_best_move != Move.STAY else fallback_move

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        me = tuple(int(v) for v in my_position)
        
        if not self.map_analyzed:
            self._analyze_map(map_state)

        if enemy_position is not None:
            self.last_known_enemy_pos = tuple(int(v) for v in enemy_position)
            
        target_threat = enemy_position or self.last_known_enemy_pos
        valid_ghost_moves = self._get_valid_moves(me)

        if target_threat is None:
            if valid_ghost_moves:
                return random.choice(valid_ghost_moves)[0]
            return Move.STAY

        p_pos = tuple(int(v) for v in target_threat)
        p_dist, _ = self.get_BFS_map(p_pos) # Lấy luồng sóng BFS của địch làm kim chỉ nam né tránh

        start_time = time.time()
        max_time = 0.85 
        
        chosen_move = self.explore_escape_routes(me, p_dist, p_pos, start_time, max_time)

        if chosen_move != Move.STAY:
            for m, n_p in self._get_valid_moves(me):
                if m == chosen_move:
                    self._history.append(n_p)
                    if len(self._history) > self._history_ban:
                        self._history.pop(0)

        return chosen_move