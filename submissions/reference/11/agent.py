import sys
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np

import random
import time 
import heapq #For priority queue
from collections import deque #For BFS 

#This pacman using A* algorithm and intercept method chase down enemy.
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "YoruNiKakeru Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))

        # Ghost position
        self.last_known_ghost_pos = None
        self.ghost_velocity = (0, 0)       #Estimated ghost's movement direction
        self.last_ghost_update_step = -1

        # Map
        self.junctions = []                
        self.dead_ends = []                
        self.map_analyzed = False

        self.current_intercept = None      
        self.intercept_confirmed_step = -1

    #Find all junctions and dead_ends on map in one time
    def analyze_map(self, map_state):
        rows, cols = map_state.shape
        for r in range(rows):
            for c in range(cols):
                if map_state[r][c] != 0:
                    continue
                neighbors = self._get_neighbors((r, c), map_state)
                deg = len(neighbors)
                if deg >= 3:
                    self.junctions.append((r, c))
                elif deg == 1:
                    self.dead_ends.append((r, c))

    #Find all possible cells nearby
    def _get_neighbors(self, pos, map_state):
        result = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            npos = (pos[0] + dr, pos[1] + dc)
            if self._is_valid_position(npos, map_state):
                result.append(npos)
        return result

    #Make a map with all cost to any cells
    def bfs_cost(self, start, map_state):
        if not self._is_valid_position(start, map_state):
            return {}
        rows, cols = map_state.shape
        cost = {}
        queue = deque()
        cost[start] = 0
        queue.append(start)
        while queue:
            cur = queue.popleft()
            for nb in self._get_neighbors(cur, map_state):
                if nb not in cost:
                    cost[nb] = cost[cur] + 1
                    queue.append(nb)
        return cost

    def a_star(self, start, goal, map_state):
        def h(p): return abs(p[0] - goal[0]) + abs(p[1] - goal[1])
        heap = [(h(start), start)]
        g = {start: 0}
        parent = {}
        while heap:
            f, cur = heapq.heappop(heap)
            if cur == goal:
                return parent
            if f > g.get(cur, 0) + h(cur):
                continue
            for nb in self._get_neighbors(cur, map_state):
                ng = g[cur] + 1
                if ng < g.get(nb, float('inf')):
                    g[nb] = ng
                    parent[nb] = cur
                    heapq.heappush(heap, (ng + h(nb), nb))
        return parent

    #Make a way from goal back to start after get ways from a_star
    def reconstruct_path(self, parent, start, goal):
        if goal not in parent and goal != start:
            return []
        path = []
        cur = goal
        while cur != start:
            path.append(cur)
            cur = parent[cur]
        path.append(start)
        path.reverse()
        return path

    #Return path from pos start to goal, this is the function that mixes reconstruct_path and a_star for convenient 
    def path_to(self, start, goal, map_state):
        parent = self.a_star(start, goal, map_state)
        return self.reconstruct_path(parent, start, goal)

    #Predict all the cells that ghost maybe move in
    def predict_ghost_positions(self, ghost_pos, pac_cost_from_ghost, map_state, steps_ahead=6):
        reachable = {ghost_pos: 0}
        frontier = deque([(ghost_pos, 0)])
        while frontier:
            pos, steps = frontier.popleft()
            if steps >= steps_ahead:
                continue
            for nb in self._get_neighbors(pos, map_state):
                if nb not in reachable:
                    reachable[nb] = steps + 1
                    frontier.append((nb, steps + 1))
        return reachable
    
    #Estimate the ghost's direction of movement from two consecutive locations.
    def estimate_ghost_velocity(self, ghost_pos, step_number):
        if self.last_known_ghost_pos and self.last_ghost_update_step == step_number - 1:
            dr = ghost_pos[0] - self.last_known_ghost_pos[0]
            dc = ghost_pos[1] - self.last_known_ghost_pos[1]
            dr = max(-1, min(1, dr))
            dc = max(-1, min(1, dc))
            self.ghost_velocity = (dr, dc)
        return self.ghost_velocity
    
    #Find best intercept with the junctions and dead_ends that near ghost as close as possible
    #And also make sure that Pacman arrive to that cell earlier than ghost
    def find_best_intercept(self, my_pos, ghost_pos, pac_cost, ghost_cost, map_state):

        best_pos = None
        best_score = float('-inf')
        candidates = self.junctions + self.dead_ends
        if not candidates:
            return ghost_pos, 0

        for cand in candidates:
            p_dist = pac_cost.get(cand)
            g_dist = ghost_cost.get(cand)

            if p_dist is None or g_dist is None:
                continue

            pac_time = p_dist / (self.pacman_speed * 1.5) # 1.5 because pac speed = 2 or = 1 if corner, so avg = 1.5
            ghost_time = g_dist  # ghost speed = 1

            if pac_time > ghost_time + 0.5:
                continue

            advantage = ghost_time - pac_time   
            proximity = -g_dist                 
            is_dead_end = 1 if cand in self.dead_ends else 0

            score = advantage * 10 + proximity * 2 + is_dead_end * 5

            if score > best_score:
                best_score = score
                best_pos = cand

        return best_pos, best_score

    #Assume Ghost is moving straight along the velocity and Pacman has the ability to intercept
    def find_direct_intercept(self, my_pos, ghost_pos, ghost_velocity, pac_cost, map_state):
        if ghost_velocity == (0, 0):
            return None

        dr, dc = ghost_velocity
        best = None
        best_dist = float('inf')

        pos = ghost_pos
        for steps in range(1, 11):
            next_pos = (pos[0] + dr, pos[1] + dc)
            if not self._is_valid_position(next_pos, map_state):
                break
            pos = next_pos

            p_dist = pac_cost.get(pos)
            if p_dist is None:
                continue

            pac_time = p_dist / (self.pacman_speed * 1.5)
            ghost_time = steps

            if pac_time <= ghost_time - 1:
                if p_dist < best_dist:
                    best_dist = p_dist
                    best = pos

        return best

    def follow_path(self, path, my_pos):
        if len(path) < 2:
            return (Move.STAY, 1)

        next_pos = path[1]
        dr = next_pos[0] - my_pos[0]
        dc = next_pos[1] - my_pos[1]

        if dr > 0:   move = Move.DOWN
        elif dr < 0: move = Move.UP
        elif dc > 0: move = Move.RIGHT
        else:         move = Move.LEFT

        steps = 1
        for i in range(2, len(path)):
            dri = path[i][0] - path[i-1][0]
            dci = path[i][1] - path[i-1][1]
            if dri == dr and dci == dc:
                steps += 1
            else:
                break

        return (move, min(steps, self.pacman_speed))

    def explore(self, my_pos, map_state):
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(moves)
        for move in moves:
            steps = self._max_valid_steps(my_pos, move, map_state, self.pacman_speed)
            if steps > 0:
                return (move, steps)
        return (Move.STAY, 1)

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):

        if not self.map_analyzed:
            self.analyze_map(map_state)
            self.map_analyzed = True

        if enemy_position is not None:
            self.estimate_ghost_velocity(enemy_position, step_number)
            self.last_known_ghost_pos = enemy_position
            self.last_ghost_update_step = step_number
            ghost_pos = enemy_position
        elif self.last_known_ghost_pos is not None:
            dr, dc = self.ghost_velocity
            predicted = (self.last_known_ghost_pos[0] + dr,
                         self.last_known_ghost_pos[1] + dc)
            if self._is_valid_position(predicted, map_state):
                ghost_pos = predicted
            else:
                ghost_pos = self.last_known_ghost_pos
        else:
            return self.explore(my_position, map_state)

        pac_cost = self.bfs_cost(my_position, map_state)
        ghost_cost = self.bfs_cost(ghost_pos, map_state)

        ghost_real_dist = pac_cost.get(ghost_pos)
        if ghost_real_dist is None:
            return self.explore(my_position, map_state)

        effective_speed = self.pacman_speed * 1.5
        if ghost_real_dist <= effective_speed * 3:
            path = self.path_to(my_position, ghost_pos, map_state)
            if path and len(path) >= 2:
                return self.follow_path(path, my_position)

        direct_intercept = self.find_direct_intercept(my_position, ghost_pos, self.ghost_velocity, pac_cost, map_state)
        intercept_pos, intercept_score = self.find_best_intercept(my_position, ghost_pos, pac_cost, ghost_cost, map_state)
        goal = None

        if direct_intercept is not None:
            d_pac = pac_cost.get(direct_intercept, float('inf'))
            d_ghost = ghost_cost.get(direct_intercept, float('inf'))
            if d_pac / effective_speed < d_ghost:
                goal = direct_intercept

        if goal is None:
            if (self.current_intercept is not None and \
                self.intercept_confirmed_step >= step_number - 5):
                cp = self.current_intercept
                cp_pac = pac_cost.get(cp, float('inf'))
                cp_ghost = ghost_cost.get(cp, float('inf'))
                if cp_pac / effective_speed <= cp_ghost and cp_pac > 0:
                    goal = cp

            if goal is None and intercept_pos is not None:
                self.current_intercept = intercept_pos
                self.intercept_confirmed_step = step_number
                goal = intercept_pos

        if goal is None:
            goal = ghost_pos

        path = self.path_to(my_position, goal, map_state)
        if not path or len(path) < 2:
            path = self.path_to(my_position, ghost_pos, map_state)
            if not path or len(path) < 2:
                return self.explore(my_position, map_state)

        return self.follow_path(path, my_position)

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



class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Hybrid Phantom v2"
        self._history = []
        self._history_ban = 4  
        self.last_known_enemy_pos = None 
        
        # Tá»‘c Ä‘á»™ giáº£ Ä‘á»‹nh cá»§a Pacman
        self.pacman_speed = max(2, int(kwargs.get("pacman_speed", 2))) 
        
        # Cáº¥u trÃºc báº£n Ä‘á»“
        self.map_analyzed = False
        self.valid_tiles = {}
        self.dead_ends = set()
        self.junctions = set()
        self.map_BFS_cache = {}
        self.dist_to_junction = {} # Má»šI: Báº£n Ä‘á»“ khoáº£ng cÃ¡ch thoÃ¡t hiá»ƒm

    def _analyze_map(self, map_state):
        if self.map_analyzed:
            return
        
        height, width = map_state.shape
        # 1. QuÃ©t map cÆ¡ báº£n
        for r in range(height):
            for c in range(width):
                if map_state[r, c] in [0, -1]:
                    self.valid_tiles[(r, c)] = True
                    if self._cell_exits((r, c), map_state) >= 3:
                        self.junctions.add((r, c))
                        
        # 2. XÃ¢y dá»±ng Ä‘á»“ thá»‹ & NgÃµ cá»¥t
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
                        
        # 3. Má»šI: Multi-source BFS tÃ­nh khoáº£ng cÃ¡ch tá»›i NgÃ£ TÆ° gáº§n nháº¥t
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

    def _cell_exits(self, pos, map_state):
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
        if len(self.map_BFS_cache) > 40:
            self.map_BFS_cache.pop(next(iter(self.map_BFS_cache)))
        return dist, parent

    def is_in_los(self, p1, p2, map_state):
        r1, c1 = p1
        r2, c2 = p2
        if r1 == r2:
            for c in range(min(c1, c2) + 1, max(c1, c2)):
                if map_state[r1][c] == 1: return False
            return True
        if c1 == c2:
            for r in range(min(r1, r2) + 1, max(r1, r2)):
                if map_state[r][c1] == 1: return False
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

    def pacman_tick(self, path: list) -> int:
        if not path: return 0
        ticks = 0
        current_move = path[0]
        steps = 0
        
        for move in path:
            if move == current_move and steps < self.pacman_speed:
                steps += 1
            else:
                ticks += 1
                current_move = move
                steps = 1
        if steps > 0:
            ticks += 1
        return ticks

    def minimax(self, ghost_pos: tuple, pacman_pos: tuple, map_state: np.ndarray, depth: int, alpha: float, beta: float, ghost_turn: bool):
        if ghost_pos == pacman_pos:
            return -10000 - depth 

        if depth == 0:
            p_dist, p_parent = self.get_BFS_map(pacman_pos)
            
            if ghost_pos not in p_dist:
                return float('inf') 
            
            dist_to_pacman = p_dist[ghost_pos]
            if dist_to_pacman == 0:
                return -10000

            path = []
            curr = ghost_pos
            while curr != pacman_pos:
                prev, move_taken = p_parent[curr]
                path.append(move_taken)
                curr = prev
            path.reverse()
            
            ticks = self.pacman_tick(path)
            score = (ticks * 100) + len(path)
            
            if ghost_pos in self.dead_ends:
                score -= 5000 
                
            score += self.get_voronoi_area(ghost_pos, p_dist) * 10 
            
            if ghost_pos in self.junctions:
                score += 50 
                
            if self.is_in_los(ghost_pos, pacman_pos, map_state):
                score -= 200 
            
            # --- VÃ Lá»–I HORIZON EFFECT: Há»† THá»NG Cáº¢NH BÃO ÄÆ¯á»œNG Háº¦M ---
            d_junc = self.dist_to_junction.get(ghost_pos, 0)
            if d_junc > 0:
                # Náº¿u thá»i gian Pacman Ä‘uá»•i tá»›i nhá» hÆ¡n thá»i gian Ghost láº¿t Ä‘Æ°á»£c ra khá»i háº»m -> Tá»± sÃ¡t!
                if (dist_to_pacman / 1.5) <= d_junc:
                    score -= 3000 
                    
            return score
            
        if ghost_turn:
            max_eval = -float('inf')
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
                next_g = (ghost_pos[0] + move.value[0], ghost_pos[1] + move.value[1]) if move != Move.STAY else ghost_pos
                
                if self._is_valid_position(next_g) or move == Move.STAY:
                    eval_score = self.minimax(next_g, pacman_pos, map_state, depth - 1, alpha, beta, False)
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break 
            return max_eval
            
        else:
            min_eval = float('inf')
            valid_moves_exist = False
            
            # --- VÃ Lá»–I XUYÃŠN THáº¤U: QUÃ‰T VA CHáº M TRÃŠN Tá»ªNG BÆ¯á»šC CHáº Y ---
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                curr = pacman_pos
                caught = False
                steps_taken = 0
                
                for _ in range(self.pacman_speed):
                    n_pos = (curr[0] + move.value[0], curr[1] + move.value[1])
                    if not self._is_valid_position(n_pos):
                        break
                    curr = n_pos
                    steps_taken += 1
                    
                    # Pacman cháº¡m trÃºng Ghost trong lÃºc Ä‘ang phi nÆ°á»›c Ä‘áº¡i
                    if curr == ghost_pos:
                        caught = True
                        break
                        
                if caught:
                    eval_score = -10000 - depth
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
                elif steps_taken > 0:
                    valid_moves_exist = True
                    eval_score = self.minimax(ghost_pos, curr, map_state, depth - 1, alpha, beta, True)
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
                        
            if not valid_moves_exist and min_eval == float('inf'):
                return 0 
            return min_eval

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        me = tuple(int(v) for v in my_position)
        
        if not self.map_analyzed:
            self._analyze_map(map_state)

        if enemy_position is not None:
            self.last_known_enemy_pos = tuple(int(v) for v in enemy_position)
            
        target_threat = enemy_position or self.last_known_enemy_pos

        if target_threat is None:
            moves = self._get_valid_moves(me)
            if moves:
                return random.choice(moves)[0]
            return Move.STAY

        p_pos = tuple(int(v) for v in target_threat)

        best_move = Move.STAY
        max_eval = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        depth_limit = 3 

        for move, next_pos in self._get_valid_moves(me):
            eval_score = self.minimax(next_pos, p_pos, map_state, depth_limit, alpha, beta, False)
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
                
            alpha = max(alpha, eval_score)

        return best_move