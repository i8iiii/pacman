"""
Template for student agent implementation.

INSTRUCTIONS:
1. Copy this file to submissions/<your_student_id>/agent.py
2. Implement the PacmanAgent and/or GhostAgent classes
3. Replace the simple logic with your search algorithm
4. Test your agent using: python arena.py --seek <your_id> --hide example_student

IMPORTANT:
- Do NOT change the class names (PacmanAgent, GhostAgent)
- Do NOT change the method signatures (step, __init__)
- Pacman step must return either a Move or a (Move, steps) tuple where
    1 <= steps <= pacman_speed (provided via kwargs)
- Ghost step must return a Move enum value
- You CAN add your own helper methods
- You CAN import additional Python standard libraries
- Agents are STATEFUL - you can store memory across steps
- enemy_position may be None when limited observation is enabled
- map_state cells: 1=wall, 0=empty, -1=unseen (fog)
"""

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
import json
import heapq
from collections import deque
UNKNOWN = -1
EMPTY = 0
WALL = 1



class PacmanAgent(BasePacmanAgent):
    """Pacman v10.0 - Sequential Patrol Strategy"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Pacman_Sequential_Patrol"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        
        self.map_size = (21, 21)
        self.global_map = np.full(self.map_size, -1)
        self.last_known_enemy_pos = None
        self.enemy_history = deque(maxlen=5)
        self.my_history = deque(maxlen=15) 
        self.current_target = None
        
        # Mode tracking
        self.pursuit_mode = False  # True when Ghost is visible
        self.turns_since_lost_sight = 0  # Track how long since we last saw Ghost
        self.PURSUIT_PERSISTENCE = 8  # Continue pursuit for 8 turns after losing sight
        self.VISION_RADIUS = 5  # Maximum vision range (no obstacles, Manhattan distance)
        
        # Sequential patrol system - just track current target index
        self.current_target_index = 0
        
        # Pre-defined strategic target positions (patrol route)
        # Pacman will visit these positions sequentially
        self.TARGET_POSITIONS = [
            (9, 15),
            (13, 11),
            (9, 5),
            (5,12), #most-seen
            (15,11),
            (2, 15),    # Top-left corner area
            (2, 5),   # Top-right corner area
            (9, 10),  # Center of map
            (20, 5),   # Bottom-left corner area
            (20, 15),  # Bottom-right corner area
        ]

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        # Update map and history
        visible_mask = map_state != -1
        self.global_map[visible_mask] = map_state[visible_mask]
        self.my_history.append(my_position)
        
        # Anti-loop check
        if self.my_history.count(my_position) >= 3:
            return self._escape_loop(my_position)
        
        # ==================== MODE SWITCHING ====================
        if enemy_position:
            # PURSUIT MODE: Ghost is visible
            self.pursuit_mode = True
            self.turns_since_lost_sight = 0
            self.last_known_enemy_pos = enemy_position
            self.enemy_history.append(enemy_position)
            
            dist = self._manhattan_distance(my_position, enemy_position)
            #print(f"[PURSUIT MODE] Ghost at {enemy_position}, dist={dist}")
            
            # Direct axis-aligned chase with speed=2 advantage
            if self._on_same_axis(my_position, enemy_position):
                straight_move, steps = self._get_straight_advantage(my_position, enemy_position)
                if straight_move and steps > 0:
                    return (straight_move, steps)
            
            # Range-based strategy (optimized for vision_radius=5)
            if dist <= 2:
                # Close range: direct aggressive pursuit
                self.current_target = enemy_position
            elif dist <= 3:
                # Medium range: try interception
                intercept = self._calculate_interception(my_position, enemy_position)
                self.current_target = intercept or enemy_position
            elif dist <= self.VISION_RADIUS:
                # Max vision range: predict + corner cutting
                corner_target = self._corner_cut(my_position, enemy_position)
                if corner_target:
                    self.current_target = corner_target
                else:
                    self.current_target = self._smart_predict(my_position, enemy_position)
            else:
                # Beyond vision: shouldn't happen, but predict anyway
                self.current_target = self._smart_predict(my_position, enemy_position)
        else:
            # Ghost not visible - check if we should persist pursuit
            self.turns_since_lost_sight += 1
            
            if self.pursuit_mode and self.turns_since_lost_sight <= self.PURSUIT_PERSISTENCE:
                # PERSISTENT PURSUIT: Continue chasing last known position
                if self.last_known_enemy_pos:
                    dist_to_last = self._manhattan_distance(my_position, self.last_known_enemy_pos)
                    
                    if dist_to_last <= 2:
                        # Reached last known position - search nearby
                        self.current_target = self._search_nearby(my_position, self.last_known_enemy_pos)
                    else:
                        # Still moving to last known position
                        self.current_target = self.last_known_enemy_pos
                else:
                    # No last known position, switch to patrol
                    self.pursuit_mode = False
            else:
                # Give up pursuit - switch to SEARCH MODE (patrol)
                if self.pursuit_mode:
                    print(f"[PURSUIT END] Lost Ghost, returning to patrol")
                self.pursuit_mode = False
            
            # Get current target from sequence
            current_target = self.TARGET_POSITIONS[self.current_target_index]
            
            # Check if we reached the current target
            if my_position == current_target or self._manhattan_distance(my_position, current_target) <= 1:
                # Move to next target in sequence immediately
                self.current_target_index = (self.current_target_index + 1) % len(self.TARGET_POSITIONS)
                current_target = self.TARGET_POSITIONS[self.current_target_index]
                #print(f"[PATROL] ✓ Reached target! at Step {step_number} Next: {current_target} ({self.current_target_index + 1}/{len(self.TARGET_POSITIONS)})")
            
            self.current_target = current_target
            
            #print(f"[PATROL] Target {self.current_target_index + 1}/{len(self.TARGET_POSITIONS)}: {current_target}")

        # ==================== MOVEMENT EXECUTION ====================
        if self.current_target:
            # Try straight line with speed optimization
            straight_move, steps = self._get_straight_advantage(my_position, self.current_target)
            if straight_move and steps > 0:
                return (straight_move, steps)
            
            # A* pathfinding
            next_move = self._a_star(my_position, self.current_target)
            if next_move:
                steps = self._max_steps(my_position, next_move)
                if steps > 0:
                    return (next_move, steps)

        # Explore unexplored areas
        frontier_move = self._find_frontier(my_position)
        if frontier_move:
            steps = self._max_steps(my_position, frontier_move)
            if steps > 0:
                return (frontier_move, steps)

        # Fallback: random move
        return self._random_move(my_position)

    def _escape_loop(self, pos):
        self.my_history.clear()
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(moves)
        for m in moves:
            if self._is_passable(self._get_next_pos(pos, m)):
                return (m, 1)
        return (Move.STAY, 1)

    def _corner_cut(self, my_pos, ghost_pos):
        escape_positions = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._get_next_pos(ghost_pos, move)
            if self._is_passable(next_pos):
                escape_positions.append(next_pos)
                extended = self._get_next_pos(next_pos, move)
                if self._is_passable(extended):
                    escape_positions.append(extended)
        
        if len(escape_positions) <= 2:
            best_target = None
            min_time = float('inf')
            for pos in escape_positions:
                pacman_time = (self._manhattan_distance(my_pos, pos) + 1) // self.pacman_speed
                ghost_time = self._manhattan_distance(ghost_pos, pos)
                if pacman_time <= ghost_time and pacman_time < min_time:
                    min_time = pacman_time
                    best_target = pos
            return best_target
        return None

    def _smart_predict(self, my_pos, ghost_pos):
        """Smart prediction based on Ghost behavior patterns (Parkour AI)"""
        if len(self.enemy_history) < 2:
            return ghost_pos
        
        # Analyze recent moves
        recent_moves = []
        for i in range(1, min(3, len(self.enemy_history))):
            curr = self.enemy_history[-i]
            prev = self.enemy_history[-i-1]
            recent_moves.append((curr[0] - prev[0], curr[1] - prev[1]))
        
        candidates = [ghost_pos]
        
        # 1. Linear extrapolation (default prediction)
        for dr, dc in recent_moves:
            for steps in range(1, 5):  # Look further ahead
                pred_pos = (ghost_pos[0] + dr*steps, ghost_pos[1] + dc*steps)
                if self._is_passable(pred_pos):
                    candidates.append(pred_pos)
        
        # 2. Predict toward opposite sector (Ghost Parkour behavior)
        # Ghost tends to move away from Pacman toward opposite map sector
        mid_r, mid_c = 10, 10
        if my_pos[0] < mid_r:
            target_r = mid_r + 5  # Move toward bottom
        else:
            target_r = mid_r - 5  # Move toward top
        
        if my_pos[1] < mid_c:
            target_c = mid_c + 5  # Move toward right
        else:
            target_c = mid_c - 5  # Move toward left
        
        opposite_sector = (target_r, target_c)
        
        # 3. Predict positions between ghost and opposite sector
        dr_sector = 1 if opposite_sector[0] > ghost_pos[0] else -1 if opposite_sector[0] < ghost_pos[0] else 0
        dc_sector = 1 if opposite_sector[1] > ghost_pos[1] else -1 if opposite_sector[1] < ghost_pos[1] else 0
        
        for steps in range(1, 4):
            sector_pred = (ghost_pos[0] + dr_sector*steps, ghost_pos[1] + dc_sector*steps)
            if self._is_passable(sector_pred):
                candidates.append(sector_pred)
        
        # 4. Predict toward high-mobility areas (intersections)
        # Ghost prefers open areas with multiple exits
        for candidate in list(candidates):
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nearby = self._get_next_pos(candidate, move)
                if self._is_passable(nearby):
                    exits = sum(1 for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT] 
                               if self._is_passable(self._get_next_pos(nearby, m)))
                    if exits >= 3:  # Intersection
                        candidates.append(nearby)
        
        # Choose best candidate: closest to Pacman for interception
        return min(candidates, key=lambda p: self._manhattan_distance(my_pos, p))
    
    def _calculate_interception(self, my_pos, ghost_pos):
        """Calculate optimal interception point"""
        # Find positions where Pacman can arrive before or same time as Ghost
        # considering Pacman speed=2 and Ghost speed=1
        
        best_intercept = None
        min_ghost_escape = float('inf')
        
        # Check positions around ghost (within 4 steps)
        for dr in range(-4, 5):
            for dc in range(-4, 5):
                if abs(dr) + abs(dc) > 4:  # Skip far positions
                    continue
                
                intercept_pos = (ghost_pos[0] + dr, ghost_pos[1] + dc)
                if not self._is_passable(intercept_pos):
                    continue
                
                # Time for Pacman to reach (with speed=2)
                pacman_dist = self._manhattan_distance(my_pos, intercept_pos)
                pacman_time = (pacman_dist + 1) // 2  # Round up, speed=2
                
                # Time for Ghost to reach (speed=1)
                ghost_dist = self._manhattan_distance(ghost_pos, intercept_pos)
                ghost_time = ghost_dist
                
                # Count Ghost escape routes from this position
                escape_count = sum(1 for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
                                 if self._is_passable(self._get_next_pos(intercept_pos, m)))
                
                # Prefer positions where:
                # 1. Pacman arrives before Ghost
                # 2. Fewer escape routes for Ghost
                if pacman_time <= ghost_time:
                    score = escape_count + ghost_dist * 0.1  # Prefer closer + fewer escapes
                    if score < min_ghost_escape:
                        min_ghost_escape = score
                        best_intercept = intercept_pos
        
        return best_intercept
    
    def _search_nearby(self, my_pos, center_pos):
        """Search positions near last known Ghost location"""
        # Expand search in spiral pattern
        search_positions = []
        for radius in range(1, 4):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) + abs(dc) == radius:  # Manhattan circle
                        search_pos = (center_pos[0] + dr, center_pos[1] + dc)
                        if self._is_passable(search_pos):
                            # Prefer high-mobility positions (where Ghost might hide)
                            exits = sum(1 for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
                                      if self._is_passable(self._get_next_pos(search_pos, m)))
                            search_positions.append((search_pos, exits))
        
        if search_positions:
            # Sort by: 1) more exits (likely hiding spots), 2) closer to Pacman
            search_positions.sort(key=lambda x: (-x[1], self._manhattan_distance(my_pos, x[0])))
            return search_positions[0][0]
        
        return center_pos

    def _get_straight_advantage(self, start_pos, target_pos):
        dr = target_pos[0] - start_pos[0]
        dc = target_pos[1] - start_pos[1]
        
        # Calculate maximum possible steps in each direction
        if dr == 0 or dc == 0:
            # Same row or column
            if dc > 0:
                move = Move.RIGHT
                max_dist = dc
            elif dc < 0:
                move = Move.LEFT
                max_dist = -dc
            elif dr > 0:
                move = Move.DOWN
                max_dist = dr
            else:  # dr < 0
                move = Move.UP
                max_dist = -dr
            
            max_steps = self._max_steps(start_pos, move)
            # Return minimum of: max possible steps, max distance, or pacman speed
            return move, min(max_steps, max_dist, self.pacman_speed)
        return None, 0

    def _a_star(self, start, goal):
        pq = [(0, start, None)]
        visited = {start: 0}
        
        while pq:
            f, current, first_move = heapq.heappop(pq)
            if current == goal:
                return first_move
            if visited[current] > 20:
                continue
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                max_steps = self._max_steps(current, move)
                if max_steps > 0:
                    for steps in range(min(max_steps, self.pacman_speed), 0, -1):
                        neighbor = current
                        for _ in range(steps):
                            neighbor = self._get_next_pos(neighbor, move)
                        
                        if self._is_passable(neighbor):
                            new_g = visited[current] + 1
                            if neighbor not in visited or new_g < visited[neighbor]:
                                visited[neighbor] = new_g
                                h = self._manhattan_distance(neighbor, goal)
                                new_first = first_move or move
                                heapq.heappush(pq, (new_g + h, neighbor, new_first))
                        break
        return None

    def _find_frontier(self, start):
        queue = deque([(start, None)])
        visited = {start}
        while queue:
            curr, move = queue.popleft()
            if self._is_frontier(curr):
                return move
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = self._get_next_pos(curr, m)
                if self._is_passable(nxt) and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, move or m))
        return None

    def _random_move(self, pos):
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(moves)
        for m in moves:
            if self._is_passable(self._get_next_pos(pos, m)):
                steps = self._max_steps(pos, m)
                if steps > 0:
                    return (m, min(steps, self.pacman_speed))
        # Absolute fallback
        return (Move.STAY, 1)

    # Helper methods
    def _update_ghost_probability(self, my_position):
        """
        Advanced probability propagation considering Ghost behavior patterns.
        Ghost prefers: corners, dead-ends, areas far from Pacman, breaking LOS.
        """
        new_prob = np.zeros_like(self.ghost_probability)
        
        for r in range(21):
            for c in range(21):
                if self.ghost_probability[r, c] <= 0.001:
                    continue  # Skip negligible probabilities
                
                current_prob = self.ghost_probability[r, c]
                valid_moves = []
                
                # Check all possible Ghost moves
                for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    nr, nc = self._get_next_pos((r, c), move)
                    if self._is_passable((nr, nc)):
                        # Calculate move preference based on Ghost behavior
                        move_weight = self._calculate_ghost_move_preference(
                            (r, c), (nr, nc), my_position
                        )
                        valid_moves.append(((nr, nc), move_weight))
                
                if not valid_moves:
                    # Ghost stuck (shouldn't happen in valid map)
                    new_prob[r, c] += current_prob
                    continue
                
                # Normalize weights
                total_weight = sum(w for _, w in valid_moves)
                
                # Distribute probability according to move preferences
                for (nr, nc), weight in valid_moves:
                    new_prob[nr, nc] += current_prob * (weight / total_weight)
        
        # Normalize to ensure sum = 1.0
        total = new_prob.sum()
        if total > 0:
            self.ghost_probability = new_prob / total
        else:
            # No valid distribution, reset to uniform
            self.ghost_probability = np.ones((21, 21)) / (21 * 21)
    
    def _calculate_ghost_move_preference(self, from_pos, to_pos, pacman_pos):
        """
        Calculate how likely Ghost is to move from from_pos to to_pos.
        Higher weight = more likely move for Ghost.
        """
        weight = 1.0  # Base weight
        
        # 1. Prefer moving away from Pacman
        old_dist = self._manhattan_distance(from_pos, pacman_pos)
        new_dist = self._manhattan_distance(to_pos, pacman_pos)
        if new_dist > old_dist:
            weight *= 2.0  # Strongly prefer moving away
        elif new_dist < old_dist:
            weight *= 0.3  # Penalize moving toward Pacman
        
        # 2. Prefer corners and dead-ends (Ghost hiding behavior)
        walls_around = self._count_walls_around(to_pos)
        if walls_around >= 3:
            weight *= 1.5  # Dead-end preference
        elif walls_around == 2:
            weight *= 1.2  # Corner preference
        
        # 3. Prefer positions that break line of sight
        if not self._has_line_of_sight(to_pos, pacman_pos):
            weight *= 1.8  # Strong preference for hiding
        
        return weight
    
    def _decay_visible_probabilities(self, map_state, my_position):
        """
        Reduce probabilities in areas we can currently see (Ghost is not there).
        """
        for r in range(21):
            for c in range(21):
                if map_state[r, c] != -1:  # We can see this cell
                    # Dramatically reduce probability here
                    self.ghost_probability[r, c] *= 0.01
        
        # Normalize after decay
        total = self.ghost_probability.sum()
        if total > 0.001:
            self.ghost_probability /= total
        else:
            # If all probabilities decayed to near-zero, reset to uniform over unseen areas
            unseen_mask = (map_state == -1) & (self.global_map != 1)
            unseen_count = unseen_mask.sum()
            if unseen_count > 0:
                self.ghost_probability = unseen_mask.astype(float) / unseen_count
            else:
                self.ghost_probability = np.ones((21, 21)) / (21 * 21)
    
    def _find_best_probability_target(self, my_position):
        """
        Find the best target position based on both probability and reachability.
        Don't just go to highest probability - consider distance and accessibility.
        """
        # Find top probability positions
        flat_probs = self.ghost_probability.flatten()
        top_indices = np.argsort(flat_probs)[-10:][::-1]  # Top 10 positions
        
        best_target = None
        best_score = -1
        
        for idx in top_indices:
            pos = np.unravel_index(idx, self.ghost_probability.shape)
            prob = self.ghost_probability[pos]
            
            if prob < 0.001:
                break  # No more significant probabilities
            
            # Calculate composite score: probability / distance
            dist = self._manhattan_distance(my_position, pos)
            if dist == 0:
                dist = 1
            
            # Score = probability * reachability_factor
            # Closer positions get higher scores
            reachability = 1.0 / (1.0 + dist / 10.0)
            score = prob * reachability
            
            if score > best_score:
                best_score = score
                best_target = pos
        
        return best_target
    
    def _count_walls_around(self, pos):
        """Count number of walls adjacent to position"""
        count = 0
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            r, c = pos[0] + dr, pos[1] + dc
            if not self._is_in_bounds((r, c)) or self.global_map[r, c] == 1:
                count += 1
        return count
    
    def _has_line_of_sight(self, pos1, pos2):
        """Check if there's a clear line of sight between two positions"""
        r1, c1 = pos1
        r2, c2 = pos2
        
        # Check horizontal line of sight
        if r1 == r2:
            c_start, c_end = (c1, c2) if c1 < c2 else (c2, c1)
            for c in range(c_start + 1, c_end):
                if self.global_map[r1, c] == 1:
                    return False
            return True
        
        # Check vertical line of sight
        if c1 == c2:
            r_start, r_end = (r1, r2) if r1 < r2 else (r2, r1)
            for r in range(r_start + 1, r_end):
                if self.global_map[r, c1] == 1:
                    return False
            return True
        
        return False  # Not on same axis
    
    # Helper methods (existing)
    def _on_same_axis(self, pos1, pos2):
        """Check if two positions are on the same row or column"""
        return pos1[0] == pos2[0] or pos1[1] == pos2[1]

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

    def _is_frontier(self, pos):
        if self.global_map[pos] != 0:
            return False
        r, c = pos
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r+dr, c+dc
            if self._is_in_bounds((nr, nc)) and self.global_map[nr, nc] == -1:
                return True
        return False

    def _is_passable(self, pos):
        return self._is_in_bounds(pos) and self.global_map[pos] != 1

    def _is_in_bounds(self, pos):
        return 0 <= pos[0] < 21 and 0 <= pos[1] < 21

    def _get_next_pos(self, pos, move):
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def _max_steps(self, pos, move):
        steps = 0
        curr = pos
        for _ in range(self.pacman_speed):
            next_p = self._get_next_pos(curr, move)
            if not self._is_passable(next_p):
                break
            steps += 1
            curr = next_p
        return steps
    

import numpy as np
import random
import os
import json
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Parkour_Ghost"
        
        # State & Memory
        self.history = deque(maxlen=4)
        self.last_known_pacman = None
        self.turns_since_seen = 0
        
        # CHIẾN THUẬT KHAI CUỘC (Ambush)
        self.opening_target = None
        self.opening_moves = [] 
        self.sectors = {}
        
        # TRẠNG THÁI
        self.in_opening_phase = True
        self.in_camping_phase = False 

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        
        height, width = map_state.shape
        
        # Setup động (Dynamic Setup) cho lần gọi đầu tiên để thích ứng mọi map
        if not self.sectors:
            self.sectors = {
                'TR': (height // 4, width * 3 // 4),
                'TL': (height // 4, width // 4),
                'BL': (height * 3 // 4, width // 4),
                'BR': (height * 3 // 4, width * 3 // 4)
            }
            # Mặc định góc phục kích theo tỷ lệ map, nếu bị vướng tường thì lấy vị trí hiện tại
            ambush_spot = (5, 12)
            if ambush_spot[0] < height and ambush_spot[1] < width and map_state[ambush_spot[0], ambush_spot[1]] != 1:
                self.opening_target = ambush_spot
            else:
                self.opening_target = my_position # Bỏ qua phase mở đầu nếu vị trí lỗi
                self.in_opening_phase = False
                self.in_camping_phase = True
                
        # --- 0. BÁO ĐỘNG ĐỎ ---
        if enemy_position is not None:
            if self.in_opening_phase or self.in_camping_phase:
                self.in_opening_phase = False
                self.in_camping_phase = False
            
            self.last_known_pacman = enemy_position
            self.turns_since_seen = 0
        else:
            self.turns_since_seen += 1

        if self.last_known_pacman is None:
            self.last_known_pacman = (height // 2, width // 2)

        # --- PHASE 1: OPENING ---
        if self.in_opening_phase:
            if my_position == self.opening_target:
                self.in_opening_phase = False
                self.in_camping_phase = True
                return Move.STAY
            else:
                if not self.opening_moves:
                    self.opening_moves = self._bfs_path(my_position, self.opening_target, map_state)
                
                if self.opening_moves:
                    next_move = self.opening_moves.pop(0)
                    if next_move:
                        dr, dc = next_move.value
                        self.history.append((my_position[0]+dr, my_position[1]+dc))
                        return next_move
                self.in_opening_phase = False 

        # --- PHASE 2: CAMPING ---
        if self.in_camping_phase:
            return Move.STAY

        # --- PHASE 3: ACTIVE EVASION (PARKOUR STYLE) ---
        pacman_dist_map = self._get_bfs_distance_map(self.last_known_pacman, map_state)
        
        # 2. Tìm điểm an toàn nhất (xa Pacman nhất theo đường đi thực tế)
        target_pos = my_position
        max_dist = -1
        for pos, dist in pacman_dist_map.items():
            if dist > max_dist:
                max_dist = dist
                target_pos = pos

        # 3. Chấm điểm nước đi dựa trên khoảng cách thực tế (truyền thêm pacman_dist_map)
        best_move = self._evaluate_best_move(my_position, self.last_known_pacman, target_pos, map_state, pacman_dist_map)
        if best_move is None: best_move = Move.STAY

        # Anti-stuck
        if best_move == Move.STAY and self.turns_since_seen < 5:
             valid = self._get_valid_neighbors(my_position, map_state)
             candidates = [n for n in valid if n not in self.history]
             if candidates:
                 next_pos = random.choice(candidates)
                 dr, dc = next_pos[0]-my_position[0], next_pos[1]-my_position[1]
                 for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                     if m.value == (dr, dc): return m

        dr, dc = best_move.value
        self.history.append((my_position[0]+dr, my_position[1]+dc))
        return best_move

    # --- HELPER METHODS ---
    
    def _get_bfs_distance_map(self, start_pos, map_state):
        """Trả về một dictionary chứa khoảng cách thực tế từ start_pos đến mọi ô"""
        distances = {start_pos: 0}
        queue = [start_pos]
        
        while queue:
            curr = queue.pop(0)
            dist = distances[curr]
            
            for nr, nc in self._get_valid_neighbors(curr, map_state):
                if (nr, nc) not in distances:
                    distances[(nr, nc)] = dist + 1
                    queue.append((nr, nc))
        return distances

    def _get_valid_neighbors(self, pos, map_state):
        r, c = pos
        height, width = map_state.shape
        valid = []
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = r + m.value[0], c + m.value[1]
            if 0 <= nr < height and 0 <= nc < width and map_state[nr, nc] != 1:
                valid.append((nr, nc))
        return valid

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

    def _bfs_path(self, start, target, map_state):
        if start == target: return []
        queue = [(start, [])]
        visited = {start}
        
        while queue:
            curr, path = queue.pop(0)
            if curr == target: return path
            
            for nr, nc in self._get_valid_neighbors(curr, map_state):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    new_path = list(path)
                    dr, dc = nr - curr[0], nc - curr[1]
                    move_enum = None
                    for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                        if m.value == (dr, dc):
                            move_enum = m
                            break
                    if move_enum:
                        new_path.append(move_enum)
                        queue.append(((nr, nc), new_path))
        return []

    def _evaluate_best_move(self, my_pos, enemy_pos, target_pos, map_state, pacman_dist_map):
        valid_neighbors = self._get_valid_neighbors(my_pos, map_state)
        moves_score = []
        height, width = map_state.shape

        for nr, nc in valid_neighbors:
            score = 0
            
            # 1. SAFETY
            real_dist_to_enemy = pacman_dist_map.get((nr, nc), 0)
            # dist_to_enemy = abs(nr - enemy_pos[0]) + abs(nc - enemy_pos[1])
            # if dist_to_enemy < 4: 
            #     score -= 2000
            #     score += dist_to_enemy * 50 
            if real_dist_to_enemy <= 3: 
                score -= 2000 # Tử địa thực sự
            elif real_dist_to_enemy <= 5:
                score -= 500
                
            score += real_dist_to_enemy * 20
            
            # 2. LOS BREAKING
            if not self._has_line_of_sight((nr, nc), enemy_pos, map_state):
                score += 300 

            # 3. ANTI-CORRIDOR & JUNCTION
            next_valid_moves = self._get_valid_neighbors((nr, nc), map_state)
            num_exits = len(next_valid_moves)
            
            is_corridor = False
            if num_exits == 2:
                r1, c1 = next_valid_moves[0]
                r2, c2 = next_valid_moves[1]
                if r1 == r2 or c1 == c2: 
                    is_corridor = True
            
            if num_exits >= 3: score += 100 
            elif num_exits == 1: score -= 500 
            elif is_corridor: score -= 100 
            else: score += 50 

            # 4. WALL HUGGING
            adjacent_walls = 0
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                 check_r, check_c = nr+dr, nc+dc
                 if 0 <= check_r < height and 0 <= check_c < width:
                     if map_state[check_r, check_c] == 1:
                         adjacent_walls += 1
            
            if adjacent_walls > 0: score += 20 * adjacent_walls 

            # 5. TARGET DIRECTION
            dist_to_target = abs(nr - target_pos[0]) + abs(nc - target_pos[1])
            score -= dist_to_target * 5 

            # 6. HISTORY
            if (nr, nc) in self.history:
                score -= 200

            dr, dc = nr - my_pos[0], nc - my_pos[1]
            move_enum = Move.STAY
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if m.value == (dr, dc): 
                    move_enum = m
                    break
            
            moves_score.append((score, move_enum))

        if not moves_score: return Move.STAY
        moves_score.sort(key=lambda x: x[0], reverse=True)
        return moves_score[0][1]