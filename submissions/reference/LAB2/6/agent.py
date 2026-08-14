import numpy as np
import heapq
from collections import deque
from typing import Tuple, Optional
from environment import Move
from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent

class GhostAgent(BaseGhostAgent):    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The Global Memory Map (Starts completely unknown: -1)
        # We assume max size is 21x21 based on the PDF.
        self.memory_map = np.full((21, 21), -1, dtype=int)
        
        # State tracking
        self.last_known_pacman = None
        self.turns_since_seen = 999
        self.current_hideout = None

        self.last_pos = None

        self.camp_timer = 0
        self.is_roaming = False
        self.roam_steps_left = 0

    def _update_memory(self, local_map_state: np.ndarray):
        """
        Overlays the visible local map onto the global memory map.
        Everything the Ghost sees is remembered permanently.
        """
        # Find all cells in the local map that are not fog (-1)
        # and copy them into our persistent memory.
        visible_mask = local_map_state != -1
        self.memory_map[visible_mask] = local_map_state[visible_mask]

    def _is_in_line_of_sight(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """
        Checks if two positions share an unbroken straight line in our memory map.
        Crucial for knowing if Pacman can see us.
        """
        r1, c1 = pos1
        r2, c2 = pos2

        if abs(r1 - r2) > 5 or abs(c1 - c2) > 5:
            return False
        
        if r1 == r2:
            c_min, c_max = min(c1, c2), max(c1, c2)
            for c in range(c_min + 1, c_max):
                if self.memory_map[r1, c] == 1: # Wall blocks vision
                    return False
            return True
            
        if c1 == c2:
            r_min, r_max = min(r1, r2), max(r1, r2)
            for r in range(r_min + 1, r_max):
                if self.memory_map[r, c1] == 1:
                    return False
            return True
            
        return False

    def step(self, map_state: np.ndarray, 
             my_position: Tuple[int, int], 
             enemy_position: Optional[Tuple[int, int]],
             step_number: int) -> Move:
             
        # Update our mental map (revealing the -1 fog)
        self._update_memory(map_state)
        
        # Track Pacman's whereabouts under TRUE Fog of War
        if enemy_position is not None:
            # The Arena says Pacman is in our 5-tile vision cross.
            # Double-check Line of Sight just to be mathematically safe against walls.
            if self._is_in_line_of_sight(my_position, enemy_position):
                self.last_known_pacman = enemy_position
                self.turns_since_seen = 0

                # PACMAN SPOTTED! Reset all migration timers!
                self.camp_timer = 0
                self.is_roaming = False
                self.roam_steps_left = 0

            else:
                self.turns_since_seen += 1
        else:
            # Pacman is completely lost in the fog.
            self.turns_since_seen += 1

        # ---------------------------------------------------------
        # STATE MACHINE LOGIC
        # ---------------------------------------------------------
        if self.turns_since_seen == 0:
            best_move = self._execute_panic_flee(my_position, self.last_known_pacman)
            
        elif self.turns_since_seen < 4:
            best_move = self._execute_relocate(my_position)
            
        else:
            # -----------------------------------------------------
            # THE MIGRATION SYSTEM
            # -----------------------------------------------------
            if self.is_roaming:
                # Actively moving to a new quadrant of the map
                best_move = self._execute_roam(my_position)
                self.roam_steps_left -= 1
                
                # If we've walked enough steps, go back to hiding!
                if self.roam_steps_left <= 0:
                    self.is_roaming = False
            else:
                best_move = self._execute_hide(my_position)
                
                # If the Ghost decided to sleep in a corner, tick the timer!
                if best_move == Move.STAY:
                    self.camp_timer += 1
                    
                    # If we've camped for 25 turns, it's time to migrate!
                    if self.camp_timer >= 7:
                        self.is_roaming = True
                        self.roam_steps_left = 12 # Walk toward center for 12 turns
                        self.camp_timer = 0
                else:
                    # If the Ghost is still creeping to the corner, reset timer
                    self.camp_timer = 0

        self.last_pos = my_position
        return best_move

    def _execute_roam(self, my_pos: Tuple[int, int]) -> Move:
        """
        Migration Phase: Walk toward the center of the map.
        This pulls the Ghost out of its old hiding spot so the Corner Creeper 
        will push it into a new quadrant later!
        """
        best_move = Move.STAY
        best_score = float('inf') # We want to MINIMIZE distance to center!
        
        safe_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = my_pos[0] + move.value[0], my_pos[1] + move.value[1]
            if 0 <= nr < 21 and 0 <= nc < 21 and self.memory_map[nr, nc] == 0:
                safe_moves.append((move, nr, nc))
                
        for move, nr, nc in safe_moves:
            # Keep momentum: Do not backtrack unless it's a dead end
            if len(safe_moves) > 1 and self.last_pos is not None and (nr, nc) == self.last_pos:
                continue
                
            # Score: Lower is better (closer to the center)
            score = abs(nr - 10) + abs(nc - 10)
            
            if score < best_score:
                best_score = score
                best_move = move
                
        # Fallback if trapped
        if best_move == Move.STAY and safe_moves:
            return safe_moves[0][0]
            
        return best_move

    def _execute_panic_flee(self, my_pos: Tuple[int, int], pacman_pos: Tuple[int, int]) -> Move:
        """
        Emergency evasion. Pacman sees us.
        Goal: Step to the nearest tile that breaks Line of Sight.
        """
        queue = deque([(my_pos, [])])
        visited = {my_pos}
        
        best_move = Move.STAY
        
        while queue:
            curr, path = queue.popleft()
            
            # Check if this tile breaks LOS from Pacman's current position
            if not self._is_in_line_of_sight(curr, pacman_pos):
                if path:
                    return path[0] # Return the very first move to get on this escape path
            
            # Continue searching
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nr, nc = curr[0] + move.value[0], curr[1] + move.value[1]
                if 0 <= nr < 21 and 0 <= nc < 21:
                    # Treat unknown (-1) as walls during panic to be safe, only walk on known empty (0)
                    if self.memory_map[nr, nc] == 0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [move]))
                        
        # Fallback if trapped: run away from Pacman using Manhattan distance
        return self._fallback_flee(my_pos, pacman_pos)
        
    def _execute_relocate(self, my_pos: Tuple[int, int]) -> Move:
        """
        Pacman just lost sight of us, but he is investigating our Last Known Position
        Goal: Run away from his last known position, prioritizing taking turns to confuse him.
        """
        if self.last_known_pacman is None:
            return Move.STAY
            
        best_move = Move.STAY
        best_score = -float('inf')
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = my_pos[0] + move.value[0], my_pos[1] + move.value[1]
            
            if 0 <= nr < 21 and 0 <= nc < 21 and self.memory_map[nr, nc] == 0:
                # Get further away from Pacman's last known location
                dist_from_pacman = abs(nr - self.last_known_pacman[0]) + abs(nc - self.last_known_pacman[1])
                
                # Count valid moves to find intersections
                valid_moves = 0
                for next_m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    r2, c2 = nr + next_m.value[0], nc + next_m.value[1]
                    if 0 <= r2 < 21 and 0 <= c2 < 21 and self.memory_map[r2, c2] in [0, -1]:
                        valid_moves += 1
                
                score = (dist_from_pacman * 10)
                
                # Got a bonus when we hit the intersection
                if valid_moves >= 3:
                    score += 50
                    
                # Small penalty for backtracking
                if self.last_pos is not None and (nr, nc) == self.last_pos:
                    score -= 100
                    
                if score > best_score:
                    best_score = score
                    best_move = move
                    
        return best_move

    def _execute_hide(self, my_pos: Tuple[int, int]) -> Move:
        """
        Active Hide: Seek the deep corners of the map! 
        Avoids straight hallways, defuses dead ends, and creeps away from the center.
        """
        structural_moves = [] 
        safe_moves = []       
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = my_pos[0] + move.value[0], my_pos[1] + move.value[1]
            if 0 <= nr < 21 and 0 <= nc < 21:
                if self.memory_map[nr, nc] in [0, -1]:
                    structural_moves.append(move)
                if self.memory_map[nr, nc] == 0:
                    safe_moves.append(move)
                    
        # Check if we are in a straight kill-zone
        is_straight_hallway = False
        if len(structural_moves) == 2:
            m1, m2 = structural_moves
            if m1.value[0] + m2.value[0] == 0 and m1.value[1] + m2.value[1] == 0:
                is_straight_hallway = True
                
        # Calculate how far we are from the exact center of the map (10, 10)
        dist_from_center = abs(my_pos[0] - 10) + abs(my_pos[1] - 10)
        
        # If we are NOT in a straight hallway, and NOT in a dead end...
        if not is_straight_hallway and len(structural_moves) >= 2:
            # AND we are deep in the outer perimeter of the map (Threshold: 12+ tiles away)
            if dist_from_center >= 12:
                # We have reached the ultimate hiding spot
                return Move.STAY

        # We need to keep moving (either to escape the center, a hallway, or a dead end)
        best_move = Move.STAY
        best_score = -float('inf')
        
        for move in safe_moves:
            nr, nc = my_pos[0] + move.value[0], my_pos[1] + move.value[1]
            
            # Prevent vibrating back and forth (unless we are in a dead end and MUST turn around)
            if len(safe_moves) > 1 and self.last_pos is not None and (nr, nc) == self.last_pos:
                continue
                
            # Score this move based on how far it pushes us into the map corners!
            score = abs(nr - 10) + abs(nc - 10)
            
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move

    def _fallback_flee(self, my_pos: Tuple[int, int], pacman_pos: Tuple[int, int]) -> Move:
        """
        If we can't break Line of Sight, run away through confirmed safe paths.
        """
        best_move = Move.STAY
        best_dist = -1
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = my_pos[0] + move.value[0], my_pos[1] + move.value[1]
            if 0 <= nr < 21 and 0 <= nc < 21:
                if self.memory_map[nr, nc] == 0:
                    dist = abs(nr - pacman_pos[0]) + abs(nc - pacman_pos[1])
                    if dist > best_dist:
                        # Prevent 180-degree backtracking into Pacman
                        if self.last_pos is None or (nr, nc) != self.last_pos:
                            best_dist = dist
                            best_move = move
                            
        return best_move


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost

    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # TODO: Initialize any data structures you need
        # Examples:
        # - self.path = []  # Store planned path
        # - self.visited = set()  # Track visited positions
        self.name = "Heatseeking Pacman"
        self.heatmap = np.ones((21, 21), float)
        self.kernel = ((0.0, 0.2, 0.0),
                       (0.2, 0.2, 0.2),
                       (0.0, 0.2, 0.0),)
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
        self.current_target = None
        self.visit_count = np.zeros((21, 21), dtype=int)

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int):
        """
        Decide the next move.

        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Ghost's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)

        Returns:
            Move or (Move, steps): Direction to move (optionally with step count)
        """
        # TODO: Implement your search algorithm here
        self.visit_count[my_position] += 1

        visible_tiles = np.argwhere(map_state == 0)
        visible_candidates = tuple(tuple(row) for row in visible_tiles.tolist())
        gateway_candidates = self.get_gateways(my_position, visible_candidates, map_state)

        # Update heatmap
        self.update_heatmap(map_state, enemy_position)

        # Choose target
        max_heat = np.max(self.heatmap)
        # if enemy_position:
        #     self.current_target = enemy_position
        # elif self.current_target is not None and self.current_target != my_position:
        #     self.current_target = self.current_target
        if max_heat >= 0.001:
            candidates = np.argwhere(self.heatmap >= max_heat * 0.9)
            general_target = tuple(candidates[0].tolist())
            # Choose visible tile closest to max heat
            self.current_target = min(visible_candidates, key=lambda c: self._manhattan_distance(c, general_target))
        else:
            # Frontier exploring
            non_wall_tiles = np.argwhere(map_state != 1)
            min_visits = np.min(self.visit_count[map_state != 1])
            least_visited = [tuple(t) for t in non_wall_tiles if self.visit_count[t[0], t[1]] == min_visits]
            global_fallback = min(least_visited, key=lambda c: self._manhattan_distance(c, my_position))
            self.current_target = min(gateway_candidates,
                                      key=lambda c: self._manhattan_distance(c, global_fallback) + self.visit_count[
                                          c] * 5)

        path = self.a_star(my_position, self.current_target, map_state, False)

        # Path processing
        move = Move.STAY
        step = 1

        if len(path) > 1:
            step1 = path[1]

            if step1[0] == my_position[0]:
                if step1[1] < my_position[1]:
                    move = Move.LEFT
                elif step1[1] > my_position[1]:
                    move = Move.RIGHT
            elif step1[1] == my_position[1]:
                if step1[0] < my_position[0]:
                    move = Move.UP
                elif step1[0] > my_position[0]:
                    move = Move.DOWN

            if len(path) > 2:
                step2 = path[2]
                if (step1[0] == my_position[0] and step2[0] == step1[0]) or (
                        step1[1] == my_position[1] and step2[1] == step1[1]):
                    step = 2
                    self.visit_count[step1] += 1

        return (move, step)

    def update_heatmap(self, map_state: np.ndarray, enemy_position: tuple | None):
        # Clear heat from walls
        self.heatmap[map_state == 1] = 0.0

        if enemy_position:
            # Collapse heat into enemy if seen
            self.heatmap.fill(0.0)
            self.heatmap[enemy_position] = 1.0
        else:
            # Decay
            self.heatmap *= 0.9
            # Diffusion
            P = np.pad(self.heatmap, 1, mode='constant', constant_values=0)
            self.heatmap = 0.2 * (
                    P[1:-1, 1:-1] + P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:]
            )
            # Clear heat from seen paths and walls
            visible_mask = map_state != -1
            self.heatmap[visible_mask] = 0.0

    def a_star(self, start_pos: tuple, end_pos: tuple, map_state: np.ndarray, include_fog: bool):
        frontier_heap = []  # Priority queue (heapq)
        frontier = set()  # Sort of a tracker for the frontier. Needed because we use lazy deletion when updating the heapq.
        explored = set()

        # Track parents and costs using dictionaries (i don't know if we're allowed to add a Node class)
        parent = {start_pos: None}
        g_cost = {start_pos: 0}
        h_cost = {start_pos: self._manhattan_distance(start_pos, end_pos)}
        f_cost = {start_pos: g_cost[start_pos] + h_cost[start_pos]}

        # Push start pos into frontier. Use tuple for priority order (f-cost -> h-cost -> coordinate itself as fallback)
        heapq.heappush(frontier_heap, (f_cost[start_pos], h_cost[start_pos], start_pos))
        frontier.add(start_pos)

        # Loop through frontier
        iterations = 0
        while frontier:
            current_node = heapq.heappop(frontier_heap)[2]

            # Handle old duplicates left behind by lazy deletion.
            if current_node not in frontier:
                continue

            # Found target or reached max iterations
            if current_node == end_pos or iterations >= 128:
                path = []
                while current_node in parent:
                    path.append(current_node)
                    current_node = parent[current_node]
                path.reverse()
                return path

            # Move node to explored
            explored.add(current_node)
            frontier.remove(current_node)

            # Process neighbors
            neighbors = self._get_neighbors(current_node, map_state, include_fog)
            for neighbor in neighbors:
                if neighbor in explored:
                    continue

                new_g_cost = g_cost[current_node] + 1
                if neighbor not in frontier:
                    h_cost[neighbor] = self._manhattan_distance(neighbor, end_pos)
                    frontier.add(neighbor)
                    g_cost[neighbor] = 1024  # Placeholder g-cost for the second if

                if new_g_cost < g_cost[neighbor]:
                    parent[neighbor] = current_node
                    g_cost[neighbor] = new_g_cost
                    f_cost[neighbor] = g_cost[neighbor] + h_cost[neighbor]
                    heapq.heappush(frontier_heap, (f_cost[neighbor], h_cost[neighbor], neighbor))

            iterations += 1

        # If no path is found
        print("PACMAN A* FAILED")
        return []

    def get_gateways(self, my_pos: tuple, visible_tiles: tuple, map_state: np.ndarray):
        gateways = []
        # Choose tiles that border the fog
        for pos in visible_tiles:
            borders_fog = False
            for neighbor in self._get_neighbors(pos, map_state, include_fog=True):
                if map_state[neighbor] == -1:
                    borders_fog = True
                    break
            if borders_fog:
                gateways.append(pos)

        # If no fog, choose furthest visible tiles
        if len(gateways) == 0:
            for pos in visible_tiles:
                edge = True
                for neighbor in self._get_neighbors(pos, map_state, include_fog=False):
                    if self._manhattan_distance(my_pos, neighbor) > self._manhattan_distance(my_pos, pos):
                        edge = False
                        break
                if edge:
                    gateways.append(pos)

        return gateways

    # Helper methods
    def _apply_move(self, pos, move):
        """Apply a move to a position, return new position."""
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)

    def _get_neighbors(self, pos, map_state, include_fog):
        """Get all valid neighboring positions and their moves."""
        neighbors = []

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state, include_fog):
                neighbors.append(next_pos)

        return neighbors

    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray, include_fog: bool) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape

        if row < 0 or row >= height or col < 0 or col >= width:
            return False

        if include_fog:
            return map_state[row, col] != 1
        else:
            return map_state[row, col] == 0
