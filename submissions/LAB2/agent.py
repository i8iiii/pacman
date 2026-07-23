import sys
from pathlib import Path
from heapq import heappush, heappop
import random
import numpy as np

# Add src to path to import framework classes
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
from hide_agent.controller import HideController


class GhostAgent(BaseGhostAgent):
    """Arena entry point for the phased Hide agent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Hide-Agent"
        self._controller = HideController(
            log_path=kwargs.get("log_path"),
            map_text_path=kwargs.get("map_text_path"),
            map_jsonl_path=kwargs.get("map_jsonl_path"),
            diagnostics_enabled=kwargs.get("diagnostics_enabled"),
            pacman_speed=kwargs.get("pacman_speed", 2),
            capture_distance=kwargs.get("capture_distance", 2),
            observation_radius=kwargs.get("observation_radius", 5),
        )

    def step(self, map_state, my_position, enemy_position, step_number):
        return self._controller.step(map_state, my_position, enemy_position, step_number)


# ============================================================
# PACMAN HELPER FUNCTIONS
# ============================================================

def _pacman_is_valid_position(pos, map_state):
    """Check if position is valid on internal map (not wall, within bounds)."""
    row, col = pos
    height, width = map_state.shape
    if row < 0 or row >= height or col < 0 or col >= width:
        return False
    return map_state[row, col] != 1


def _pacman_get_neighbors(pos, map_state):
    """Get valid neighboring positions and their moves."""
    neighbors = []
    for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
        dr, dc = move.value
        next_pos = (pos[0] + dr, pos[1] + dc)
        if _pacman_is_valid_position(next_pos, map_state):
            neighbors.append((next_pos, move))
    return neighbors


def _pacman_manhattan(a, b):
    """Manhattan distance between two positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _pacman_astar(start, goal, map_state):
    """A* pathfinding from start to goal. Returns list of Move enums, or empty list if no path."""
    if start == goal:
        return []

    # Pre-validate that goal is reachable (BFS check) to avoid wasted search
    open_set = [(0, 0, start, [])]  # (f_score, counter, pos, path_of_moves)
    g_score = {start: 0}
    closed_set = set()
    counter = 0

    while open_set:
        _, _, current, path = heappop(open_set)

        if current in closed_set:
            continue
        closed_set.add(current)

        if current == goal:
            return path

        for next_pos, move in _pacman_get_neighbors(current, map_state):
            if next_pos in closed_set:
                continue

            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(next_pos, float('inf')):
                g_score[next_pos] = tentative_g
                h = _pacman_manhattan(next_pos, goal)
                counter += 1
                heappush(open_set, (tentative_g + h, counter, next_pos, path + [move]))

    return []  # No path found


def _pacman_find_frontier(my_pos, internal_map):
    """Find the best frontier cell (boundary between known and unknown) for exploration."""
    if internal_map is None:
        return None
    h, w = internal_map.shape
    best = None
    best_score = -1
    for r in range(h):
        for c in range(w):
            if internal_map[r, c] != 0:
                continue
            # Check if this cell borders an unseen cell
            has_unknown = False
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and internal_map[nr, nc] == -1:
                    has_unknown = True
                    break
            if has_unknown:
                dist = _pacman_manhattan(my_pos, (r, c))
                if dist == 0:
                    continue
                score = 1.0 / dist
                if score > best_score:
                    best_score = score
                    best = (r, c)
    return best


# ============================================================
# PACMAN AGENT — DQN primary + A* fallback
# ============================================================

# Attempt to import torch and model; gracefully degrade if unavailable
try:
    import torch
    from model import PacmanCNN
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    PacmanCNN = None


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) agent with CNN-DQN primary + A* fallback.

    Decision flow:
    1. If enemy visible and DQN confident → use DQN move
    2. If enemy visible but DQN uncertain → A* to enemy
    3. If enemy hidden but recently seen → A* to last known position
    4. If enemy hidden too long → A* to frontier (explore)
    """

    CONFIDENCE_THRESHOLD = 0.5
    ALL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get('pacman_speed', 1)))
        self.name = "Pacman-1.2-DQN"

        # ── State tracking ──
        self.internal_map = None
        self.map_initialized = False
        self.last_known_enemy_pos = None
        self.steps_since_seen = 0
        self.last_move = None

        # ── Load DQN model ──
        self.device = torch.device("cpu") if TORCH_AVAILABLE else None
        self.model = None
        if TORCH_AVAILABLE and PacmanCNN is not None:
            try:
                self.model = PacmanCNN()
                current_dir = Path(__file__).parent
                # Try multiple possible model file names
                for model_name in ["pacman_dqn.pt", "best_pacman_dqn.pt"]:
                    model_path = current_dir / model_name
                    if model_path.exists():
                        self.model.load_state_dict(
                            torch.load(model_path, map_location=self.device, weights_only=True)
                        )
                        self.model.eval()
                        break
                else:
                    # No trained weights found — DQN will return None
                    self.model = None
            except Exception:
                self.model = None

    def step(self, map_state, my_position, enemy_position, step_number):
        # 1. Update internal map memory
        self._update_map_memory(map_state)

        # 2. Track enemy visibility
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.steps_since_seen = 0
        else:
            self.steps_since_seen += 1

        # 3. Choose move
        chosen_move = Move.STAY
        path = None

        if enemy_position is not None:
            # Enemy visible — try DQN first
            dqn_move = self._get_dqn_move(map_state, my_position, enemy_position)
            if dqn_move is not None:
                chosen_move = dqn_move
            else:
                # DQN not confident or unavailable — A* to enemy
                path = _pacman_astar(my_position, enemy_position, self.internal_map)
                if path:
                    chosen_move = path[0]
                else:
                    chosen_move = self._greedy_toward(my_position, enemy_position)
        elif self.last_known_enemy_pos is not None and self.steps_since_seen <= 10:
            # Lost sight recently — A* to last known position
            path = _pacman_astar(my_position, self.last_known_enemy_pos, self.internal_map)
            if path:
                chosen_move = path[0]
            else:
                chosen_move = self._greedy_toward(my_position, self.last_known_enemy_pos)
        else:
            # Lost sight too long — explore frontier
            frontier = _pacman_find_frontier(my_position, self.internal_map)
            if frontier:
                path = _pacman_astar(my_position, frontier, self.internal_map)
                if path:
                    chosen_move = path[0]
            if chosen_move == Move.STAY:
                chosen_move = self._random_valid_move(my_position)

        # 4. Compute speed steps
        steps = self._compute_steps(my_position, chosen_move, path)

        self.last_move = chosen_move
        return (chosen_move, steps)

    # ── DQN Inference ──

    def _get_dqn_move(self, map_state, my_pos, enemy_pos):
        """Run DQN forward pass. Returns Move if confident, None otherwise."""
        if self.model is None or not TORCH_AVAILABLE:
            return None

        try:
            # Encode state
            input_map = self.internal_map.copy().astype(np.float32)
            input_map[my_pos] = 2.0
            input_map[enemy_pos] = 3.0

            state_tensor = torch.FloatTensor(input_map).unsqueeze(0).unsqueeze(0).to(self.device)

            # Encode last move
            last_move_vec = np.zeros(4, dtype=np.float32)
            if self.last_move in self.ALL_MOVES:
                last_move_vec[self.ALL_MOVES.index(self.last_move)] = 1.0
            move_tensor = torch.FloatTensor(last_move_vec).unsqueeze(0).to(self.device)

            # Forward pass
            with torch.no_grad():
                q_values = self.model(state_tensor, move_tensor).squeeze(0)  # [4]

            # Confidence check
            confidence = q_values.max().item() - q_values.mean().item()
            if confidence < self.CONFIDENCE_THRESHOLD:
                return None

            # Get best valid move
            best_idx = q_values.argmax().item()
            predicted_move = self.ALL_MOVES[best_idx]

            # Validate: must not move into wall
            if self._can_move(my_pos, predicted_move):
                return predicted_move

        except Exception:
            pass

        return None

    # ── Map Memory ──

    def _update_map_memory(self, map_state):
        """Build and maintain internal map by merging fog observations over time."""
        if not self.map_initialized:
            self.internal_map = np.full_like(map_state, -1)
            self.internal_map[map_state == 1] = 1
            self.map_initialized = True

        # Merge visible cells into memory
        visible_mask = map_state != -1
        self.internal_map[visible_mask] = map_state[visible_mask]

        # Restore agent position markers back to empty (they move)
        self.internal_map[(self.internal_map == 2) | (self.internal_map == 3)] = 0

    # ── Movement Helpers ──

    def _greedy_toward(self, my_pos, target):
        """Move greedily toward target using Manhattan distance."""
        best_move = Move.STAY
        best_dist = _pacman_manhattan(my_pos, target)
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            nxt = (my_pos[0] + dr, my_pos[1] + dc)
            if self._is_valid_on_internal(nxt):
                d = _pacman_manhattan(nxt, target)
                if d < best_dist:
                    best_dist = d
                    best_move = move
        return best_move

    def _random_valid_move(self, my_pos):
        """Pick a random valid move."""
        moves = [mv for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
                 if self._can_move(my_pos, mv)]
        return random.choice(moves) if moves else Move.STAY

    def _can_move(self, pos, move):
        """Check if a single step in the given direction is valid."""
        dr, dc = move.value
        nxt = (pos[0] + dr, pos[1] + dc)
        return self._is_valid_on_internal(nxt)

    def _is_valid_on_internal(self, pos):
        """Check if position is valid on internal map."""
        if self.internal_map is None:
            return False
        r, c = pos
        h, w = self.internal_map.shape
        return 0 <= r < h and 0 <= c < w and self.internal_map[r, c] != 1

    def _compute_steps(self, my_pos, chosen_move, path):
        """Compute how many speed steps to take (1 or 2) based on path straightness."""
        steps = 1
        if chosen_move != Move.STAY and self.pacman_speed >= 2:
            # Only use speed=2 if the path is straight for 2+ tiles
            can_move_2 = self._can_move_n(my_pos, chosen_move, 2)

            # Check path doesn't turn at step 2 (no overshooting corners)
            if path and len(path) >= 2 and path[0] == chosen_move and path[1] != chosen_move:
                can_move_2 = False

            if can_move_2:
                steps = 2
        return steps

    def _can_move_n(self, pos, move, n):
        """Check if we can move n steps in a straight line."""
        r, c = pos
        dr, dc = move.value
        for i in range(1, n + 1):
            nr, nc = r + dr * i, c + dc * i
            if not self._is_valid_on_internal((nr, nc)):
                return False
        return True
