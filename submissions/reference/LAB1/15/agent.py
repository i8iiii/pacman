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
from enum import Enum
from pathlib import Path


# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import helper

"""
    Group ID: 15
    Group Name: AIO
    Members:
    + Hoàng Minh Huy - 24127278
    + Nguyễn Chí Minh - 24127450
    + Nguyễn Xuân Quyên - 24127302
"""

"""
    - Algorithm: We decided to combine Alpha-Beta Pruning and heuristic search.
    - This algorithm is applied to both the seek agent and the hide agent.
    - The reason we chose this algorithm is because:
        + In this state of project, fog of war is not applied -> Perfect Information
        + In this state, Minimax algorithm can deduce and return the best move for both agents.
        -> But this map is 21x21, minimax will run forever.
        -> We add a limit (depth) to how many moves Minimax can predict.
        -> If Minimax still can't define a move after reaching limit, we calculate a move heuristically
        -> Then we choose the best move based on Minimax and Heuristic function.
        -> Still slow, so we upgrade Minimax into Alpha-Beta Pruning 
"""

"""
    - Details about the algorithm we implemented:
        + State: my_position, enemy_position, step
        + Action: 
            * pacman: (Move.UP, 1), (Move.UP, 2), ...
            * ghost: (Move.DOWN), (Move.LEFT), ...
        + Role: pacman will try to maximize the score, ghost will try to minimize it.
"""

class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.max_depth = 4
        self.name = "Template Pacman"
        self.last_known_enemy_pos = None

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int):

        # Our Alpha-Beta Pruning

        if enemy_position is None:
            return Move.STAY

        # Initialize necessary variables
        v = -float('inf')      # Keep track of the max score
        move = None            # The action which is bounded to the max score
        alpha = -float('inf')  # Keep track of the maximum node of a level
        beta = float('inf')    # Keep track of the minimum node of a level

        # A loop through every action of pacman
        for action in self._seek_actions(my_position, map_state):
            next_pos = self._result(my_position, action) # Transition model
            # Perform recursive loops to find the best move.
            # pacman calls min_value: Predicts ghost next move to make a clever move.
            score = self.min_value(next_pos, enemy_position, step_number + 1, map_state, 1, alpha, beta)
            if score > v:
                v = score       # The highest score.
                move = action   # The action that goes with it.
        return move if move is not None else (Move.STAY, 0)

    # Helper methods

    # Return a set of every possible moves for pacman
    def _seek_actions(self, pos, map_state: np.ndarray):
        actions = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            # Find the maximum clear straight line distance
            max_steps = self._max_valid_steps(pos, move, map_state, self.pacman_speed)

            # Allow the agent to evaluate taking 1 step OR taking 2 steps (if valid)
            for steps in range(1, max_steps + 1):
                actions.append((move, steps))

        return actions # Example output: [(Move.UP, 2),...]

    # Return a set of every possible moves for ghost
    def _hide_action(self, my_position, map_state: np.ndarray):
        ghost_action = [(Move.STAY, 0)]
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_position, move, map_state):
                ghost_action.append((move, 1)) # Step is always 1
        return ghost_action # Example output: [(Move.UP, 1), ...]

    # Check if the game ends
    def _terminal(self, my_position, enemy_position, steps):
        if helper._manhattan(my_position, enemy_position) < 2:
            return True
        if steps > 200:
            return True
        return False

    # The transition model
    def _result(self, my_position, action):
        move, step = action
        delta_row, delta_col = move.value
        return my_position[0] + delta_row * step, my_position[1] + delta_col * step

    # Gives out scores for the agents
    def _utility(self, my_position, enemy_position, steps, depth):
        if helper._manhattan(my_position, enemy_position) < 2:
            return 1000 - depth # The longer pacman finds the ghost, the lower the score is
        return -1000            # Every step the ghost survives, drag the point down

    """
    ---------------------------------- IMPORTANT -------------------------------------
    """

    def max_value(self, my_position, enemy_position, steps, map_state, depth, alpha, beta):
        # If any transition model reaches the terminal state, gives out the scores
        if self._terminal(my_position, enemy_position, steps):
            return self._utility(my_position, enemy_position, steps, depth)

        # If depth is reached before terminal, calculate heuristically instead
        if depth > self.max_depth:
            return helper._evaluation_heuristic(my_position, enemy_position, map_state)

        v = -float('inf') # Keeps track of the highest score
        for action in self._seek_actions(my_position, map_state):
            # Simulate pacman's moves.
            next_pos = self._result(my_position, action)
            v = max(v, self.min_value(next_pos, enemy_position, steps + 1, map_state, depth + 1, alpha, beta))

            # If we know that this moves is higher than Beta
            # Which means in the next loop, the opponent is already having a better move already.
            # We prune the rest away (Stop the recursive loop).

            if v >= beta: return v
            alpha = max(alpha, v)
        return v

    def min_value(self, my_position, enemy_position, steps, map_state, depth, alpha, beta):
        v = float('inf') # Keep track of the lowest score.
        for action in self._hide_action(enemy_position, map_state):
            next_enemy_pos = self._result(enemy_position, action)

            """
            - In this condition, we need to evaluate both our move and the enemy as well.
            - From here, we encountered the guarding situation:
               + Instead of catching the ghost, pacman always runs behind (Guarding) it.
               + This happens because ghost and pacman moves at the same time.
               + When pacman is right behind the ghost, it treats the next block as the terminal state (High score).
               + Yet the ghost moves at the same time as well, so it will run away from pacman.
               + While pacman still treats the block where ghost used to stay as the terminal state => Guarding situation.

            - To Counter this, we need pacman to be more clever, for pacman to predict the ghost ACTUAL spot
            - So we threw both the terminal and depth check inside the loop, for pacman to locate the ghost.
            """

            if self._terminal(my_position, next_enemy_pos, steps + 1):
                # min_value is called by max_value, so the prediction happens here.
                score = self._utility(my_position, next_enemy_pos, steps + 1, depth)
            elif depth > self.max_depth:
                score = helper._evaluation_heuristic(my_position, next_enemy_pos, map_state)
            else:
                # Predicting ghost's moves
                score = self.max_value(my_position, next_enemy_pos, steps + 1, map_state, depth + 1, alpha, beta)

            v = min(v, score) # Chooses the best action for the ghost.
            # If we know that this moves is lower than Alpha
            # Which means in the next loop, the opponent is already having a better move already.
            # We prune the rest away (Stop the recursive loop).
            if v <= alpha: return v
            beta = min(beta, v)
        return v

    """
    --------------------------------- ///// -------------------------------------
    """

    def _choose_action(self, pos: tuple, moves, map_state: np.ndarray, desired_steps: int):
        for move in moves:
            max_steps = min(self.pacman_speed, max(1, desired_steps))
            steps = self._max_valid_steps(pos, move, map_state, max_steps)
            if steps > 0:
                return (move, steps)
        return None

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:
        steps = 0
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps

    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid for at least one step."""
        return self._max_valid_steps(pos, move, map_state, 1) == 1

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape

        if row < 0 or row >= height or col < 0 or col >= width:
            return False

        return map_state[row, col] == 0



class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught by minimizing Pacman's performance metrics.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_depth = 4
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Simultaneous_Minimax_Ghost"
        self.last_known_enemy_pos = None

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int) -> Move:

        if enemy_position is None:
            return Move.STAY

        # Same with pacman
        v = float('inf')
        best_move = Move.STAY
        alpha = -float('inf')
        beta = float('inf')

        # This time, my position is the ghost, and enemy is pacman
        for action in self._hide_action(my_position, map_state):
            next_hide_pos = self._result(my_position, action)
            # Ghost must predict enemy's counter move with every move it makes.
            score = self.max_value(next_hide_pos, enemy_position, step_number + 1, map_state, 1, alpha, beta)

            if score < v:
                v = score
                best_move = action[0]  # Return the clean Move enum instead of a tuple layout
        return best_move

    # Return a list of every possible moves for ghost
    def _hide_action(self, my_position, map_state: np.ndarray):
        actions = [(Move.STAY, 0)]
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_position, move, map_state):
                actions.append((move, 1)) # Ghost always moves with 1 step
        return actions

    # Return a list of every possible moves for pacman
    def _seek_actions(self, pos, map_state):
        actions = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            current = pos
            for steps in range(1, self.pacman_speed + 1):
                delta_row, delta_col = move.value
                next_pos = (current[0] + delta_row, current[1] + delta_col)
                if not self._is_valid_position(next_pos, map_state):
                    break
                actions.append((move, steps))
                current = next_pos
        return actions

    # Check if the game ends
    def _terminal(self, my_position, enemy_position, steps):
        if helper._manhattan(my_position, enemy_position) < 2:
            return True
        if steps > 200:
            return True
        return False

    # The transition model, but this time my_position is the ghost.
    def _result(self, my_position, action):
        move, step = action
        delta_row, delta_col = move.value
        return (my_position[0] + delta_row * step, my_position[1] + delta_col * step)

    # Gives out score (Same with pacman)
    def _utility(self, my_position, enemy_position, steps, depth):
        if helper._manhattan(my_position, enemy_position) < 2:
            return 1000 - depth
        return -1000

    # Same with pacman, but this time my_position is the ghost.
    def max_value(self, my_position, enemy_position, steps, map_state, depth, alpha, beta):

        # If any transition model reaches the terminal state, gives out the scores
        if self._terminal(my_position, enemy_position, steps):
            return self._utility(my_position, enemy_position, steps, depth)

        # If depth is reached before terminal, calculate heuristically instead
        if depth > self.max_depth:
            return helper._evaluation_heuristic(my_position, enemy_position, map_state)

        v = -float('inf')
        for action in self._seek_actions(enemy_position, map_state):
            # Predicting pacman's move
            next_enemy_pos = self._result(enemy_position, action)
            v = max(v, self.min_value(my_position, next_enemy_pos, steps + 1, map_state, depth + 1, alpha, beta))

            # If we know that this moves is higher than Beta
            # Which means in the next loop, the opponent is already having a better move already.
            # We prune the rest away (Stop the recursive loop).

            if v >= beta: return v
            alpha = max(alpha, v)
        return v

    def min_value(self, my_position, enemy_position, steps, map_state, depth, alpha, beta):
        v = float('inf')
        for action in self._hide_action(my_position, map_state):
            next_hide_pos = self._result(my_position, action)

            # Same with pacman agent's min_value, but this time my_position = ghost's position.
            # We also prevent the guarding situation here, read pacman's agent comments for more details.

            if self._terminal(next_hide_pos, enemy_position, steps + 1):
                score = self._utility(next_hide_pos, enemy_position, steps + 1, depth)
            elif depth > self.max_depth:
                score = helper._evaluation_heuristic(next_hide_pos, enemy_position, map_state)
            else:
                score = self.max_value(next_hide_pos, enemy_position, steps + 1, map_state, depth + 1, alpha, beta)

            v = min(v, score) # Pick best move.

            # If we know that this moves is lower than Alpha
            # Which means in the next loop, the opponent is already having a better move already.
            # We prune the rest away (Stop the recursive loop).

            if v <= alpha: return v
            beta = min(beta, v)
        return v

    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0


