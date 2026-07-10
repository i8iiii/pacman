"""
CLI:
cd src
python arena.py --seek 24127108_24127329_24127142 --hide 24127108_24127329_24127142

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
from queue import Queue

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np

import math


class Node:
    def __init__(self, parent, action: Move, state: tuple[int, int]):
        self.parent = parent
        self.action = action
        self.state = state

class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost

    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.map_state: np.ndarray = np.ndarray(2)

        self.name = "EightEye Pacman"

        self.available_path: list[Node] = []


        self.last_known_enemy_pos = None

    @classmethod
    def contain_state(cls, state, frontier):
        return any(state == s for s in list(frontier.queue))


    def bfs(self, my_position, enemy_position) -> list[Node]:
        """
        Return a list of nodes that represent the shortest path from my_position to enemy_position
        """
        frontier = Queue()

        initial_node = Node(
            parent=None,
            action=None,
            state=my_position,
        )

        frontier.put(initial_node)

        # List of state
        explored: list[tuple[int, int]] = []

        while True:
            if frontier.empty():
                return []

            node = frontier.get()

            if node.state == enemy_position:
                path = []

                while node.parent is not None:
                    path.append(node)
                    node = node.parent

                path.reverse()

                return path

            explored.append(node.state)

            # Create and evaluate available tiles
            up_tile = (node.state[0] - 1, node.state[1])
            if self._is_valid_position(up_tile, self.map_state) and not self.contain_state(state=up_tile, frontier=frontier) and up_tile not in explored:
                new_node = Node(
                    parent=node,
                    action=Move.UP,
                    state=up_tile
                )
                frontier.put(new_node)

            down_tile = (node.state[0] + 1, node.state[1])
            if self._is_valid_position(down_tile, self.map_state) and not self.contain_state(state=down_tile, frontier=frontier) and down_tile not in explored:
                new_node = Node(
                    parent=node,
                    action=Move.DOWN,
                    state=down_tile
                )
                frontier.put(new_node)

            left_tile = (node.state[0], node.state[1] - 1)
            if self._is_valid_position(left_tile, self.map_state) and not self.contain_state(state=left_tile, frontier=frontier) and left_tile not in explored:
                new_node = Node(
                    parent=node,
                    action=Move.LEFT,
                    state=left_tile
                )
                frontier.put(new_node)

            right_tile = (node.state[0], node.state[1] + 1)
            if self._is_valid_position(right_tile, self.map_state) and not self.contain_state(state=right_tile, frontier=frontier) and right_tile not in explored:
                new_node = Node(
                    parent=node,
                    action=Move.RIGHT,
                    state=right_tile
                )
                frontier.put(new_node)

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
        self.map_state = map_state

        list_of_action = self.bfs(my_position, enemy_position)
        new_list_of_action = self._append_step(list_of_action)
        self.available_path = list_of_action


        if len(new_list_of_action) > 0:
            action = new_list_of_action[0]
            return action

        return Move.STAY, 1

    # Helper methods
    def _append_step(self, list_of_action: list[Node]):
        """
        Convert same move to step
        """
        res: list[tuple[Move, int]] = []

        cur_move = list_of_action[0].action
        cur_step = 0
        for action in list_of_action:
            if cur_step == self.pacman_speed or cur_move != action.action:
                res.append((cur_move, cur_step))
                cur_move = action.action
                cur_step = 1

            elif cur_move == action.action:
                cur_step += 1


        # Last move
        res.append((cur_move, cur_step))
        return res

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape

        if row < 0 or row >= height or col < 0 or col >= width:
            return False

        return map_state[row, col] == 0


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught

    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Initialize any data structures you need
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
        self.maxDepth = 10

    def step(self, map_state: np.ndarray,   #Minmax
             my_position: tuple,
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Decide the next move.

        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Pacman's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)

        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        # TODO: Implement your search algorithm here

        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        # Use current sighting, fallback to last known, or move randomly
        threat = enemy_position or self.last_known_enemy_pos

        if threat is None:
            # No information about enemy - move randomly
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    return move
            return Move.STAY

        best_move = Move.STAY
        best_val = -999
        alpha = -999
        beta = 999

        for move in [Move.UP, Move.DOWN, Move.RIGHT, Move.LEFT]:
            if not self._is_valid_move(my_position, move, map_state):
                continue

            next_my_pos = self.updatePos(my_position, move)

            val = self.minValue(map_state, next_my_pos, threat, 1, alpha, beta)

            if val > best_val:
                best_val = val
                best_move = move

            # elif val == best_val:
            
            alpha = max(alpha, best_val)
        return best_move

        # Example: Simple evasive approach (replace with your algorithm)
        # row_diff = my_position[0] - threat[0]
        # col_diff = my_position[1] - threat[1]

        # # Try to move away from Pacman
        # if abs(row_diff) > abs(col_diff):
        #     move = Move.DOWN if row_diff > 0 else Move.UP
        # else:
        #     move = Move.RIGHT if col_diff > 0 else Move.LEFT

        # # Check if move is valid
        # if self._is_valid_move(my_position, move, map_state):
        #     return move

        # # If not valid, try other moves
        # for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
        #     if self._is_valid_move(my_position, move, map_state):
        #         return move

        # return Move.STAY

    # Helper methods (you can add more)

    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape

        if row < 0 or row >= height or col < 0 or col >= width:
            return False

        return map_state[row, col] == 0
    
    def pac_max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:
        steps = 0
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self.pac_is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps

    def pac_is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid for at least one step."""
        return self.pac_max_valid_steps(pos, move, map_state, 1) == 1

    def pac_is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape

        if row < 0 or row >= height or col < 0 or col >= width:
            return False

        return map_state[row, col] == 0

    def minValue(self, map: np.ndarray, myPos : tuple, threatPos: tuple, depth : int, alpha : int, beta):
        """
        Pac value
        """
        if self.terminal(myPos, threatPos) or depth == self.maxDepth:
            return self.utility(myPos, threatPos, map, depth)

        val = 999

        avai_actions = set()

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self.pac_is_valid_move(threatPos, move, map):
                maxSteps = self.pac_max_valid_steps(threatPos, move, map, 2)

                for i in range(1, maxSteps + 1, 1):
                    avai_actions.add((move, i))

        for move in avai_actions:
            val = min(val, self.maxValue(map, myPos, self.updatePos(threatPos, move[0], move[1]), depth + 1, alpha, beta))
            if val <= alpha:
                return val

            beta = min(beta, val)

        return val
    
    def maxValue(self, map: np.ndarray, myPos : tuple, threatPos: tuple, depth : int, alpha: int, beta : int):
        """
        Ghost value
        """
        if self.terminal(myPos, threatPos) or depth == self.maxDepth:
            return self.utility(myPos, threatPos, map, depth)   # Not GameOver
        
        val = -999
        avai_actions = set()
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(myPos, move, map):
                avai_actions.add(move)

        for move in avai_actions:
            val = max(val, self.minValue(map, self.updatePos(myPos, move), threatPos, depth + 1, alpha, beta))

            if val >= beta:
                break

            alpha = max(alpha, val)
        
        return val

    def manHatDis(self, myPos : tuple, threatPos: tuple):
        return abs(myPos[0] - threatPos[0]) + abs(myPos[1] - threatPos[1])

    def updatePos(self, pos : tuple, move : Move, steps = 1):
        x, y = pos
        match move:
            case Move.UP:
                x -= 1 * steps
            case Move.DOWN:
                x += 1 * steps
            case Move.LEFT: 
                y -= 1 * steps
            case Move.RIGHT:
                y += 1 * steps
        return x, y
    
    def utility(self, ghostPos, pacPos, map, depth):
        dis = self.manHatDis(ghostPos, pacPos)

        if dis < 2:    # Pacman wins
            return -2 * self.maxDepth + depth
        
        turnsToCapture = None
        escapeBonus = 0
        
        if ghostPos[0] == pacPos[0] and self.checkWall(ghostPos, pacPos, map, True):
            turnsToCapture = math.ceil(dis / 2)

            # if self._is_valid_position((ghostPos[0] + 1, ghostPos[1]), map):    # Check if there are escapes Ghost can run into
            #     escapeBonus += 2
            # if self._is_valid_position((ghostPos[0] - 1, ghostPos[1]), map):
            #     escapeBonus += 2
        elif ghostPos[1] == pacPos[1] and self.checkWall(ghostPos, pacPos, map, False):

            turnsToCapture = math.ceil(dis / 2)

            # if self._is_valid_position((ghostPos[0], ghostPos[1] + 1), map):
            #     escapeBonus += 2
            # if self._is_valid_position((ghostPos[0], ghostPos[1] - 1), map):
            #     escapeBonus += 2
        else:  # This condition is true when Ghost is not on the same row or col with Pacman
        #     # => Pacman must turn to reach ghost
            xDiff = abs(ghostPos[0] - pacPos[0])
            yDiff = abs(ghostPos[1] - pacPos[1])

            if xDiff % 2 != 0 and yDiff % 2 != 0:
                turnsToCapture = math.ceil(dis / 2) + 1
            else:
                turnsToCapture = math.ceil(dis / 2)

            # if self._is_valid_position((ghostPos[0] + 1, ghostPos[1]), map):
            #     escapeBonus += 1
            # if self._is_valid_position((ghostPos[0] - 1, ghostPos[1]), map):
            #     escapeBonus += 1
            # if self._is_valid_position((ghostPos[0], ghostPos[1] + 1), map):
            #     escapeBonus += 1
            # if self._is_valid_position((ghostPos[0], ghostPos[1] - 1), map):
            #     escapeBonus += 1

        # return dis # Game not over
        # return turnsToCapture + escapeBonus
        return turnsToCapture
    
    def terminal(self, ghostPos, pacPos):
        if self.manHatDis(ghostPos, pacPos) < 2:    # Pacman wins
            return True
        return False    # Game not over

    def checkWall(self, ghostPos : tuple, pacPos : tuple, map : np.ndarray, sameRow : bool) -> bool:
        if sameRow:
            for i in range (min(ghostPos[1], pacPos[1]), max(ghostPos[1], pacPos[1]) + 1):
                if map[ghostPos[0], i] != 0: # There is a wall
                    return False
            return True
        for i in range (min(ghostPos[0], pacPos[0]), max(ghostPos[0], pacPos[0]) + 1):
            if map[i, ghostPos[1]] != 0: # There is a wall
                return False
        return True
            
        