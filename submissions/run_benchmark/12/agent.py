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

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np

import itertools
import heapq
import math

IS_DEBUG_LOG = False


class Node:
    def __init__(self, parent, action: Move, state: tuple[int, int], g_cost: int):
        self.parent = parent
        self.action = action
        self.state = state
        self.g_cost = g_cost

class PriorityQueue:
    def __init__(self):
        self.pq: list[tuple[int, int, Node]] = []
        self.counter = itertools.count()

    def push(self, node: Node, priority: int):
        heapq.heappush(self.pq, (priority, next(self.counter), node))

    def pop(self):
        if not self.pq:
            raise KeyError("Pop from an empty priority queue")
        priority, count, item = heapq.heappop(self.pq)
        return item

    def isEmpty(self):
        return len(self.pq) == 0


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

        self.cur_pd_count = 0
        self.patrol_destination = [
            # (1, 5),     # Top left
            # (1, 15),    # Top Right
            # (7, 10),    # Middle cage I
            # (11 ,10),   # Middle cage II
            # (15, 10),   # Middle cage III
            # (19, 10)    # Middle bottom
            # (10, 15),
            # (10, 5),
            # (4, 5),
            # (1, 1),
            # (7, 10),
            # (4, 15),
            # (1, 19)
            (10, 5),
            (10, 15),
            (1, 19),
            (4, 15),
            (7, 10),
            (1, 1),
            (4, 5),
            (19, 1),
            (19, 19),
            (16, 10)
        ]

    @classmethod
    def manhattan_dis(cls, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def a_star(self, my_position, enemy_position) -> list[Node] | None:
        """
        Return a list of nodes that represent the shortest path from my_position to enemy_position
        """
        if enemy_position is None:
            return []

        frontier = PriorityQueue()
        step = 0
        manhattan_dis = self.manhattan_dis(my_position, enemy_position)
        total_cost = step + manhattan_dis

        initial_node = Node(
            parent=None,
            action=None,
            state=my_position,
            g_cost=0
        )

        frontier.push(node=initial_node, priority=total_cost)

        # List of state
        explored = set()

        while True:
            if frontier.isEmpty():
                return []

            node = frontier.pop()

            if node.state == enemy_position:
                path = []

                while node.parent is not None:
                    path.append(node)
                    node = node.parent

                path.reverse()

                return path

            if node.state in explored:
                continue

            explored.add(node.state)

            direction = [
                (1, 0, Move.DOWN),
                (-1, 0, Move.UP),
                (0, 1, Move.RIGHT),
                (0, -1, Move.LEFT)
            ]

            for dx, dy, action in direction:
                next_tile = (node.state[0] + dx, node.state[1] + dy)

                if self._is_valid_position(next_tile, self.map_state) and next_tile not in explored:
                    new_g_cost = node.g_cost + 1

                    new_node = Node(
                        parent=node,
                        action=action,
                        state=next_tile,
                        g_cost=new_g_cost
                    )
                    total_cost = new_g_cost + self.manhattan_dis(next_tile, enemy_position)
                    frontier.push(node=new_node, priority=total_cost)

    def patrol(self, my_position):
        if my_position == self.patrol_destination[self.cur_pd_count]:
            # Update pd counter
            self.cur_pd_count = (self.cur_pd_count + 1) % len(self.patrol_destination)

        if IS_DEBUG_LOG:
            print(f"Patrolling to {self.patrol_destination[self.cur_pd_count]}")
        return self.a_star(my_position, self.patrol_destination[self.cur_pd_count])

    def take_nearest_patrol_des(self, my_pos):
        next_pd = 0
        smallest_mahdis = 999
        for i, pd in enumerate(self.patrol_destination):
            if self.manhattan_dis(my_pos, pd) < smallest_mahdis:
                next_pd = i
                smallest_mahdis = self.manhattan_dis(my_pos, pd)
        return next_pd

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
        list_of_action = []

        if self.last_known_enemy_pos == my_position:
            self.last_known_enemy_pos = None
            self.cur_pd_count = self.take_nearest_patrol_des(my_position)

        # Enemy spotted
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

            if IS_DEBUG_LOG:
                print("Enemy spotted!")
            list_of_action = self.a_star(my_position, enemy_position)
        # Lost sight so go to last spotted pos
        elif self.last_known_enemy_pos is not None:
            if IS_DEBUG_LOG:
                print("Lost sight of enemy, moving to last spotted destination")
            list_of_action = self.a_star(my_position, self.last_known_enemy_pos)
        # Patrol
        else:
            list_of_action = self.patrol(my_position)

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
        if not list_of_action:
            return []

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

        return not map_state[row, col] == 1


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

        #Belief dictionary, where the Ghost believes the Pacman is
        self.belief = {
            (15, 10): 1.0,
        }

        self.maxDepth = 6
        self.believePacAtStartPos = True
        # Dictionary of nodes, saves walkways and walls based on the Ghost's vision
        # If a node isn't in either sets, then it's an unknown node
        self.knowledgeBase = { 
            "known_walkways": set(),
            "known_walls": set(),
        }
        self.previous_pos = None

    def step(self, map_state: np.ndarray,  # Minmax
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
        self.updateKnowledge(my_position, map_state)

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.believePacAtStartPos = False
        elif self.believePacAtStartPos == True:   # Starting position of Pacman
            self.last_known_enemy_pos = (15, 10)

        self.updateBelief(my_position, enemy_position, map_state)

        if IS_DEBUG_LOG:
            print(self.belief)

        threat = self.last_known_enemy_pos or enemy_position

        best_move = Move.STAY
        best_val = -999
        alpha = -999
        beta = 999

        for move in [Move.UP, Move.DOWN, Move.RIGHT, Move.LEFT]:
            if not self._is_valid_move(my_position, move, map_state):
                continue

            next_my_pos = self.updatePos(my_position, move)

            val = self.minValue(map_state, next_my_pos, threat, 1, alpha, beta)

            if self.previous_pos is not None and self.previous_pos == next_my_pos:
                val -= 0.1

            if val > best_val:
                best_val = val
                best_move = move

            # elif val == best_val:

            alpha = max(alpha, best_val)
        self.previous_pos = my_position
        return best_move

    # Helper methods (you can add more)

    def updateKnowledge(self, myPos: tuple, map: np.ndarray): # 1 -> Wall, 0 -> walkway, -1 -> unseen
        # Evaluate rows ontop and below to Ghost to update its knowledge
        for i in range(1, 6, 1):
            checkPosUp = (myPos[0] - i, myPos[1])
            if self.positionValue(checkPosUp, map) == 1: # If it hits a wall, it stops updating and break the loop
                self.knowledgeBase["known_walls"].add(checkPosUp)
                break
            elif self.positionValue(checkPosUp, map) == 0: # If it hits an empty space, it can still see so keep looping
                self.knowledgeBase["known_walkways"].add(checkPosUp)
        for i in range(1, 6, 1):
            checkPosDown = (myPos[0] + i, myPos[1])
            if self.positionValue(checkPosDown, map) == 1: # If it hits a wall, it stops updating and break the loop
                self.knowledgeBase["known_walls"].add(checkPosDown)
                break
            elif self.positionValue(checkPosDown, map) == 0: # If it hits an empty space, it can still see so keep looping
                self.knowledgeBase["known_walkways"].add(checkPosDown)

        # Evaluate columns to the left and right of Ghost to update its knowledge.
        for i in range(1, 6, 1):
            checkPosLeft = (myPos[0], myPos[1] - i)
            if self.positionValue(checkPosLeft, map) == 1:
                self.knowledgeBase["known_walls"].add(checkPosLeft)
                break
            elif self.positionValue(checkPosLeft, map) == 0:
                self.knowledgeBase["known_walkways"].add(checkPosLeft)
        for i in range(1, 6, 1):
            checkPosRight = (myPos[0], myPos[1] + i)
            if self.positionValue(checkPosRight, map) == 1:
                self.knowledgeBase["known_walls"].add(checkPosRight)
                break
            elif self.positionValue(checkPosRight, map) == 0:
                self.knowledgeBase["known_walkways"].add(checkPosRight)

    def get_memory_path(self, map_state: np.ndarray, my_position: tuple):
        row, col = my_position
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return -1
        current_val = map_state[row][col]
        if current_val != -1:
            return current_val
        if my_position in self.knowledgeBase["known_walkways"]:
            return 0
        if my_position in self.knowledgeBase["known_walls"]:
            return 1
        return -1

    def updateBelief(self, myPos: tuple, pacPos: tuple, map: np.ndarray):
        if pacPos == None and len(self.belief) > 0: # Don't see Pacman -> Start predicting
            newBelief = {}
            for believeNodes in self.belief.copy():
                degrees = 0
                up = (believeNodes[0] - 1, believeNodes[1])
                down = (believeNodes[0] + 1, believeNodes[1])
                left = (believeNodes[0], believeNodes[1] - 1)
                right = (believeNodes[0], believeNodes[1] + 1)

                checkList = {
                                up : 0, 
                                down : 0, 
                                left : 0, 
                                right : 0
                            }

                for node in checkList.copy():
                    if node in self.knowledgeBase["known_walkways"]:
                        degrees += 1
                        # dis = self.manHatDis(myPos, node)
                        # checkList.update({node : dis})
                    else:
                        del checkList[node]
                if degrees == 0:
                    return
                baseProbability = 1 / degrees
                self.belief.clear()
                # maxNode = max(checkList, key = checkList.get)
                
                for node in checkList:
                #     deviation = (checkList[node] - checkList[maxNode]) / checkList[maxNode]
                    # self.belief.update({node : baseProbability - deviation})
                    newBelief.update({node: baseProbability})
            self.belief = newBelief.copy()
            # self.normalize()
        else:
            self.belief.clear()
            self.belief.update({pacPos: 1.0})

    def normalize(self):
        currTotalProb = 0
        for prob in self.belief.values():
            currTotalProb += prob
        alpha = 1 / currTotalProb
        for node in self.belief:
            probability = self.belief[node]
            self.belief.update({node: probability * alpha})
    
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

        return self.get_memory_path(map_state, pos) != 1

    def positionValue(self, pos: tuple, map_state: np.ndarray): # Like _is_valid_position but returns the value of that position
        row, col = pos
        height, width = map_state.shape

        if row < 0 or row >= height or col < 0 or col >= width:
            return -1

        return map_state[row, col]

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

        return self.get_memory_path(map_state, pos) != 1

    def minValue(self, map: np.ndarray, myPos: tuple, threatPos: tuple, depth: int, alpha: int, beta: int):
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
            val = min(val,
                      self.maxValue(map, myPos, self.updatePos(threatPos, move[0], move[1]), depth + 1, alpha, beta))
            if val <= alpha:
                return val

            beta = min(beta, val)  # giá trị tốt nhất (nhỏ nhất) mà pacman có thể đạt được

        return val

    def maxValue(self, map: np.ndarray, myPos: tuple, threatPos: tuple, depth: int, alpha: int, beta: int):
        """
        Ghost value
        """
        if self.terminal(myPos, threatPos) or depth == self.maxDepth:
            return self.utility(myPos, threatPos, map, depth)  # Not GameOver

        val = -999
        avai_actions = set()

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(myPos, move, map):
                avai_actions.add(move)

        for move in avai_actions:
            val = max(val, self.minValue(map, self.updatePos(myPos, move), threatPos, depth + 1, alpha, beta))
            if val >= beta:
                return val

            alpha = max(alpha, val)  # giá trị tốt nhất (lớn nhất) mà ghost có thể đạt được

        return val

    def manHatDis(self, myPos: tuple, threatPos: tuple):
        return abs(myPos[0] - threatPos[0]) + abs(myPos[1] - threatPos[1])

    def updatePos(self, pos: tuple, move: Move, steps=1):
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

        if dis < 2:  # Pacman wins
            return -2 * self.maxDepth + depth

        score = 0

        if ghostPos[0] == pacPos[0] and self.checkWall(map, ghostPos, pacPos, True):
            score = 1
        elif ghostPos[1] == pacPos[1] and self.checkWall(map, ghostPos, pacPos, False):
            score = 1

        return dis + score

    def terminal(self, ghostPos, pacPos):
        if self.manHatDis(ghostPos, pacPos) < 2:  # Pacman wins
            return True
        return False  # Game not over

    def checkWall(self, map: np.ndarray, ghostPos: tuple, pacPos: tuple, sameRow: bool) -> bool:
        if sameRow:
            for i in range(min(ghostPos[1], pacPos[1]), max(ghostPos[1], pacPos[1]) + 1):
                if self.get_memory_path(map, (ghostPos[0], i)) == 1:  # There is a wall
                    return False
            return True
        for i in range(min(ghostPos[0], pacPos[0]), max(ghostPos[0], pacPos[0]) + 1):
            if self.get_memory_path(map, (i, ghostPos[1])) == 1: # There is a wall
                return False
        return True