# hide_agent/core.py

from collections import deque
from heapq import heappop, heappush
from environment import Move


INF = 10 ** 9

CAPTURE_DISTANCE = 2

PACMAN_MOVES = [
   Move.UP,
   Move.DOWN,
   Move.LEFT,
   Move.RIGHT,
]

GHOST_MOVES = [
   Move.UP,
   Move.DOWN,
   Move.LEFT,
   Move.RIGHT,
   Move.STAY,
]

def manhattan(a, b):
   return abs(a[0] - b[0]) + abs(a[1] - b[1])

def next_position(pos: tuple, move: Move) -> tuple:
   dr, dc = move.value
   return (pos[0] + dr, pos[1] + dc)

def is_valid_position(pos: tuple, map_state) -> bool:
   row, col = pos
   height, width = map_state.shape
   return 0 <= row < height and 0 <= col < width and map_state[pos[0], pos[1]] == 0

def is_capture(pacman_pos: tuple[int, int], ghost_pos: tuple[int, int]) -> bool:
   return manhattan(pacman_pos, ghost_pos) < CAPTURE_DISTANCE

def legal_ghost_moves(pos: tuple, map_state) -> list[Move]:
   moves = []

   for move in GHOST_MOVES:
      new_pos = next_position(pos, move)

      if is_valid_position(new_pos, map_state):
         moves.append(move)

   if not moves:
      return [Move.STAY]

   return moves


def legal_ghost_positions(pos: tuple, map_state) -> list[tuple]:
   return [
      next_position(pos, move)
      for move in legal_ghost_moves(pos, map_state)
   ]


def legal_pacman_actions(
   pos: tuple,
   map_state,
   pacman_speed: int,
   allow_stay: bool = False,
) -> list[tuple[Move, int]]:
   actions = []
   pacman_speed = max(1, int(pacman_speed))

   for move in PACMAN_MOVES:
      current = pos

      for steps in range(1, pacman_speed + 1):
         current = next_position(current, move)

         if not is_valid_position(current, map_state):
               break

         actions.append((move, steps))

   if allow_stay:
      actions.append((Move.STAY, 1))

   if not actions:
      actions.append((Move.STAY, 1))

   return actions


def apply_pacman_action(
   pos: tuple,
   action: tuple[Move, int],
   map_state,
) -> tuple:
   move, steps = action

   if move == Move.STAY:
      return pos

   current = pos

   for _ in range(max(1, int(steps))):
      new_pos = next_position(current, move)

      if not is_valid_position(new_pos, map_state):
         break

      current = new_pos

   return current


def bfs_distances(start: tuple, map_state) -> dict[tuple, int]:
   if not is_valid_position(start, map_state):
      return {}

   distances = {start: 0}
   queue = deque([start])

   while queue:
      current = queue.popleft()

      for move in PACMAN_MOVES:
         new_pos = next_position(current, move)

         if is_valid_position(new_pos, map_state) and new_pos not in distances:
               distances[new_pos] = distances[current] + 1
               queue.append(new_pos)

   return distances


def pacman_turn_distances(
   start: tuple,
   map_state,
   pacman_speed: int,
) -> dict[tuple, int]:
   """
   BFS where one edge = one Pacman turn.

   This matters because Pacman can move multiple cells in one straight
   direction per turn.
   """
   if not is_valid_position(start, map_state):
      return {}

   distances = {start: 0}
   queue = deque([start])

   while queue:
      current = queue.popleft()

      for action in legal_pacman_actions(current, map_state, pacman_speed):
         new_pos = apply_pacman_action(current, action, map_state)

         if new_pos not in distances:
               distances[new_pos] = distances[current] + 1
               queue.append(new_pos)

   return distances


def capture_zone(ghost_pos: tuple, map_state) -> list[tuple]:
   """
   Pacman captures when Manhattan distance < 2.

   So Pacman does not need to stand exactly on the Ghost.
   The dangerous cells are:
   - Ghost's own cell
   - Adjacent walkable cells
   """
   cells = []

   if is_valid_position(ghost_pos, map_state):
      cells.append(ghost_pos)

   for move in PACMAN_MOVES:
      pos = next_position(ghost_pos, move)

      if is_valid_position(pos, map_state):
         cells.append(pos)

   return cells


def capture_turn_distance(
   pacman_pos: tuple[int, int],
   ghost_pos: tuple[int ,int],
   map_state,
   pacman_speed: int,
) -> int:
   """
   Minimum number of Pacman turns needed to reach Ghost's capture zone,
   assuming Ghost stays at ghost_pos.

   This is a point-to-goal query, so A* avoids building the complete
   distance map that the previous BFS implementation produced. One edge
   still represents one Pacman turn, including every legal speed choice.
   """
   if not is_valid_position(pacman_pos, map_state):
      return INF

   goals = set(capture_zone(ghost_pos, map_state))

   if not goals:
      return INF

   if pacman_pos in goals:
      return 0

   pacman_speed = max(1, int(pacman_speed))

   def heuristic(pos: tuple[int, int]) -> int:
      """Admissible lower bound on turns to any capture-zone cell."""
      cell_distance = min(
         manhattan(pos, goal)
         for goal in goals
      )

      return (
         cell_distance + pacman_speed - 1
      ) // pacman_speed

   best_turns = {pacman_pos: 0}
   frontier = [(
      heuristic(pacman_pos),
      0,
      pacman_pos,
   )]

   while frontier:
      _, turns, current = heappop(frontier)

      if turns != best_turns.get(current):
         continue

      if current in goals:
         return turns

      next_turns = turns + 1

      for action in legal_pacman_actions(
         current,
         map_state,
         pacman_speed,
      ):
         new_pos = apply_pacman_action(
            current,
            action,
            map_state,
         )

         if next_turns >= best_turns.get(new_pos, INF):
            continue

         best_turns[new_pos] = next_turns
         priority = next_turns + heuristic(new_pos)
         heappush(
            frontier,
            (priority, next_turns, new_pos),
         )

   return INF


def exit_count(pos: tuple, map_state) -> int:
   count = 0

   for move in PACMAN_MOVES:
      new_pos = next_position(pos, move)

      if is_valid_position(new_pos, map_state):
         count += 1

   return count


def dead_end_depth(pos: tuple, map_state, max_depth: int = 8) -> int:
   """
   Distance from pos to the nearest junction.

   Higher value means the Ghost is deeper inside a corridor/dead-end.
   Lower is better in panic mode.
   """
   if not is_valid_position(pos, map_state):
      return max_depth + 1

   queue = deque([(pos, 0)])
   visited = {pos}

   while queue:
      current, depth = queue.popleft()

      if exit_count(current, map_state) >= 3:
         return depth

      if depth >= max_depth:
         return max_depth + 1

      for new_pos in legal_ghost_positions(current, map_state):
         if new_pos != current and new_pos not in visited:
               visited.add(new_pos)
               queue.append((new_pos, depth + 1))

   return max_depth + 1
