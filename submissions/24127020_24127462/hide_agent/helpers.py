from environment import Move
import numpy as np
from collections import deque

# BASIC HELPERS

def is_valid_position(pos: tuple, map_state):
   """Check if position is valid (not wall or unseen boundaries)."""
   row, col = pos
   height, width = map_state.shape
   
   # Check bounds
   if row < 0 or row >= height or col < 0 or col >= width:
      return False
   
   # Check not a wall 
   # (Optional: treat unseen (-1) carefully based on your strategy!)
   return map_state[row, col] == 0

def translate_move(current: tuple, next_pos: tuple) -> Move:
   dr = next_pos[0] - current[0]
   dc = next_pos[1] - current[1]
   return {
      (-1, 0): Move.UP,
      (1, 0): Move.DOWN,
      (0, -1): Move.LEFT,
      (0, 1): Move.RIGHT,
   }.get((dr, dc), Move.STAY)

def apply_move(pos: tuple, move) -> tuple:
   """Apply a move to a position, return new position."""
   delta_row, delta_col = move.value
   return (pos[0] + delta_row, pos[1] + delta_col)

def get_neighbors(pos: tuple, map_state: np.ndarray) -> list[tuple[int, int]]:
   """Get all valid neighboring positions and their moves."""
   neighbors = []
   
   for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
      next_pos = apply_move(pos, move)
      if is_valid_position(next_pos, map_state):
         neighbors.append(next_pos)
   
   return neighbors

# CUSTOM HELPERS
def list_dead_end_cells(map_state: np.ndarray) -> list[tuple]:
   dead_end_cells = set()
   rows, cols = map_state.shape

   for row in range(rows):
      for col in range(cols):
         start = (row, col)
         if not is_valid_position(start, map_state):
               continue
         if len(get_neighbors(start, map_state)) != 1:
               continue

         # Found a dead end, walk until we hit a junction
         prev, cur = None, start
         while True:
               dead_end_cells.add(cur)
               neighbors = [n for n in get_neighbors(cur, map_state) if n != prev]
               if not neighbors:
                  break
               nxt = neighbors[0]
               if len(get_neighbors(nxt, map_state)) >= 3:
                  break  # nxt is a junction, stop
               prev, cur = cur, nxt

   return list(dead_end_cells)


def build_dead_end_exit_map(map_state: np.ndarray) -> dict[tuple, tuple]:
   dead_end_cells = set(list_dead_end_cells(map_state))
   exit_map = {}

   for cell in dead_end_cells:
      for neighbor in get_neighbors(cell, map_state):
         if neighbor not in dead_end_cells:
               gate = neighbor
               stack = [cell]
               visited = set()

               while stack:
                  cur = stack.pop()
                  if cur in visited:
                     continue
                  visited.add(cur)
                  exit_map[cur] = gate
                  for nxt in get_neighbors(cur, map_state):
                     if nxt in dead_end_cells:
                           stack.append(nxt)

   return exit_map

def _print_map(file, map_state, dead_end_exit) -> None:
   dead_ends = set(dead_end_exit.keys())
   gates = set(dead_end_exit.values())

   rows, cols = map_state.shape
   
   for row in range(rows):
      line = []

      for col in range(cols):
         cell = (row, col)

         if map_state[row, col] != 0:
            line.append("#")
         elif cell in gates:
            line.append("O")
         elif cell in dead_ends:
            line.append("X")
         else:
            line.append(" ")

      file.write(" ".join(line) + "\n")
